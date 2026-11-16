#!/usr/bin/env python3
"""
LoRA fine-tuning of an OCR vision-language model on Saint Sophia inscriptions
=============================================================================

Default: LightOnOCR-2-1B, starting from the Old Church Slavonic line adapter
(wjbmattingly/LightOnOCR-2-1B-old-church-slavonic-line), which reads real graffiti
formulas zero-shot. Also works for the Qwen-VL family (incl. CHURRO) with --family qwenvl.

Every (inscription, view) pair is one training example, so with --views drawing,photo
each inscription contributes its Korniienko drawing and photo separately.

Splits
    --cv_csv FULL.csv --folds 5 --fold K   reproduces cross_validate.py's language-stratified
                                           folds and inner model-selection split exactly
    --train_csv / --val_csv / --test_csv   fixed split

Needs transformers >= 5 and peft. Fits a 24 GB GPU (RTX 4090) in bf16 with batch 4.

    python scripts/pretrained_candidates/finetune_vlm_lora.py \\
        --cv_csv data/complete_dataset.csv --data_dir data --fold 1 \\
        --views drawing,photo --epochs 8 --out runs/lightonocr_fold1
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zero_shot_compare import normalise, scores  # noqa: E402

VIEW_COLUMNS = {
    'drawing': ['korniienko_drawing'],
    'photo': ['korniienko_photo', 'iiif_crop'],
    'original': ['original_image'],
    'normal': ['normal_image'],
}


# ------------------------------------------------------------------ data / splits

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def clean_text(t):
    import html
    import unicodedata
    t = str(t or '')
    for _ in range(3):
        u = html.unescape(t)
        if u == t:
            break
        t = u
    return unicodedata.normalize('NFC', t).strip()


def stratified_folds(rows, n_folds, seed):
    """Same assignment as cross_validate.create_stratified_folds (np.random, per language)."""
    np.random.seed(seed)
    langs = [r.get('language_name') or 'Unknown' for r in rows]
    fold = [-1] * len(rows)
    seen = []
    for l in langs:
        if l not in seen:
            seen.append(l)
    for lang in seen:
        idx = [i for i, l in enumerate(langs) if l == lang]
        np.random.shuffle(idx)
        for k, i in enumerate(idx):
            fold[i] = k % n_folds
    return [([r for r, f in zip(rows, fold) if f != k], [r for r, f in zip(rows, fold) if f == k])
            for k in range(n_folds)]


def make_splits(args):
    if args.cv_csv:
        rows = [r for r in read_csv(args.cv_csv)
                if str(r.get('transcription_clean', '')).strip() not in ('', 'nan')]
        train, test = stratified_folds(rows, args.folds, args.seed)[args.fold - 1]
        n_inner = max(2, int(round(1.0 / args.inner_val_frac)))
        fit, select = stratified_folds(train, n_inner, args.seed + args.fold - 1)[0]
        return fit, select, test
    return (read_csv(args.train_csv), read_csv(args.val_csv),
            read_csv(args.test_csv) if args.test_csv else [])


def target_text(raw, args):
    t = clean_text(raw)
    if args.line_breaks:
        t = t.replace(' | ', '\n').replace(' / ', '\n')
    return t


def examples(rows, args, data_dir):
    out = []
    for r in rows:
        text = target_text(r.get('transcription_clean'), args)
        if not text or text == 'nan':
            continue
        for view in args.views:
            for col in VIEW_COLUMNS[view]:
                rel = (r.get(col) or '').strip()
                if rel and rel != 'nan':
                    path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
                    if os.path.exists(path):
                        out.append({'path': path, 'view': view, 'text': text,
                                    'id': r.get('id'), 'language': r.get('language_name')})
                        break
    return out


def load_image(ex, augment, rng):
    img = Image.open(ex['path']).convert('RGB')
    if ex['view'] == 'drawing':
        grey = ImageOps.grayscale(img).resize((64, 16))
        if sum(grey.getdata()) / (64 * 16) < 110:  # white-on-black tracing -> dark ink on light
            img = ImageOps.invert(img)
    if augment:
        img = img.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(255, 255, 255) if ex['view'] == 'drawing' else None)
        if rng.random() < 0.5:
            from PIL import ImageEnhance
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img.thumbnail((2048, 2048))
    return img


# ------------------------------------------------------------------ model wrappers

class OCRModel:
    def __init__(self, args, device, dtype):
        from peft import LoraConfig, PeftModel, get_peft_model
        self.args, self.device, self.dtype = args, device, dtype
        if args.family == 'lightonocr':
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
            self.proc = LightOnOcrProcessor.from_pretrained(args.init_adapter or args.model)
            base = LightOnOcrForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
            targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.proc = AutoProcessor.from_pretrained(args.model)
            base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)
            targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        if args.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        if args.init_adapter:
            self.model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        else:
            cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules=targets, task_type='CAUSAL_LM')
            self.model = get_peft_model(base, cfg)
        self.model.to(device)
        self.model.print_trainable_parameters()
        self.prompt_text = args.prompt

    def _messages(self, with_answer=None):
        content = [{'type': 'image'}]
        if self.prompt_text:
            content.append({'type': 'text', 'text': self.prompt_text})
        msgs = [{'role': 'user', 'content': content}]
        if with_answer is not None:
            msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': with_answer}]})
        return msgs

    def _proc_kwargs(self):
        if self.args.family == 'lightonocr':
            return {'size': {'longest_edge': self.args.longest_edge}}
        return {}

    def batch(self, images, texts):
        """Tokenise prompt+answer; loss only on the answer (and EOS)."""
        prompts = [self.proc.apply_chat_template(self._messages(), tokenize=False, add_generation_prompt=True)
                   for _ in texts]
        eos = self.proc.tokenizer.eos_token or ''
        full = [p + t + eos for p, t in zip(prompts, texts)]
        img_arg = [[im] for im in images] if self.args.family == 'lightonocr' else images
        enc = self.proc(text=full, images=img_arg, return_tensors='pt', padding=True, **self._proc_kwargs())
        p_enc = self.proc(text=prompts, images=img_arg, return_tensors='pt', padding=True, **self._proc_kwargs())
        labels = enc['input_ids'].clone()
        pad_id = self.proc.tokenizer.pad_token_id
        left_pad = self.proc.tokenizer.padding_side == 'left'
        for i in range(labels.size(0)):
            n_prompt = int(p_enc['attention_mask'][i].sum())
            n_full = int(enc['attention_mask'][i].sum())
            start = labels.size(1) - n_full if left_pad else 0
            labels[i, :start + n_prompt] = -100
        labels[enc['input_ids'] == pad_id] = -100
        enc['labels'] = labels
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if 'pixel_values' in enc:
            enc['pixel_values'] = enc['pixel_values'].to(self.dtype)
        return enc

    @torch.no_grad()
    def generate(self, images, max_new_tokens):
        prompts = [self.proc.apply_chat_template(self._messages(), tokenize=False, add_generation_prompt=True)
                   for _ in images]
        img_arg = [[im] for im in images] if self.args.family == 'lightonocr' else images
        enc = self.proc(text=prompts, images=img_arg, return_tensors='pt', padding=True, **self._proc_kwargs())
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if 'pixel_values' in enc:
            enc['pixel_values'] = enc['pixel_values'].to(self.dtype)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  num_beams=self.args.num_beams, repetition_penalty=1.05)
        return [self.proc.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for o in out]


# ------------------------------------------------------------------ train / eval

def evaluate(model, exs, args, rng, desc):
    model.model.eval()
    rows = []
    for i in range(0, len(exs), args.eval_batch_size):
        chunk = exs[i:i + args.eval_batch_size]
        preds = model.generate([load_image(e, False, rng) for e in chunk], args.max_new_tokens)
        for e, p in zip(chunk, preds):
            ref = e['text'].replace('\n', ' ')
            p = ' '.join(p.split())
            rows.append({'id': e['id'], 'language': e['language'], 'view': e['view'],
                         'target': ref, 'prediction': p, **scores(ref, p)})
    summary = {}
    for v in sorted({r['view'] for r in rows}) + ['all']:
        sel = [r for r in rows if v == 'all' or r['view'] == v]
        if sel:
            summary[v] = {k: statistics.mean(min(r[k], 1.0) for r in sel)
                          for k in ('cer_raw', 'cer_norm', 'cer_norm_nospace')}
            summary[v]['exact_norm'] = sum(normalise(r['target']) == normalise(r['prediction']) for r in sel) / len(sel)
            summary[v]['n'] = len(sel)
    line = '  '.join(f"{v}: raw {s['cer_raw']:.1%} norm {s['cer_norm']:.1%} exact {s['exact_norm']:.0%}"
                     for v, s in summary.items())
    print(f'  [{desc}] {line}', flush=True)
    return summary, rows


def write_rows(path, rows):
    if rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', choices=['lightonocr', 'qwenvl'], default='lightonocr')
    ap.add_argument('--model', default='lightonai/LightOnOCR-2-1B-base')
    ap.add_argument('--init_adapter', default='wjbmattingly/LightOnOCR-2-1B-old-church-slavonic-line',
                    help="LoRA adapter to continue from ('' for a fresh LoRA)")
    ap.add_argument('--prompt', default=None, help='text prompt (LightOnOCR uses none; Qwen-VL needs one)')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--cv_csv', default=None)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--inner_val_frac', type=float, default=0.1)
    ap.add_argument('--train_csv'); ap.add_argument('--val_csv'); ap.add_argument('--test_csv')
    ap.add_argument('--views', default='drawing,photo')
    ap.add_argument('--line_breaks', action='store_true', help="train on ' | ' and ' / ' as new lines")
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--eval_batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--longest_edge', type=int, default=1024)
    ap.add_argument('--max_new_tokens', type=int, default=160)
    ap.add_argument('--num_beams', type=int, default=1)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--gradient_checkpointing', action='store_true')
    ap.add_argument('--max_train', type=int, default=None, help='debug: cap training examples')
    ap.add_argument('--max_eval', type=int, default=None, help='debug: cap eval examples')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    args.views = [v.strip() for v in args.views.split(',') if v.strip()]
    args.init_adapter = args.init_adapter or None
    if args.family == 'qwenvl' and not args.prompt:
        args.prompt = 'Transcribe the text in this image exactly. Output only the transcription.'

    if torch.cuda.is_available():
        device, dtype = 'cuda', torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = 'mps', torch.float32
    else:
        device, dtype = 'cpu', torch.float32
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'config.json'), 'w'), indent=2)

    fit, select, test = make_splits(args)
    train_ex = examples(fit, args, args.data_dir)[:args.max_train]
    select_ex = examples(select, args, args.data_dir)[:args.max_eval]
    test_ex = examples(test, args, args.data_dir)[:args.max_eval]
    print(f'Inscriptions: fit {len(fit)}, select {len(select)}, test {len(test)} | '
          f'examples: train {len(train_ex)}, select {len(select_ex)}, test {len(test_ex)} '
          f'(views {args.views}) | device {device}', flush=True)

    model = OCRModel(args, device, dtype)
    params = [p for p in model.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_ex) / (args.batch_size * args.grad_accum))
    total = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    _, base_rows = evaluate(model, select_ex, args, rng, 'select, before training')
    best, best_epoch, bad = float('inf'), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.model.train()
        rng.shuffle(train_ex)
        t0, losses = time.time(), []
        for step, i in enumerate(range(0, len(train_ex), args.batch_size)):
            chunk = train_ex[i:i + args.batch_size]
            enc = model.batch([load_image(e, True, rng) for e in chunk], [e['text'] for e in chunk])
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                loss = model.model(**enc).loss / args.grad_accum
            loss.backward()
            losses.append(loss.item() * args.grad_accum)
            if (step + 1) % args.grad_accum == 0 or i + args.batch_size >= len(train_ex):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        summary, _ = evaluate(model, select_ex, args, rng, f'select, epoch {epoch}')
        score = summary['all']['cer_norm']
        history.append({'epoch': epoch, 'loss': statistics.mean(losses), 'select': summary,
                        'minutes': (time.time() - t0) / 60})
        print(f'Epoch {epoch}: loss {statistics.mean(losses):.3f}, select CER(norm) {score:.1%}, '
              f'{(time.time() - t0) / 60:.1f} min', flush=True)
        if score < best:
            best, best_epoch, bad = score, epoch, 0
            model.model.save_pretrained(os.path.join(args.out, 'best_adapter'))
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f'Early stopping (best epoch {best_epoch})')
                break

    result = {'best_epoch': best_epoch, 'select_cer_norm': best, 'history': history}
    if test_ex:
        # load the best epoch's LoRA weights as a separate adapter and switch to it
        best_dir = os.path.join(args.out, 'best_adapter')
        if best_epoch and os.path.isdir(best_dir):
            model.model.load_adapter(best_dir, adapter_name='best')
            model.model.set_adapter('best')
        test_summary, test_rows = evaluate(model, test_ex, args, rng, 'TEST (best epoch)')
        result['test'] = test_summary
        write_rows(os.path.join(args.out, 'test_predictions.csv'), test_rows)
    json.dump(result, open(os.path.join(args.out, 'results.json'), 'w'), indent=2)
    print(f'Done. Best epoch {best_epoch}; results in {args.out}/results.json')


if __name__ == '__main__':
    main()
