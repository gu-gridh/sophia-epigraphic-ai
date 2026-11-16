#!/usr/bin/env python3
"""
Zero-shot comparison of pretrained text-recognition models on real Saint Sophia images.

Backends
    trocr       any TrOCR checkpoint (VisionEncoderDecoder)                  transformers 4.x/5.x
    lightonocr  LightOnOCR-2 (+ optional LoRA adapter, e.g. the OCS one)      transformers >= 5, peft
    qwenvl      Qwen2/2.5/3-VL family incl. CHURRO                            transformers >= 4.57
    kraken      a Kraken recognition model, image treated as one line       kraken >= 7 (own venv)

Self-contained (no project imports) so it can run in a separate environment.

    python zero_shot_compare.py --backend lightonocr \\
        --model lightonai/LightOnOCR-2-1B-base \\
        --adapter wjbmattingly/LightOnOCR-2-1B-old-church-slavonic-line \\
        --csv data/benchmark_real/benchmark.csv --views drawing,photo \\
        --out results/lightonocr_ocs.csv

CER is reported three ways (see `normalise`): raw, normalised, normalised without spaces.
Per-sample CER is capped at 1.0 for the mean so a single runaway output cannot dominate.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

from PIL import Image, ImageOps

# ------------------------------------------------------------------ metrics

EDITORIAL = set('[](){}⟨⟩<>/|\\_·.,:;!?"\'«»-–—=+*~^`')


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def normalise(s, keep_spaces=True):
    """Lower-case; drop accents/titla/combining marks, Private-Use editorial codes and brackets."""
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) not in ('Mn', 'Me', 'Co', 'Cf'))
    s = ''.join(c for c in s.lower() if c not in EDITORIAL)
    s = ' '.join(s.split()) if keep_spaces else ''.join(s.split())
    return unicodedata.normalize('NFC', s)


def scores(ref, hyp):
    return {
        'cer_raw': cer(ref, hyp),
        'cer_norm': cer(normalise(ref), normalise(hyp)),
        'cer_norm_nospace': cer(normalise(ref, False), normalise(hyp, False)),
    }


# ------------------------------------------------------------------ image prep

def load_view(path, view, invert_drawings=True):
    img = Image.open(path).convert('RGB')
    if view == 'drawing' and invert_drawings:
        # Korniienko drawings are white strokes on black; OCR models expect dark ink on light
        grey = ImageOps.grayscale(img)
        if sum(grey.resize((64, 16)).getdata()) / (64 * 16) < 110:
            img = ImageOps.invert(img)
    img.thumbnail((2048, 2048))
    return img


# ------------------------------------------------------------------ backends

def pick_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda', torch.bfloat16
    if torch.backends.mps.is_available():
        return 'mps', torch.float16
    return 'cpu', torch.float32


class TrOCRBackend:
    def __init__(self, args):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.device, _ = pick_device()
        self.proc = TrOCRProcessor.from_pretrained(args.model)
        self.model = VisionEncoderDecoderModel.from_pretrained(args.model).to(self.device).eval()
        self.max_new = args.max_new_tokens

    def __call__(self, img):
        pv = self.proc(images=img, return_tensors='pt').pixel_values.to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(pv, num_beams=1, max_length=self.max_new, no_repeat_ngram_size=0)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0]


class LightOnOCRBackend:
    def __init__(self, args):
        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        self.torch = torch
        self.device, self.dtype = pick_device()
        if self.device == 'mps':
            self.dtype = torch.float32  # half precision is unreliable for this model on MPS
        self.proc = LightOnOcrProcessor.from_pretrained(args.adapter or args.model)
        model = LightOnOcrForConditionalGeneration.from_pretrained(args.model, dtype=self.dtype)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.max_new = args.max_new_tokens
        self.longest_edge = args.longest_edge

    def __call__(self, img):
        messages = [{'role': 'user', 'content': [{'type': 'image'}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[[img]], return_tensors='pt', padding=True,
                           size={'longest_edge': self.longest_edge}).to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                      repetition_penalty=1.1)
        return self.proc.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


class QwenVLBackend:
    PROMPT = ('Transcribe the text in this image exactly as written, character by character. '
              'It is a medieval graffito (Church Slavonic, Old Ukrainian, Greek, Latin, Polish or '
              'Armenian). Output only the transcription.')

    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device, self.dtype = pick_device()
        self.proc = AutoProcessor.from_pretrained(args.model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=self.dtype).to(self.device).eval()
        self.max_new = args.max_new_tokens
        self.prompt = args.prompt or self.PROMPT

    def __call__(self, img):
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': self.prompt}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img], return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                      repetition_penalty=1.05)
        return self.proc.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


class KrakenBackend:
    """Runs the kraken CLI (from --kraken_bin) on the whole image as a single line."""

    def __init__(self, args):
        self.bin = args.kraken_bin
        self.model = args.model
        self.tmp = tempfile.mkdtemp()

    def __call__(self, img):
        src = os.path.join(self.tmp, 'in.png')
        dst = os.path.join(self.tmp, 'out.txt')
        # kraken expects a line image of moderate height
        h = 120
        img = img.resize((max(1, int(img.width * h / img.height)), h))
        img.save(src)
        r = subprocess.run([self.bin, '-i', src, dst, 'binarize', 'segment', '-bl', 'ocr', '-m', self.model],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            r = subprocess.run([self.bin, '-i', src, dst, 'ocr', '-m', self.model, '--no-segmentation'],
                               capture_output=True, text=True)
        try:
            return ' '.join(open(dst, encoding='utf-8').read().split())
        except FileNotFoundError:
            sys.stderr.write(r.stderr[-500:])
            return ''


BACKENDS = {'trocr': TrOCRBackend, 'lightonocr': LightOnOCRBackend,
            'qwenvl': QwenVLBackend, 'kraken': KrakenBackend}


# ------------------------------------------------------------------ main

def summarise(rows, view):
    sel = [r for r in rows if r['view'] == view]
    if not sel:
        return
    out = [f'{view:8s} n={len(sel):3d}']
    for k in ('cer_raw', 'cer_norm', 'cer_norm_nospace'):
        vals = [float(r[k]) for r in sel]
        out.append(f'{k}={statistics.mean(min(v, 1.0) for v in vals):6.1%} (median {statistics.median(vals):5.1%})')
    exact = sum(normalise(r['target']) == normalise(r['prediction']) for r in sel)
    out.append(f'exact(norm)={exact}/{len(sel)}')
    print('  ' + '  '.join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', required=True, choices=list(BACKENDS))
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', default=None, help='PEFT/LoRA adapter (lightonocr)')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--root', default=None, help='image paths are relative to this (default: CSV folder)')
    ap.add_argument('--views', default='drawing,photo')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--max_new_tokens', type=int, default=160)
    ap.add_argument('--longest_edge', type=int, default=1024)
    ap.add_argument('--no_invert', action='store_true', help='keep drawings white-on-black')
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--kraken_bin', default='kraken')
    args = ap.parse_args()

    root = args.root or os.path.dirname(os.path.abspath(args.csv))
    with open(args.csv, newline='', encoding='utf-8') as f:
        items = list(csv.DictReader(f))[:args.limit]
    views = [v.strip() for v in args.views.split(',') if v.strip()]

    t0 = time.time()
    backend = BACKENDS[args.backend](args)
    print(f'Loaded {args.backend}:{args.model}{" + " + args.adapter if args.adapter else ""} '
          f'in {time.time() - t0:.0f}s', flush=True)

    rows = []
    t0 = time.time()
    for it in items:
        for view in views:
            rel = it.get(f'korniienko_{view}') or it.get(f'{view}_image')
            if not rel:
                continue
            img = load_view(os.path.join(root, rel), view, not args.no_invert)
            try:
                pred = backend(img)
            except Exception as e:  # keep going; record the failure
                pred = ''
                sys.stderr.write(f'{it.get("id")} {view}: {type(e).__name__}: {e}\n')
            ref = it['transcription_clean']
            rows.append({'id': it.get('id'), 'title': it.get('title'), 'language': it.get('language_name'),
                         'view': view, 'target': ref, 'prediction': pred, **scores(ref, pred)})
    secs = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{args.backend}:{args.model}{" + adapter" if args.adapter else ""}  '
          f'({len(rows)} images, {secs / max(1, len(rows)):.1f}s/image) -> {args.out}')
    for v in views:
        summarise(rows, v)


if __name__ == '__main__':
    main()
