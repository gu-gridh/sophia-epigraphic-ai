"""
Shared sequence helpers for training and evaluation.

- shift_tokens:       next-token training targets (decoder input = ids[:-1], target = ids[1:])
- build_image_inputs: model image kwargs that respect the --use_rti / --use_korniienko flags,
                      with optional image ablation (zero / shuffle) as a sanity check
- language_inputs:    optional gold language conditioning (off by default)
- greedy_generate:    autoregressive decoding from images only (no ground truth is fed in)
- decode_ids:         token ids -> text
"""

import torch

ABLATION_MODES = ('none', 'zero', 'shuffle')


def get_special_ids(tokenizer):
    """Return (sos_id, eos_id, pad_id) for the character tokenizer or a HF tokenizer."""
    if hasattr(tokenizer, 'char_to_idx'):
        c = tokenizer.char_to_idx
        return c['<SOS>'], c['<EOS>'], c['<PAD>']
    return tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id


def shift_tokens(input_ids, attention_mask=None):
    """Teacher forcing with a one-step shift: predict token t+1 from tokens <= t."""
    decoder_input = input_ids[:, :-1]
    target = input_ids[:, 1:]
    decoder_mask = attention_mask[:, :-1] if attention_mask is not None else None
    return decoder_input, decoder_mask, target


def _ablate(t, mode):
    if t is None or mode == 'none':
        return t
    if mode == 'zero':
        return torch.zeros_like(t)
    if mode == 'shuffle':
        # Give every sample the images of a different sample in the batch
        # (has no effect when the batch holds a single sample).
        return torch.roll(t, shifts=1, dims=0)
    raise ValueError(f"Unknown image ablation mode: {mode}")


def build_image_inputs(batch, device, use_rti=True, use_korniienko=True, ablation='none'):
    """
    Image kwargs for model.forward / model.encode. Disabled modalities are passed as None;
    `modality_mask` tells the model which images actually exist for each sample.
    """
    def get(key, enabled):
        if not enabled or batch.get(key) is None:
            return None
        return _ablate(batch[key].to(device), ablation)

    def present(key, enabled):
        if not enabled or batch.get(key) is None:
            return None
        # under 'shuffle' the flags travel with their (shuffled) images
        return _ablate(batch[key].to(device), 'shuffle' if ablation == 'shuffle' else 'none')

    return {
        'images': get('rti_images', use_rti),
        'korniienko_photo': get('korniienko_photo', use_korniienko),
        'korniienko_drawing': get('korniienko_drawing', use_korniienko),
        'modality_mask': {
            'rti': present('rti_present', use_rti),
            'photo': present('photo_present', use_korniienko),
            'drawing': present('drawing_present', use_korniienko),
        },
    }


def apply_modality_dropout(image_inputs, p=0.0):
    """
    Training-time augmentation: hide each available modality (RTI / photo / drawing) of a
    sample with probability p, but never all of them. Forces the model to read from every
    modality instead of leaning on one.
    """
    mm = image_inputs.get('modality_mask') or {}
    keys = [k for k, v in mm.items() if v is not None]
    if p <= 0 or len(keys) < 2:
        return image_inputs
    present = torch.stack([mm[k].bool() for k in keys], dim=1)            # [B, K]
    keep = present & (torch.rand(present.shape, device=present.device) >= p)
    nothing_left = ~keep.any(dim=1)
    keep[nothing_left] = present[nothing_left]
    new_mask = dict(mm)
    for j, k in enumerate(keys):
        new_mask[k] = keep[:, j]
    return {**image_inputs, 'modality_mask': new_mask}


def language_inputs(batch, device, use_language=False):
    """
    Gold language / writing-system labels as decoder conditioning. Off by default: the
    baselines do not get this information, and for a new inscription it may be unknown.
    """
    if not use_language:
        return {'languages': None, 'writing_systems': None}
    return {'languages': batch['language'].to(device),
            'writing_systems': batch['writing_system'].to(device)}


@torch.no_grad()
def greedy_generate(model, image_inputs, tokenizer, max_length=128,
                    languages=None, writing_systems=None, batch_size=None, device=None):
    """
    Greedy autoregressive decoding. The image encoder runs once; the decoder is
    re-run on the growing prefix. Returns ids [batch, <=max_length] starting with SOS.
    """
    sos_id, eos_id, pad_id = get_special_ids(tokenizer)
    if batch_size is None:
        ref = next(v for k, v in image_inputs.items() if k != 'modality_mask' and v is not None)
        batch_size, device = ref.size(0), ref.device

    memory = model.encode(**image_inputs)

    generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        logits = model.decode(generated, memory,
                              languages=languages, writing_systems=writing_systems)
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == eos_id
        if finished.all():
            break
    return generated


def decode_ids(id_batch, tokenizer):
    """Convert id sequences to strings, stopping at EOS and dropping special tokens."""
    sos_id, eos_id, pad_id = get_special_ids(tokenizer)
    texts = []
    for ids in id_batch:
        ids = [int(i) for i in ids]
        if ids and ids[0] == sos_id:
            ids = ids[1:]  # TrOCR starts from </s>, so drop the start token before looking for EOS
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        if hasattr(tokenizer, 'char_to_idx'):
            unk_id = tokenizer.char_to_idx['<UNK>']
            text = ''.join(tokenizer.idx_to_char[i] for i in ids
                           if i not in (sos_id, pad_id, unk_id) and i in tokenizer.idx_to_char)
        else:
            text = tokenizer.decode(ids, skip_special_tokens=True)
        texts.append(text.strip())
    return texts
