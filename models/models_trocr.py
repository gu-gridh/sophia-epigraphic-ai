#!/usr/bin/env python3
"""
Pretrained TrOCR for Saint Sophia graffiti (multi-view)
=======================================================

Starts from a pretrained TrOCR (ViT encoder + Transformer decoder). The default is
`cyrillic-trocr/trocr-handwritten-cyrillic` (MIT), fine-tuned on Church Slavonic,
Russian and Ukrainian handwriting.

Multi-view extension: the *shared* pretrained ViT encodes every available view
(RTI original / blended / normal / texture, Korniienko photo / drawing) separately; the
patch tokens of all views are concatenated, each view gets a learned embedding, and the
pretrained decoder cross-attends over all of them. Views missing for a sample are masked.

The tokenizer is TrOCR's byte-level BPE, so every Unicode character (ѣ, ѧ, ꙗ, titlo,
Glagolitic, Armenian ...) is representable without extending the vocabulary.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

DEFAULT_TROCR = 'cyrillic-trocr/trocr-handwritten-cyrillic'
RTI_VIEWS = ('original', 'blended', 'normal', 'texture')
ALL_VIEWS = RTI_VIEWS + ('photo', 'drawing')
DEFAULT_VIEWS = ('original', 'normal', 'photo', 'drawing')


class TrOCRTokenizerWrapper:
    """
    Makes a TrOCR tokenizer fit the dataset / decoding interface.

    Sequences are laid out as TrOCR was fine-tuned:  [decoder_start] <s> text </s> <pad>...
    so after the one-step shift the decoder sees [decoder_start, <s>, ...] and predicts
    [<s>, ..., </s>], exactly like HF `shift_tokens_right` training.
    """

    def __init__(self, name_or_path=DEFAULT_TROCR, decoder_start_token_id=None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.name_or_path = name_or_path
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # generation starts from decoder_start (TrOCR uses </s> = 2)
        self.bos_token_id = (decoder_start_token_id if decoder_start_token_id is not None
                             else self.tokenizer.eos_token_id)
        self.vocab_size = len(self.tokenizer)

    def _encode(self, text, max_length, padding, truncation):
        ids = [self.bos_token_id] + self.tokenizer(text, add_special_tokens=True).input_ids
        if truncation and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_token_id]
        mask = [1] * len(ids)
        if padding and len(ids) < max_length:
            pad = max_length - len(ids)
            ids += [self.pad_token_id] * pad
            mask += [0] * pad
        return ids, mask

    def __call__(self, text, max_length=384, padding='max_length', truncation=True):
        pad = padding == 'max_length'
        if isinstance(text, list):
            enc = [self._encode(t, max_length, pad, truncation) for t in text]
            return {'input_ids': [e[0] for e in enc], 'attention_mask': [e[1] for e in enc]}
        ids, mask = self._encode(text, max_length, pad, truncation)
        return {'input_ids': ids, 'attention_mask': mask}

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_length(self, text):
        return len(self.tokenizer(text, add_special_tokens=True).input_ids) + 1


class SophiaTrOCRModel(nn.Module):
    """Pretrained TrOCR with a shared encoder over several image views."""

    def __init__(self, pretrained=DEFAULT_TROCR, views: Sequence[str] = DEFAULT_VIEWS,
                 view_dropout: float = 0.0, gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import VisionEncoderDecoderModel
        ved = VisionEncoderDecoderModel.from_pretrained(pretrained)
        self.encoder = ved.encoder
        self.decoder = ved.decoder
        enc_dim = ved.encoder.config.hidden_size
        dec_dim = ved.decoder.config.hidden_size
        self.needs_proj = (enc_dim != dec_dim and ved.decoder.config.cross_attention_hidden_size is None)
        self.enc_to_dec_proj = ved.enc_to_dec_proj if self.needs_proj else None
        self.config = ved.config
        self.decoder_start_token_id = ved.config.decoder_start_token_id
        self.image_size = ved.encoder.config.image_size

        unknown = set(views) - set(ALL_VIEWS)
        if unknown:
            raise ValueError(f"Unknown views {unknown}; choose from {ALL_VIEWS}")
        self.views = tuple(views)
        # One learned embedding per view, zero-initialised so a single view behaves exactly
        # like the original pretrained model at the start of fine-tuning.
        self.view_embedding = nn.Parameter(torch.zeros(len(self.views), enc_dim))
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            self.decoder.gradient_checkpointing_enable()
        self.model_type = 'trocr'
        del ved

    # ------------------------------------------------------------------ encoding
    def _view_tensors(self, images, korniienko_photo, korniienko_drawing, modality_mask):
        modality_mask = modality_mask or {}
        out = []
        for i, view in enumerate(self.views):
            if view in RTI_VIEWS:
                if images is None:
                    continue
                k = RTI_VIEWS.index(view)
                x = images[:, 3 * k:3 * k + 3]
                present = modality_mask.get('rti')
                if present is None:
                    present = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
                # a layer that is all zeros after normalisation was missing in the data
                present = present & (x.flatten(1).abs().amax(dim=1) > 0)
            elif view == 'photo':
                if korniienko_photo is None:
                    continue
                x, present = korniienko_photo, modality_mask.get('photo')
            else:
                if korniienko_drawing is None:
                    continue
                x, present = korniienko_drawing, modality_mask.get('drawing')
            if present is None:
                present = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            out.append((i, x, present.to(x.device).bool()))
        return out

    def encode(self, images=None, korniienko_photo=None, korniienko_drawing=None,
               modality_mask=None, **_):
        """Returns (encoder_hidden_states [B, N, D], encoder_attention_mask [B, N], 1 = keep)."""
        views = self._view_tensors(images, korniienko_photo, korniienko_drawing, modality_mask)
        if not views:
            raise ValueError(f"None of the configured views {self.views} were provided")
        states, masks = [], []
        for i, x, present in views:
            interpolate = tuple(x.shape[-2:]) != (self.image_size, self.image_size)
            h = self.encoder(pixel_values=x, interpolate_pos_encoding=interpolate).last_hidden_state
            h = h + self.view_embedding[i]
            states.append(h)
            masks.append(present[:, None].expand(-1, h.size(1)))
        hidden = torch.cat(states, dim=1)
        mask = torch.cat(masks, dim=1)
        # never leave a sample with nothing to attend to
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty] = True
        if self.needs_proj:
            hidden = self.enc_to_dec_proj(hidden)
        return hidden, mask.long()

    # ------------------------------------------------------------------ decoding
    def decode(self, input_ids, memory, attention_mask=None, languages=None, writing_systems=None):
        hidden, mask = memory
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden,
            encoder_attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    def forward(self, input_ids, attention_mask=None, images=None, languages=None,
                writing_systems=None, korniienko_photo=None, korniienko_drawing=None,
                modality_mask=None):
        """Teacher-forced forward pass: input_ids is the (shifted) decoder input."""
        memory = self.encode(images, korniienko_photo, korniienko_drawing, modality_mask)
        return self.decode(input_ids, memory, attention_mask)
