#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import warnings
import os
import html
import unicodedata

warnings.filterwarnings('ignore')

# Import model architectures
from models.models_multichannel import MultiChannelModel
from models.models_enhanced import EnhancedModel
from models.models_transformer import SophiaTransformerModel
from models.models_trocr import SophiaTrOCRModel, TrOCRTokenizerWrapper, DEFAULT_TROCR, DEFAULT_VIEWS
from decoding import (shift_tokens, build_image_inputs, language_inputs, get_special_ids,
                      apply_modality_dropout)

# For tokenization
try:
    from transformers import XLMRobertaTokenizer
except ImportError:
    print(" XLM-RoBERTa tokenizer not available. Using basic character tokenizer.")
    XLMRobertaTokenizer = None

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None


def clean_transcription(text):
    """Decode HTML entities (e.g. '&kappa;', '&middot;', also double-escaped) and NFC-normalise."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ''
    text = str(text)
    for _ in range(3):  # handle '&amp;kappa;' style double escaping
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return unicodedata.normalize('NFC', text).strip()


class CharacterTokenizer:
    """Character-level tokenizer for the multi-script Saint Sophia corpus."""
    
    def __init__(self, vocab_file=None):
        # Create character vocabulary
        self.char_to_idx = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        }
        
        # Original hand-picked characters (kept first so their order is stable)
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        chars += "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψωϫ"
        chars += "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        chars += "ЄІєіѕѣѥѦѧѩѫѯѰѱѲѵѿ҂"
        chars += "ꙀꙁꙂꙃꙄꙅꙆꙇꙈꙉꙊꙋꙌꙍꙎꙏꙐꙑꙒꙓꙔꙕꙖꙗꙘꙙꙚꙛꙜꙝꙞꙟꙠꙡꙢꙣ"
        chars += "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖաբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ"
        chars += "0123456789०१२३४५६७८९୳၊"
        chars += ".,;:!?-–—'\"()[]{}/@#$%&*+=<>|\\~`_"
        chars += "✠ⰰⰱⰲⰳⰴⰵⰶⰷⰸⰹⰺⰻⰼⰽⰾⰿⱀⱁⱂⱃⱄⱅⱆⱇⱈⱉⱊⱋⱌⱍⱎⱏⱐⱑⱒⱓⱔⱕⱖⱗⱘⱙⱚⱛⱜⱝⱞ"
        chars += " \n\t"
        
        # Full Unicode blocks for the scripts in the corpus, so that e.g. lowercase fita,
        # omega, big yus, Polish diacritics and combining titlo are not mapped to <UNK>.
        ranges = [
            (0x00A0, 0x017F),  # Latin-1 Supplement + Latin Extended-A (Polish, German, ...)
            (0x0300, 0x036F),  # Combining diacritical marks
            (0x0370, 0x03FF),  # Greek and Coptic
            (0x0400, 0x052F),  # Cyrillic + Cyrillic Supplement (incl. titlo U+0483)
            (0x0531, 0x058F),  # Armenian
            (0x1C80, 0x1C8F),  # Cyrillic Extended-C
            (0x1F00, 0x1FFF),  # Greek Extended (polytonic)
            (0x2010, 0x205E),  # General punctuation
            (0x2C00, 0x2C5F),  # Glagolitic
            (0x2DE0, 0x2DFF),  # Cyrillic Extended-A (combining superscript letters)
            (0xA640, 0xA69F),  # Cyrillic Extended-B
        ]
        for lo, hi in ranges:
            chars += ''.join(chr(c) for c in range(lo, hi + 1)
                             if unicodedata.category(chr(c)) not in ('Cn', 'Cc', 'Cs', 'Co'))
        # Private Use Area: the transcriptions use PUA code points from the project's own
        # epigraphic font (~10% of lines). Keep them learnable instead of mapping to <UNK>.
        chars += ''.join(chr(c) for c in range(0xE000, 0xE100))
        
        for char in chars:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = len(self.char_to_idx)
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def __call__(self, text, max_length=128, padding='max_length', truncation=True):
        """Tokenize text into character indices."""
        if isinstance(text, list):
            return {'input_ids': [self._encode_single(t, max_length, padding=='max_length', truncation) for t in text],
                    'attention_mask': [self._create_mask(t, max_length) for t in text]}
        else:
            return {'input_ids': self._encode_single(text, max_length, padding=='max_length', truncation),
                    'attention_mask': self._create_mask(text, max_length)}
    
    def _encode_single(self, text, max_length, padding, truncation):
        """Encode single text."""
        indices = [self.char_to_idx['<SOS>']]
        for char in text:
            indices.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
        indices.append(self.char_to_idx['<EOS>'])
        
        if truncation and len(indices) > max_length:
            indices = indices[:max_length-1] + [self.char_to_idx['<EOS>']]
        if padding and len(indices) < max_length:
            indices += [self.char_to_idx['<PAD>']] * (max_length - len(indices))
        return indices
    
    def _create_mask(self, text, max_length):
        """Create attention mask."""
        length = min(len(text) + 2, max_length)  # +2 for SOS and EOS
        return [1] * length + [0] * (max_length - length)
    
    def unknown_chars(self, texts):
        """Characters in `texts` that are not in the vocabulary, with counts."""
        counts = {}
        for t in texts:
            for ch in t:
                if ch not in self.char_to_idx:
                    counts[ch] = counts.get(ch, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    
    def decode(self, indices):
        """Decode indices back to text."""
        chars = []
        for idx in indices:
            if idx in [0, 1, 2]:  # PAD, SOS, EOS
                continue
            chars.append(self.idx_to_char.get(idx, '<UNK>'))
        return ''.join(chars)


class LetterboxResize:
    """Resize keeping the aspect ratio, then pad to (height, width) with neutral grey."""

    def __init__(self, size, fill=(128, 128, 128)):
        self.size, self.fill = size, fill

    def __call__(self, img):
        th, tw = self.size
        w, h = img.size
        scale = min(tw / w, th / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (tw, th), self.fill)
        canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
        return canvas


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_tokenizer(model_type, tokenizer_name='character', pretrained=DEFAULT_TROCR):
    """Tokenizer matching the model: TrOCR's own BPE for 'trocr', else character-level."""
    if model_type == 'trocr':
        return TrOCRTokenizerWrapper(pretrained)
    if tokenizer_name == 'xlm' and XLMRobertaTokenizer is not None:
        return XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    return CharacterTokenizer()


RTI_TYPES = ['original', 'blended', 'normal', 'texture']
# Photometric augmentation only makes sense for photographic layers; normal maps encode
# surface orientation in their colours and must not be colour-jittered.
PHOTOMETRIC_RTI_TYPES = {'original', 'blended'}


class SophiaMultiModalDataset(Dataset):
    """
    Multi-modal dataset for Saint Sophia graffiti recognition.
    
    Returns per sample: 12-channel RTI tensor, Korniienko photo and drawing, tokenized
    transcription, language / writing-system indices, and presence flags telling the model
    which images really exist for this sample (missing ones are masked, not read as black images).
    """
    
    def __init__(self, csv_file, data_dir, tokenizer, max_length=256,
                 use_rti=True, use_korniienko=True, model_type='enhanced',
                 image_size=224, augment=False, split='train', keep_aspect=False):
        """
        Args:
            csv_file: Path to comprehensive dataset CSV
            data_dir: Base directory containing all data
            tokenizer: Text tokenizer
            max_length: Maximum token sequence length (incl. SOS/EOS)
            use_rti: Whether to load RTI images
            use_korniienko: Whether to load Korniienko images
            model_type: 'multichannel', 'enhanced', or 'transformer'
            image_size: int (square) or (height, width)
            augment: Whether to apply data augmentation
            split: Dataset split ('train', 'val', or 'test')
            keep_aspect: letterbox instead of stretching images to image_size
        """
        self.df = pd.read_csv(csv_file)
        self.keep_aspect = keep_aspect
        self.data_dir = Path(data_dir)
        self.rti_dir = self.data_dir
        self.korniienko_dir = self.data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_rti = use_rti
        self.use_korniienko = use_korniienko
        self.model_type = model_type
        self.image_hw = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.augment = augment
        self.split = split
        
        # Clean transcriptions once (HTML entities, Unicode normalisation)
        if 'transcription_clean' in self.df.columns:
            self.df['transcription_clean'] = self.df['transcription_clean'].apply(clean_transcription)
        
        self._create_transforms(augment)
        self._create_mappings()
        self._filter_valid_samples()
        
        print(f"✓ Dataset loaded: {len(self.df)} samples")
        print(f"  RTI: {'✓' if use_rti else '✗'}, Korniienko: {'✓' if use_korniienko else '✗'}")
        self._report_text_coverage()
    
    def _report_text_coverage(self):
        texts = self.df['transcription_clean'].astype(str).tolist() if len(self.df) else []
        if hasattr(self.tokenizer, 'unknown_chars') and texts:
            unk = self.tokenizer.unknown_chars(texts)
            if unk:
                total = sum(len(t) for t in texts)
                shown = ' '.join(f"{c!r}×{n}" for c, n in list(unk.items())[:15])
                print(f"  ⚠ {sum(unk.values())} of {total} characters are not in the vocabulary "
                      f"({len(unk)} distinct): {shown}")
        length = getattr(self.tokenizer, 'token_length', lambda t: len(t) + 2)
        too_long = sum(length(t) > self.max_length for t in texts)
        if too_long:
            print(f"  ⚠ {too_long} transcriptions are longer than max_length={self.max_length} and will be truncated")
        
    def _create_transforms(self, augment):
        """Resize/ToTensor per image; augmentations are applied in __getitem__."""
        resize = LetterboxResize(self.image_hw) if self.keep_aspect else transforms.Resize(self.image_hw)
        self.base_transform = transforms.Compose([resize, transforms.ToTensor()])
        if augment:
            # Photometric: per image (and never on normal maps)
            self.photometric = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
            ])
            # Geometric: one random draw applied to all 12 RTI channels together so the
            # layers stay pixel-aligned. No horizontal flip: mirrored letters are different letters.
            self.geometric = transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5
            )
        else:
            self.photometric = None
            self.geometric = None
    
    def _create_mappings(self):
        """Create language and writing system mappings."""
        languages = self.df['language_name'].fillna('unknown').unique()
        self.language_to_idx = {lang: idx for idx, lang in enumerate(sorted(languages))}
        self.idx_to_language = {idx: lang for lang, idx in self.language_to_idx.items()}
        
        writing_systems = self.df['writing_system_name'].fillna('unknown').unique()
        self.ws_to_idx = {ws: idx for idx, ws in enumerate(sorted(writing_systems))}
        self.idx_to_ws = {idx: ws for ws, idx in self.ws_to_idx.items()}
        
        self.num_languages = len(self.language_to_idx)
        self.num_writing_systems = len(self.ws_to_idx)
    
    def _filter_valid_samples(self):
        """Filter for samples with valid transcriptions and available images."""
        valid_indices = []
        for idx, row in self.df.iterrows():
            if len(str(row.get('transcription_clean', ''))) < 1:
                continue
            if self.use_rti and not self._check_rti_images(row):
                continue
            if self.use_korniienko and not self._check_korniienko_images(row):
                continue
            valid_indices.append(idx)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
    
    def _check_rti_images(self, row):
        """Check if RTI images or IIIF crops exist."""
        for img_type in RTI_TYPES:
            col_name = f"{img_type}_image"
            if col_name in row.index and pd.notna(row[col_name]):
                if (self.data_dir / str(row[col_name])).exists():
                    return True
        
        isialy_id = row.get('isialy_id', row.get('id', ''))
        if not pd.isna(isialy_id):
            rti_dir = self.data_dir / 'cropped_images_hq' / self.split
            for img_type in RTI_TYPES:
                if (rti_dir / img_type / f"{isialy_id}_{img_type}.png").exists():
                    return True
        
        return self._iiif_path(row) is not None
    
    def _check_korniienko_images(self, row):
        """Check if Korniienko images or IIIF crops exist."""
        for col in ('korniienko_photo', 'korniienko_drawing'):
            value = row.get(col, '')
            if not pd.isna(value) and str(value).strip() and (self.data_dir / str(value)).exists():
                return True
        return self._iiif_path(row) is not None
    
    def _iiif_path(self, row):
        value = row.get('iiif_crop', '')
        if not pd.isna(value) and str(value).strip():
            path = os.path.join(self.data_dir, str(value))
            if os.path.exists(path):
                return path
        return None
    
    def _resolve(self, candidates):
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None
    
    def _load_image(self, path, photometric=True):
        """Load -> resize -> [0,1] tensor (+ per-image photometric augmentation)."""
        img = self.base_transform(Image.open(path).convert('RGB'))
        if self.photometric is not None and photometric:
            img = self.photometric(img)
        return img
    
    @staticmethod
    def _normalize(t):
        return (t - 0.5) / 0.5
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isialy_id = row.get('isialy_id', row.get('id', ''))
        h, w = self.image_hw
        
        transcription = str(row.get('transcription_clean', ''))
        encoded = self.tokenizer(
            transcription,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        iiif_crop_path = self._iiif_path(row)
        
        # ---- RTI: 4 layers x 3 RGB = 12 channels ---------------------------------
        layers, layer_present = [], []
        if self.use_rti:
            for rti_type in RTI_TYPES:
                col_name = f"{rti_type}_image"
                path = None
                if col_name in row.index and pd.notna(row[col_name]) and str(row[col_name]).strip():
                    path = self._resolve([os.path.join(self.data_dir, str(row[col_name]))])
                if path is None:
                    path = self._resolve([os.path.join(self.rti_dir, str(isialy_id), f"{rti_type}.jpg")])
                layers.append(self._load_image(path, rti_type in PHOTOMETRIC_RTI_TYPES) if path else None)
                layer_present.append(path is not None)
            # No RTI at all: use the IIIF crop as the 'original' layer only (not copied into
            # all four layers, which would pretend we have blended/normal/texture data).
            if not any(layer_present) and iiif_crop_path:
                layers[0] = self._load_image(iiif_crop_path, True)
                layer_present[0] = True
        else:
            layers, layer_present = [None] * 4, [False] * 4
        
        rti_tensor = torch.cat([l if l is not None else torch.zeros(3, h, w) for l in layers], dim=0)
        if self.geometric is not None and any(layer_present):
            rti_tensor = self.geometric(rti_tensor)  # one draw for all 12 channels
        rti_tensor = self._normalize(rti_tensor)
        for i, present in enumerate(layer_present):
            if not present:
                rti_tensor[3 * i:3 * i + 3] = 0.0
        rti_present = any(layer_present)
        
        # ---- Korniienko photo / drawing ------------------------------------------
        photo_path = drawing_path = None
        if self.use_korniienko:
            for col in ['korniienko_photo', 'korniienko_photo_path', 'korniienko_image']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    value = str(row[col])
                    photo_path = self._resolve([
                        value,
                        os.path.join(self.korniienko_dir, value),
                        os.path.join(self.korniienko_dir, f"{isialy_id}_photo.jpg"),
                        os.path.join(self.korniienko_dir, f"{isialy_id}.jpg"),
                    ])
                    if photo_path:
                        break
            for col in ['korniienko_drawing', 'korniienko_drawing_path']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    value = str(row[col])
                    drawing_path = self._resolve([
                        value,
                        os.path.join(self.korniienko_dir, value),
                        os.path.join(self.korniienko_dir, f"{isialy_id}_drawing.jpg"),
                        os.path.join(self.korniienko_dir, f"{isialy_id}_tracing.jpg"),
                    ])
                    if drawing_path:
                        break
            # IIIF crop as photo fallback only if it is not already used as the RTI layer
            if photo_path is None and iiif_crop_path and not (self.use_rti and rti_present):
                photo_path = iiif_crop_path
        
        def load_aug(path):
            if path is None:
                return torch.zeros(3, h, w)
            t = self._load_image(path, True)
            if self.geometric is not None:
                t = self.geometric(t)
            return self._normalize(t)
        
        return {
            'rti_images': rti_tensor,                       # [12, H, W]
            'korniienko_photo': load_aug(photo_path),       # [3, H, W]
            'korniienko_drawing': load_aug(drawing_path),   # [3, H, W]
            'rti_present': rti_present,
            'photo_present': photo_path is not None,
            'drawing_present': drawing_path is not None,
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'language': self.language_to_idx.get(row.get('language_name', 'unknown'), 0),
            'writing_system': self.ws_to_idx.get(row.get('writing_system_name', 'unknown'), 0),
            'transcription': transcription,
            'isialy_id': isialy_id
        }


def collate_fn(batch):
    """Stack a list of dataset items into a batch."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'language': torch.tensor([item['language'] for item in batch]),
        'writing_system': torch.tensor([item['writing_system'] for item in batch]),
        'transcription': [item['transcription'] for item in batch],
        'isialy_id': [item['isialy_id'] for item in batch],
        'rti_images': torch.stack([item['rti_images'] for item in batch]),
        'korniienko_photo': torch.stack([item['korniienko_photo'] for item in batch]),
        'korniienko_drawing': torch.stack([item['korniienko_drawing'] for item in batch]),
        'rti_present': torch.tensor([item['rti_present'] for item in batch], dtype=torch.bool),
        'photo_present': torch.tensor([item['photo_present'] for item in batch], dtype=torch.bool),
        'drawing_present': torch.tensor([item['drawing_present'] for item in batch], dtype=torch.bool),
    }


def create_model(model_type, vocab_size, num_languages, num_writing_systems, 
                 use_korniienko=True, max_length=256, pretrained=DEFAULT_TROCR,
                 views=DEFAULT_VIEWS, gradient_checkpointing=False):
    """
    Create model instance based on type.
    
    Args:
        model_type: 'multichannel', 'enhanced', or 'transformer'
        vocab_size: Size of vocabulary
        num_languages: Number of language classes
        num_writing_systems: Number of writing system classes
        use_korniienko: Whether to enable Korniienko support
        max_length: Maximum token sequence length
    """
    if model_type == 'multichannel':
        model = MultiChannelModel(
            vocab_size=vocab_size,
            vision_dim=512,
            hidden_dim=512,
            num_layers=8,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            use_korniienko=use_korniienko,
            max_length=max_length
        )
    elif model_type == 'enhanced':
        model = EnhancedModel(
            vocab_size=vocab_size,
            vision_dim=512,
            hidden_dim=512,
            num_layers=8,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            use_korniienko=use_korniienko,
            max_length=max_length
        )
    elif model_type == 'trocr':
        model = SophiaTrOCRModel(pretrained=pretrained, views=views,
                                 gradient_checkpointing=gradient_checkpointing)
        
    elif model_type == 'transformer':
        model = SophiaTransformerModel(
            vocab_size=vocab_size,
            embed_dim=512,
            num_torso_layers=6,
            num_decoder_layers=3,
            num_heads=8,
            num_languages=num_languages,
            num_writing_systems=num_writing_systems,
            max_text_length=max_length
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def load_initial_weights(model, path):
    """Load weights from a checkpoint (e.g. synthetic pretraining) before fine-tuning."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(compatible))
    model.load_state_dict(compatible, strict=False)
    missing = sorted(set(own) - set(compatible))
    print(f"✓ Initialised {len(compatible)}/{len(own)} tensors from {path}")
    if skipped:
        print(f"  skipped {len(skipped)} incompatible tensors, e.g. {skipped[:3]}")
    if missing:
        print(f"  {len(missing)} tensors keep their fresh initialisation, e.g. {missing[:3]}")


def autocast_context(device, enabled):
    """bf16 autocast on CUDA when --amp is given (no-op otherwise)."""
    import contextlib
    if enabled and str(device).startswith('cuda'):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _forward(model, batch, device, use_rti, use_korniienko, decoder_input, decoder_mask,
             use_language=False, modality_dropout=0.0):
    """Single forward call shared by all architectures."""
    image_inputs = build_image_inputs(batch, device, use_rti, use_korniienko)
    if modality_dropout > 0:
        image_inputs = apply_modality_dropout(image_inputs, modality_dropout)
    return model(
        input_ids=decoder_input,
        attention_mask=decoder_mask,
        **language_inputs(batch, device, use_language),
        **image_inputs
    )


def train_epoch(model, dataloader, optimizer, criterion, device, model_type,
                use_rti=True, use_korniienko=True, scheduler=None, use_language=False,
                modality_dropout=0.0, amp=False, grad_accum=1):
    """Train for one epoch with next-token prediction (targets shifted by one)."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input, decoder_mask, target = shift_tokens(input_ids, attention_mask)
        
        with autocast_context(device, amp):
            logits = _forward(model, batch, device, use_rti, use_korniienko,
                              decoder_input, decoder_mask, use_language, modality_dropout)
            loss = criterion(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1))
        
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0 or step + 1 == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()  # per optimizer step
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, model_type,
             use_rti=True, use_korniienko=True, use_language=False, amp=False):
    """Validation loss with next-token prediction (targets shifted by one)."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input, decoder_mask, target = shift_tokens(input_ids, attention_mask)
            
            with autocast_context(device, amp):
                logits = _forward(model, batch, device, use_rti, use_korniienko,
                                  decoder_input, decoder_mask, use_language)
            loss = criterion(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description='Unified Training for Saint Sophia Graffiti Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-Channel model with full multi-modal training
  python train.py --model multichannel --epochs 15 --batch_size 6
  
  # Enhanced model with Korniienko-only (Phase 1)
  python train.py --model enhanced --use_korniienko --no_rti --epochs 10
  
  # Transformer model with all modalities
  python train.py --model transformer --epochs 20 --batch_size 4 --lr 5e-5
        """
    )
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['multichannel', 'enhanced', 'transformer', 'trocr'],
                        help='Model architecture to train (trocr = pretrained TrOCR, fine-tuned)')
    parser.add_argument('--pretrained', type=str, default=DEFAULT_TROCR,
                        help='Hugging Face id or local path of the TrOCR checkpoint (model=trocr)')
    parser.add_argument('--views', type=str, default=','.join(DEFAULT_VIEWS),
                        help='TrOCR input views, comma-separated, from '
                             'original,blended,normal,texture,photo,drawing')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Initialise weights from a checkpoint (e.g. after synthetic pretraining)')
    parser.add_argument('--modality_dropout', type=float, default=0.0,
                        help='Training: hide each available modality with this probability')
    parser.add_argument('--keep_aspect', action='store_true',
                        help='Letterbox images instead of stretching them')
    parser.add_argument('--amp', action='store_true', help='bf16 mixed precision on CUDA')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Trade compute for memory (model=trocr)')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, 
                        default='/home/aram/GRIDH/Saint_Sophia/sophia-epigraphic-ai',
                        help='Base data directory')
    parser.add_argument('--train_csv', type=str, default='train_comprehensive.csv',
                        help='Training CSV file (relative to data_dir/data/)')
    parser.add_argument('--val_csv', type=str, default='val_comprehensive.csv',
                        help='Validation CSV file (relative to data_dir/data/)')
    
    # Modality selection
    parser.add_argument('--use_rti', action='store_true', default=False,
                        help='Use RTI images')
    parser.add_argument('--no_rti', action='store_true', default=False,
                        help='Disable RTI images')
    parser.add_argument('--use_korniienko', action='store_true', default=False,
                        help='Use Korniienko images')
    parser.add_argument('--no_korniienko', action='store_true', default=False,
                        help='Disable Korniienko images')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, nargs='+', default=None,
                        help='Input image size: one value (square) or HEIGHT WIDTH, '
                             'e.g. 128 512 for line-shaped inscriptions')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum token sequence length (incl. SOS/EOS)')
    parser.add_argument('--use_language', action='store_true',
                        help='Condition the decoder on the gold language / writing system '
                             '(off by default: baselines do not get this information)')
    
    # Tokenizer selection
    parser.add_argument('--tokenizer', type=str, default='character',
                        choices=['xlm', 'character'],
                        help='Tokenizer type: xlm (XLM-RoBERTa) or character (character-level)')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Learning rate warmup steps')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (batches)')
    parser.add_argument('--save_interval', type=int, default=0,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=default_device(),
                        help='Device to use')
    
    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = [384] if args.model == 'trocr' else [224]
    args.image_size = args.image_size[0] if len(args.image_size) == 1 else args.image_size[:2]
    if args.lr is None:
        args.lr = 3e-5 if args.model == 'trocr' else 1e-4
    args.views = [v.strip() for v in args.views.split(',') if v.strip()]
    if args.model == 'trocr':
        args.tokenizer = 'trocr'
    
    # Handle modality flags (default to both enabled if neither specified)
    if not args.use_rti and not args.no_rti and not args.use_korniienko and not args.no_korniienko:
        args.use_rti = True
        args.use_korniienko = True
    elif args.use_rti:
        args.use_rti = True
    elif args.no_rti:
        args.use_rti = False
    else:
        args.use_rti = True
    
    if args.use_korniienko:
        args.use_korniienko = True
    elif args.no_korniienko:
        args.use_korniienko = False
    else:
        args.use_korniienko = True
    
    # Validate modality selection
    if not args.use_rti and not args.use_korniienko:
        raise ValueError("At least one modality (RTI or Korniienko) must be enabled!")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("=" * 70)
    print("SAINT SOPHIA GRAFFITI RECOGNITION - UNIFIED TRAINING")
    print("=" * 70)
    print(f"\n Model: {args.model.upper()}")
    print(f"  Modalities:")
    print(f"   RTI Images: {'✓' if args.use_rti else '✗'}")
    print(f"   Korniienko: {'✓' if args.use_korniienko else '✗'}")
    print(f"\n  Hyperparameters:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Image Size: {args.image_size}")
    print(f"   Language conditioning: {'✓' if args.use_language else '✗'}")
    print(f"   Max Text Length: {args.max_length}")
    print(f"\n Data:")
    print(f"   Base Dir: {args.data_dir}")
    print(f"   Train CSV: {args.train_csv}")
    print(f"   Val CSV: {args.val_csv}")
    print(f"\n Device: {args.device}")
    print("=" * 70)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n Checkpoints: {checkpoint_dir}")
    
    # Save configuration
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Config saved: {config_path}")
    
    # Initialize tokenizer
    print("\n Initializing tokenizer...")
    tokenizer = build_tokenizer(args.model, args.tokenizer, args.pretrained)
    vocab_size = tokenizer.vocab_size if not hasattr(tokenizer, '__len__') else len(tokenizer)
    print(f"✓ Tokenizer: {type(tokenizer).__name__} (vocab size: {vocab_size})")
    
    # Create datasets
    print("\n Loading datasets...")
    
    # Handle CSV paths - if they contain 'data/', remove it since data_dir already points to data/
    train_csv_path = args.train_csv.replace('data/', '') if 'data/' in args.train_csv else args.train_csv
    val_csv_path = args.val_csv.replace('data/', '') if 'data/' in args.val_csv else args.val_csv
    
    train_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / train_csv_path,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=True,
        split='train',
        keep_aspect=args.keep_aspect
    )
    
    val_dataset = SophiaMultiModalDataset(
        csv_file=Path(args.data_dir) / val_csv_path,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_rti=args.use_rti,
        use_korniienko=args.use_korniienko,
        model_type=args.model,
        image_size=args.image_size,
        augment=False,
        split='val',
        keep_aspect=args.keep_aspect
    )
    
    # Use the training mappings for validation so language indices agree
    val_dataset.language_to_idx, val_dataset.ws_to_idx = train_dataset.language_to_idx, train_dataset.ws_to_idx
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\n  Creating model...")
    model = create_model(
        model_type=args.model,
        vocab_size=vocab_size,
        num_languages=train_dataset.num_languages,
        num_writing_systems=train_dataset.num_writing_systems,
        use_korniienko=args.use_korniienko,
        max_length=args.max_length,
        pretrained=args.pretrained,
        views=args.views,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    if args.init_from:
        load_initial_weights(model, args.init_from)
    
    model = model.to(args.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created: {args.model}")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    _, _, pad_id = get_special_ids(tokenizer)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore padding token
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, -(-len(train_loader) // args.gradient_accumulation)),
        pct_start=0.1
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✓ Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n Epoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            args.model, args.use_rti, args.use_korniienko, scheduler=scheduler,
            use_language=args.use_language, modality_dropout=args.modality_dropout,
            amp=args.amp, grad_accum=args.gradient_accumulation
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, args.device,
            args.model, args.use_rti, args.use_korniienko,
            use_language=args.use_language, amp=args.amp
        )
        
        # Print metrics
        print(f"\n Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (args.save_interval and (epoch + 1) % args.save_interval == 0) or val_loss < best_val_loss:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            if args.save_interval and (epoch + 1) % args.save_interval == 0:
              torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': vars(args),
                'num_languages': train_dataset.num_languages,
                'num_writing_systems': train_dataset.num_writing_systems,
                'language_to_idx': train_dataset.language_to_idx,
                'idx_to_language': train_dataset.idx_to_language,
                'ws_to_idx': train_dataset.ws_to_idx,
                'idx_to_ws': train_dataset.idx_to_ws,
                'vocab_size': vocab_size
            }, checkpoint_path)
              print(f" Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': vars(args),
                    'num_languages': train_dataset.num_languages,
                    'num_writing_systems': train_dataset.num_writing_systems,
                    'language_to_idx': train_dataset.language_to_idx,
                    'idx_to_language': train_dataset.idx_to_language,
                    'ws_to_idx': train_dataset.ws_to_idx,
                    'idx_to_ws': train_dataset.idx_to_ws,
                    'vocab_size': vocab_size
                }, best_model_path)
                print(f" Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n Best Validation Loss: {best_val_loss:.4f}")
    print(f" Checkpoints saved in: {checkpoint_dir}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'num_languages': train_dataset.num_languages,
        'num_writing_systems': train_dataset.num_writing_systems,
        'language_to_idx': train_dataset.language_to_idx,
        'idx_to_language': train_dataset.idx_to_language,
        'ws_to_idx': train_dataset.ws_to_idx,
        'idx_to_ws': train_dataset.idx_to_ws,
        'vocab_size': vocab_size
    }, final_model_path)
    print(f" Final model saved: {final_model_path}")
    
    print("\n Done!")


if __name__ == '__main__':
    main()
