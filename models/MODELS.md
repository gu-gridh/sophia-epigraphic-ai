# SOPHIA Epigraphic AI — Model Architectures

This document provides a comprehensive technical analysis of the three neural network architectures developed for automated transcription of historical graffiti from Saint Sophia Cathedral in Kyiv. These models address the unique challenges of Byzantine epigraphy through specialized multi-modal architectures that integrate visual, textual, and spatial information.

## Research Context and Challenges

The automated transcription of medieval graffiti presents unprecedented challenges in computer vision and natural language processing:

1. **Severely degraded surfaces**: Centuries of weathering, human damage, and environmental factors have rendered many inscriptions barely visible
2. **Complex multilingual environment**: Graffiti spans 10+ languages (Church Slavonic, Ukrainian, Polish, Ancient Greek, Armenian, Latin, etc.) across 7 writing systems
3. **Extremely limited supervision**: Unlike modern OCR with abundant training data, historical epigraphy suffers from scarce, expert-annotated datasets
4. **Multi-modal evidence**: Historians traditionally combine multiple imaging techniques (photography, hand-traced drawings) to decipher inscriptions
5. **Spatial context matters**: Location within the cathedral (wall, elevation, panel) provides crucial dating and attribution clues

Our approach combines state-of-the-art transformer architectures with domain-specific innovations to address these challenges through three complementary model families.

## Design Philosophy and Main Objectives

### Core Design Principles

1. **Multi-modal Integration**: Mirrors human expert methodology by combining multiple evidence sources
2. **Robustness to Missing Data**: Real-world deployment often lacks complete multi-modal data
3. **Language-Aware Processing**: Explicit modeling of multilingual and multi-script challenges
4. **Attention-Based Focus**: Learned attention mechanisms to focus on degraded inscription regions
5. **Historical Context**: Incorporation of spatial and temporal information for attribution

### Primary Research Goals

- **9.65% Character Error Rate**: Establish state-of-the-art performance on low-resource historical text
- **86.6% Relative Improvement**: Outperform transformer baselines through multi-modal fusion
- **Cross-Modal Robustness**: Maintain performance when imaging modalities are unavailable
- **Scalable Architecture**: Support future expansion to other historical text domains

## Files and Implementation

- **Transformer**: `models/models_transformer.py` (class `SophiaTransformerModel`) — Multi-modal fusion transformer
- **Enhanced**: `models/models_enhanced.py` (class `EnhancedModel`) — Deep vision + language conditioning
- **Multichannel**: `models/models_multichannel.py` (class `MultiChannelModel`) — Channel-specific processing
- **Legacy baseline**: `models/models.py` (class `SophiaModel`) — Original multimodal architecture

## At-a-glance comparison

| Aspect | Transformer | Enhanced | Multichannel |
|---|---|---|---|
| Vision input | Standard RGB images + optional Korniienko photo/drawing | Standard RGB images, optional Korniienko fusion | Standard RGB images, optional Korniienko fusion |
| Vision encoder | ResNet-style per-modality encoders + modality attention | Deep CNN with spatial attention, multiscale fusion | CNN with multi-head spatial attention + spatial self-attention |
| Language conditioning | Optional via TextEncoder (character-level) | Yes (language + writing system embeddings fused into decoder) | Yes (language + writing system embeddings fused into decoder) |
| Spatial info | Explicit SpatialEncoder (bbox + location) | N/A (vision-only + optional language conditioning) | N/A (vision-only + optional language conditioning) |
| Fusion | Transformer torso over modalities | Cross-modal attention inside decoder | Cross-modal attention inside decoder |
| Heads | Transcription head (seq2seq), Attribution (language, writing system, date) | Decoder-only transcription | Decoder-only transcription + generation helper |
| Output API | Dict with logits and attribution | Logits tensor | Logits tensor, plus `generate(...)` |

## Technical Design Rationale

### Why Three Different Architectures?

Each model addresses different aspects of the historical graffiti transcription challenge:

1. **Transformer Model**: **Comprehensive Multi-Modal Fusion**
   - **Motivation**: Mirrors human expert methodology that combines visual evidence, existing transcriptions, and spatial context
   - **Innovation**: Cross-modal attention mechanism allows each modality to attend to others, creating historically-enriched representations
   - **Use case**: When complete multi-modal data is available and attribution metadata is needed

2. **Enhanced Model**: **Deep Visual Processing + Language Conditioning**
   - **Motivation**: Many inscriptions are primarily visual challenges requiring sophisticated image analysis
   - **Innovation**: Channel attention over RTI types + spatial attention + language-conditioned decoding
   - **Use case**: When visual quality is the primary bottleneck and language context is known

3. **Multichannel Model**: **Specialized Image Processing**
   - **Motivation**: Multiple imaging sources (photography, drawings, lighting conditions) contain unique information about inscription details
   - **Innovation**: Dedicated processors per image source with learned weighting and multi-head spatial attention
   - **Use case**: When multiple imaging sources are available and maximum visual information extraction is needed

### Core Architectural Innovations

#### Multi-Modal Attention Mechanisms

All models implement specialized attention patterns designed for historical epigraphy:

```
Visual Attention: Focus on degraded inscription regions
Cross-Modal Attention: Align visual features with linguistic priors  
Spatial Attention: Weight features by cathedral location context
Channel Attention: Emphasize informative RTI imaging types
```

#### Character-Level Tokenization Strategy

**Problem**: Subword tokenizers (BPE, WordPiece) fragment rare historical characters, losing critical paleographic information.

**Solution**: Character-level vocabulary (407 symbols) preserving:
- Complete Cyrillic alphabet (33 characters)
- Ancient Greek alphabet (24 characters) 
- Latin alphabet variants
- Historical ligatures and abbreviation marks
- Byzantine numerical symbols
- Punctuation and special characters

#### Language Conditioning Architecture

**Problem**: Medieval graffiti exhibits complex code-switching and multilingual interference.

**Solution**: Explicit language and writing system embeddings that condition the decoder:

```python
# Simplified conditioning mechanism
lang_emb = self.language_embedding(language_ids)      # [B, D/2]
ws_emb = self.writing_system_embedding(ws_ids)        # [B, D/2] 
conditioning = self.fusion(concat([lang_emb, ws_emb])) # [B, D]

# Broadcast conditioning to all decoder positions
decoder_input = token_emb + pos_emb + conditioning.unsqueeze(1)
```

### Handling Missing Modalities

Real-world deployment often lacks complete imaging data. Our models implement graceful degradation:

```python
# Modality dropout during training
if self.training and random.random() < modality_dropout_prob:
    korniienko_photo = None  # Simulate missing reference image

# Zero-padding for missing modalities
if korniienko_photo is None:
    photo_features = torch.zeros_like(rti_features)
else:
    photo_features = self.photo_encoder(korniienko_photo)
```

This training strategy ensures robust performance when deployed with incomplete imaging pipelines.

## Common inputs and data conventions

- Image inputs: Standard RGB images → shape `[B, 3, H, W]` (can be extended to multiple images as `[B, 3*N, H, W]` for N different image sources).
- Korniienko references: photo `[B, 3, H, W]`, drawing `[B, 3, H, W]` (both optional; models gracefully handle None and use zero features when absent).
- Language/writing system: either integer IDs or string names depending on the model. See model sections below for details.
- Character vocabulary: Use a character-level tokenizer that preserves rare historical characters (e.g., 407-symbol vocabulary). The models treat token id 0 as padding in transformer encoders/decoders where specified.

## 1) Transformer model (`SophiaTransformerModel`)

Source: `models/models_transformer.py`

### Inputs

- `rti_images`: `[B, 12, H, W]` — RTI stacked channels
- `korniienko_photo`: `[B, 3, H, W]` — optional
- `korniienko_drawing`: `[B, 3, H, W]` — optional
- `spatial_info`: dict with
  - `bbox`: `[B, 4]` (x, y, w, h) in normalized 0–1 coordinates
  - `location`: `[B, 3]` (panel_id, room, elevation)
- `text_indices`: `[B, T]` — tokenized transcription (optional; used both for text-guided encoding and as teacher forcing target)
- `text_mask`: `[B, T]` — 1 for valid tokens, 0 for padding (optional)
- `return_embeddings`: bool — if true, returns pooled embeddings

### Design Philosophy: Multi-Modal Evidence Fusion

The Transformer model embodies the principle that **historical inscription analysis requires multiple, complementary evidence sources**. This architecture directly mirrors the methodology used by expert epigraphers who combine:

1. **Visual analysis** from multiple imaging techniques
2. **Textual context** from partial transcriptions or related inscriptions  
3. **Spatial context** from location within the architectural space
4. **Historical context** for dating and attribution

#### Why Cross-Modal Attention?

Traditional approaches process modalities independently and concatenate features. Our cross-modal attention allows each modality to **actively query and attend to relevant information** in other modalities:

```text
Visual features attend to text: "What linguistic patterns should I expect?"
Text features attend to vision: "Which visual regions support this transcription?"
Spatial features attend to both: "What languages/scripts are typical for this cathedral location?"
```

This creates **historically-enriched representations** where visual features are enhanced with linguistic priors, and textual features are grounded in visual evidence.

### Core Components and Rationale

#### MultiModalVisionEncoder: Dedicated Processing per Modality

**Problem**: Different imaging modalities capture complementary information:
- Standard photography reveals color and overall context
- Korniienko photos provide expert documentation
- Korniienko drawings highlight expert-interpreted character shapes
- Different lighting conditions reveal varying surface details

**Solution**: Separate encoders with learned modality tokens:

```text
Image Processing: ResNet-style → Surface texture and visual features
Photo Processing: Lightweight CNN → Color and context features  
Drawing Processing: Lightweight CNN → Character shape features
Modality Fusion: Multi-head attention with learned query tokens
```

#### SpatialEncoder: Architectural Context Integration

**Insight**: Inscription location provides crucial historical evidence:
- **Elevation**: Higher inscriptions often older (less accessible for later additions)
- **Panel type**: Different architectural elements associated with different periods
- **Room**: Liturgical vs. auxiliary spaces have different inscription patterns

**Implementation**: 
- Bounding box MLP: Encodes geometric relationships
- Location MLP: Encodes categorical spatial information  
- Combined projection: Creates unified spatial representation

#### TransformerTorso: Cross-Modal Reasoning

**Design**: Standard Transformer encoder over concatenated modality tokens enables:
- **Cross-modal attention**: Each modality can attend to others
- **Hierarchical reasoning**: Deep layers build complex multi-modal concepts
- **Missing modality robustness**: Attention masks handle absent data

### TranscriptionHead: Sequence-to-Sequence Generation

**Architecture**: Transformer decoder with memory from the torso
- **Memory**: Multi-modal enriched features serve as context
- **Target**: Character sequence with teacher forcing during training
- **Attention**: Decoder can selectively attend to different modality combinations

### Attribution Heads: Historical Context Modeling

Inspired by Assael et al.'s ITHACA model, we predict historical metadata:

- **Language classification**: 10 medieval languages (Church Slavonic, Greek, etc.)
- **Writing system classification**: 7 scripts (Cyrillic, Latin, Greek, etc.)  
- **Dating**: 160 time bins covering 1010-1715 CE (similar to ITHACA)

These auxiliary tasks provide **multi-task learning** benefits and enable **historical analysis** beyond transcription.

### Transformer Model Inputs

- `images`: `[B, 3, H, W]` — Standard RGB images
- `korniienko_photo`: `[B, 3, H, W]` — optional
- `korniienko_drawing`: `[B, 3, H, W]` — optional
- `spatial_info`: dict with
  - `bbox`: `[B, 4]` (x, y, w, h) in normalized 0–1 coordinates
  - `location`: `[B, 3]` (panel_id, room, elevation)
- `text_indices`: `[B, T]` — tokenized transcription (optional; used both for text-guided encoding and as teacher forcing target)
- `text_mask`: `[B, T]` — 1 for valid tokens, 0 for padding (optional)
- `return_embeddings`: bool — if true, returns pooled embeddings

### Transformer Model Architecture

- MultiModalVisionEncoder
  - Image: Standard RGB CNN → ResNet-style blocks → global average pooling → dense projection to `embed_dim`
  - Korniienko: separate small CNN encoders for photo and drawing
  - Learned modality tokens + multi-head attention for cross-modality fusion
- TextEncoder
  - Character embeddings + sinusoidal positional encodings
  - TransformerEncoder (batch_first) → sequence features and mean-pooled features
- SpatialEncoder
  - MLPs for bbox and location, concatenated and projected to `embed_dim`
- TransformerTorso
  - TransformerEncoder across concatenated modalities (vision, optional text-pooled, optional spatial)
- Decoder heads
  - TranscriptionHead: TransformerDecoder over memory (enriched multi-modal sequence) with output projection to vocabulary
  - AttributionHead: MLPs for language, writing system, and date-bin classification

### Outputs

- `outputs['transcription_logits']`: `[B, T, V]`
- `outputs['attribution']`: dict with keys `language`, `writing_system`, `date` (logits)
- `outputs['embeddings']`: `[B, embed_dim]` if `return_embeddings=True`

### Objective sketch

- Character-level cross-entropy for transcription:
  $$\mathcal{L}_{\text{CE}} = -\frac{1}{B}\sum_{b=1}^B \sum_{t=1}^T w_t\,\log p(y_{b,t}\mid x_b)$$
  with optional label smoothing and padding mask $w_t$.
- Auxiliary multi-task losses for language/writing system/date attribution (softmax CE), optionally weighted.

### Typical instantiation

```python
from models.models_transformer import create_sophia_transformer
model = create_sophia_transformer(
    vocab_size=407,
    num_languages=10,
    num_writing_systems=7,
    embed_dim=512,
    num_torso_layers=6,
    num_decoder_layers=3,
    num_heads=8,
    dropout=0.1,
)
```

## 2) Enhanced model (`EnhancedModel`)

Source: `models/models_enhanced.py`

### Design Philosophy: Deep Visual Processing + Language Conditioning

The Enhanced model addresses a key insight: **many historical graffiti transcription failures stem from inadequate visual feature extraction** rather than insufficient multi-modal fusion. This architecture prioritizes:

1. **Deep visual understanding** through sophisticated attention mechanisms
2. **Language-conditioned decoding** that adapts to script and linguistic context
3. **Computational efficiency** by focusing resources on the most critical components

#### Why Channel Attention for Multiple Images?

Instead of RTI-specific processing, this model can work with multiple image sources:

- **Base photography**: Standard inscription photography
- **Enhanced lighting**: Images taken under different lighting conditions  
- **Detailed shots**: Close-up photography of specific character regions
- **Reference materials**: Historical documentation or tracings

**Problem**: Simply concatenating multiple images treats all information equally, but different inscription conditions require emphasizing different imaging sources.

**Solution**: Learned channel attention weights that adapt to inscription characteristics:

```python
# Channel attention produces weights for each image source
channel_weights = self.channel_attention(multi_channel_input)  # [B, N, 1, 1]

# Apply learned weights to emphasize informative modalities
weighted_features = []
for i in range(num_image_sources):
    image_features = input_images[:, i*3:(i+1)*3, :, :]
    weight = channel_weights[:, i:i+1, :, :]
    weighted_features.append(image_features * weight)
```

#### Multi-Scale Spatial Attention

**Insight**: Historical inscriptions contain details at multiple scales:
- **Character level**: Individual letter shapes and strokes
- **Word level**: Spacing and layout patterns
- **Inscription level**: Overall structure and organization

**Implementation**: Multiple attention heads operating at different receptive fields:

```python
# Multiple spatial attention heads with different kernel sizes
spatial_attention_heads = [
    attention_head_fine,     # 3x3 kernels for character details
    attention_head_medium,   # 5x5 kernels for word structure  
    attention_head_coarse,   # 7x7 kernels for layout patterns
]

# Combine attention outputs
multi_scale_attention = torch.stack([
    head(features) for head in spatial_attention_heads
], dim=1).mean(dim=1)
```

#### Language-Conditioned Decoding Strategy

**Problem**: Medieval multilingual environment requires script-aware processing. The same visual features need different interpretations depending on the linguistic context.

**Solution**: Language and writing system embeddings condition every step of the decoding process:

```python
# Create rich linguistic context
lang_embedding = self.language_embedding(language_ids)     # Church Slavonic vs Greek
script_embedding = self.writing_system_embedding(ws_ids)   # Cyrillic vs Greek script
linguistic_context = self.fusion([lang_embedding, script_embedding])

# Broadcast to all sequence positions  
decoder_input = token_embeddings + position_embeddings + linguistic_context
```

This conditioning enables the model to adapt character recognition patterns based on expected linguistic and paleographic context.

### Enhanced Model Inputs

- `input_ids`: `[B, T]` — decoding token IDs
- `images`: `[B, 3*N, H, W]` — Multiple images input (optional; zeros used if None, where N is number of image sources)
- `languages`: either tensor of IDs `[B]` or list of string labels
- `writing_systems`: either tensor of IDs `[B]` or list of string labels
- `korniienko_photo`, `korniienko_drawing`: optional `[B, 3, H, W]`

### Enhanced Model Architecture

- EnhancedVisionEncoder
  - Channel attention over multiple image sources (grouped as N image sources × 3 channels)
  - Deep residual CNN stack (four stages)
  - Multiscale fusion with upsampling and a spatial attention head
  - Optional Korniienko photo/drawing encoders + fusion MLP
- LanguageConditionedDecoder
  - Token + position embeddings
  - Language and writing system embeddings, fused and projected into the decoder
  - Cross-modal attention to image features (as memory)
  - 8-layer TransformerDecoder with pre-norm, GELU feed-forward

### Outputs

- Logits `[B, T, V]` (character vocabulary size `V`)

### Notes

- Provides a compact decoder-only training flow: pass ground truth `input_ids` shifted for teacher forcing.
- If `languages`/`writing_systems` are strings, the decoder maps them through built-in dictionaries. Integer ID tensors are also supported.

## 3) Multichannel model (`MultiChannelModel`)

Source: `models/models_multichannel.py`

### Design Philosophy: Specialized Image Processing + Spatial Self-Attention

The Multichannel model embodies the principle that **each image source should be processed by specialized encoders** before fusion. This approach maximizes information extraction from multiple image inputs while maintaining computational efficiency.

#### Key Innovation: Dedicated Image-Source Processors

**Problem**: Multiple imaging sources produce fundamentally different image types, each with distinct statistical properties and information content. A single CNN encoder cannot optimally process all types.

**Solution**: Multiple specialized sub-encoders, each designed for specific image characteristics:

```python
# Dedicated processors for each image source
channel_processors = nn.ModuleList([
    BasePhotographyProcessor(),    # Handles standard photography
    EnhancedLightingProcessor(),   # Enhanced contrast, different lighting
    DetailedShotProcessor(),       # Close-up detail analysis  
    ReferenceProcessor()           # Historical reference materials
])

# Process each image source separately, then fuse
processed_channels = []
for i, processor in enumerate(channel_processors):
    image_source = images[:, i*3:(i+1)*3, :, :]  # Extract 3 RGB channels
    processed = processor(image_source)
    processed_channels.append(processed)
```

#### Multi-Head Spatial Attention with Self-Attention

**Insight**: Different spatial regions of an inscription require different attention patterns. Some areas may need fine-grained character analysis, while others need broader contextual understanding.

**Architecture**: Multiple attention heads followed by spatial self-attention:

```python
# Multiple attention heads for different spatial patterns
spatial_attentions = []
for attention_head in self.spatial_attention_heads:
    attention_weights = attention_head(features)
    attended = features * attention_weights
    spatial_attentions.append(attended)

# Combine multi-head attention
multi_attended = torch.stack(spatial_attentions, dim=1).mean(dim=1)

# Spatial self-attention: each location attends to all others
spatial_tokens = features.view(B, C, H*W).permute(0, 2, 1)  # [B, HW, C]
self_attended, _ = self.self_attention(spatial_tokens, spatial_tokens, spatial_tokens)
```

This allows the model to:
1. **Focus on degraded regions** that need enhanced processing
2. **Integrate information** across spatially separated character fragments  
3. **Model long-range dependencies** in inscription layout and structure

#### Spatial Grid Pooling Strategy

**Problem**: Global average pooling loses important spatial information needed for character recognition.

**Solution**: Adaptive pooling to a small spatial grid (4×4) that preserves local structure:

```python
# Preserve spatial structure with grid pooling
self.feature_projection = nn.Sequential(
    nn.AdaptiveAvgPool2d((4, 4)),  # Keep 16 spatial locations
    nn.Flatten(),                   # [B, hidden_dim * 16]
    nn.Linear(hidden_dim * 16, hidden_dim * 4),  # Gradual reduction
    nn.Linear(hidden_dim * 4, hidden_dim)       # Final projection
)
```

This retains sufficient spatial information for character-level analysis while maintaining computational efficiency.

### Enhanced Cross-Modal Language Conditioning

The Multichannel decoder extends the Enhanced model's language conditioning with additional cross-attention layers:

```python
# Multi-level cross-modal attention
vision_text_attention = self.vision_text_attention(embeddings, vision_memory)
cross_modal_attention = self.cross_modal_attention(embeddings, vision_memory)

# Combine multiple attention mechanisms
enhanced_embeddings = embeddings + vision_text_attention + cross_modal_attention
```

This creates richer vision-language interactions, allowing the model to better align visual features with linguistic expectations.

### Multichannel Model Inputs

- `input_ids`: `[B, T]` — decoding token IDs
- `images`: `[B, 3*N, H, W]` — Multiple image sources (optional; zeros used if None, where N is number of sources)
- `languages`: list of strings (e.g., 'Church Slavonic', 'Ukrainian', ...)
- `writing_systems`: list of strings (e.g., 'Cyrillic', 'Greek', ...)
- `korniienko_photo`, `korniienko_drawing`: optional `[B, 3, H, W]`

### Multichannel Model Architecture

- MultiChannelVisionEncoder
  - Multiple dedicated sub-encoders (one per image source) that process 3 channels each
  - Channel attention to reweight the different image sources
  - Deep CNN stack to 512/hidden_dim channels
  - Multiple spatial attention heads averaged, plus spatial self-attention over tokens
  - Projection with small spatial grid pooling to retain structure
  - Optional Korniienko photo/drawing encoders + fusion
- LanguageConditionedDecoder
  - Similar to Enhanced, with richer language fusion and additional cross-attention

### Multichannel Model Outputs

- Logits `[B, T, V]`
- Convenience `generate(...)` method for autoregressive inference with sampling temperature and BOS/EOS control.

### Multichannel Implementation Notes

- Accepts language and writing system as human-readable strings; internally maps to IDs via `encode_language_info`.
- The `generate` method performs simple multinomial sampling; swap with greedy or beam search as needed.

## Input/Output shapes summary

- Images: `[B, 12, H, W]` (RTI). Korniienko: `[B, 3, H, W]` each.
- Tokens: `[B, T]` with padding index 0 if applicable.
- Transformer model returns a dict; Enhanced/Multichannel return logits directly.

## Training recipes

- Tokenization: character-level tokenizer preserving rare historical characters (e.g., 407-symbol vocab); reserve 0 for padding, 1 for BOS, 2 for EOS.
- Losses
  - Cross-entropy over vocabulary (mask padding positions)
  - Optional label smoothing (e.g., 0.1)
  - For Transformer model, add auxiliary CE losses for attribution heads
- Optimizer/schedule
  - AdamW with weight decay (1e-4 to 1e-2)
  - Warmup + cosine/linear decay
- Regularization
  - Dropout (0.1–0.3 across encoders/decoders)
  - Modality dropout (Transformer: randomly drop vision/text/spatial channels during training)
- Augmentation (vision)
  - Color jitter, random crop/resize, Gaussian noise
  - RTI-aware perturbations per channel type

## Inference examples

### Transformer
```python
import torch
from models.models_transformer import create_sophia_transformer

model = create_sophia_transformer(vocab_size=407, embed_dim=512, num_torso_layers=6, num_decoder_layers=3, num_heads=8)
model.eval()

B, H, W, T = 2, 224, 224, 64
images = torch.randn(B, 3, H, W)  # Standard RGB images
spatial_info = {"bbox": torch.rand(B, 4), "location": torch.randn(B, 3)}
text_indices = torch.full((B, T), 1, dtype=torch.long)  # dummy BOS-filled sequence

with torch.no_grad():
    out = model(images=images, spatial_info=spatial_info, text_indices=text_indices, return_embeddings=True)
    logits = out["transcription_logits"]  # [B, T, V]
    emb = out["embeddings"]                # [B, D]
```

### Enhanced
```python
import torch
from models.models_enhanced import EnhancedModel

model = EnhancedModel(vocab_size=407, vision_dim=512, hidden_dim=512, num_layers=8)
model.eval()

B, H, W, T = 2, 224, 224, 64
images = torch.randn(B, 3, H, W)  # Standard RGB images
input_ids = torch.full((B, T), 1, dtype=torch.long)
langs = torch.tensor([0, 1])            # or ["church_slavonic", "ukrainian"]
writings = torch.tensor([0, 1])         # or ["cyrillic", "latin"]

with torch.no_grad():
    logits = model(input_ids=input_ids, images=images, languages=langs, writing_systems=writings)
```

### Multichannel (with generation)
```python
import torch
from models.models_multichannel import MultiChannelModel

model = MultiChannelModel(vocab_size=407, vision_dim=512, hidden_dim=512, num_layers=8)
model.eval()

B, H, W, T = 2, 224, 224, 64
images = torch.randn(B, 3, H, W)  # Standard RGB images

with torch.no_grad():
    generated = model.generate(images, languages=["Church Slavonic"]*B, writing_systems=["Cyrillic"]*B, max_length=64)
```

## Parameter counts

Use the snippet below to report parameter counts for your current configurations.

```python
import torch
from models.models_transformer import create_sophia_transformer
from models.models_enhanced import EnhancedModel
from models.models_multichannel import MultiChannelModel

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

# Adjust hyperparameters to your experiment settings
tr = create_sophia_transformer(vocab_size=407, embed_dim=512, num_torso_layers=6, num_decoder_layers=3, num_heads=8)
en = EnhancedModel(vocab_size=407, vision_dim=512, hidden_dim=512, num_layers=8)
mc = MultiChannelModel(vocab_size=407, vision_dim=512, hidden_dim=512, num_layers=8)

for name, m in [("Transformer", tr), ("Enhanced", en), ("Multichannel", mc)]:
    t, trn = count_params(m)
    print(f"{name}: total={t:,} trainable={trn:,}")
```

## Training Methodology and Objectives

### Multi-Task Learning Framework

Our models are trained with multiple complementary objectives that reflect the diverse requirements of historical epigraphy:

#### 1. Primary Transcription Objective

**Character-level Cross-Entropy Loss** with label smoothing to handle annotation uncertainty:

```text
L_transcription = -∑∑ (1-ε)y_ij log(p_ij) + (ε/|V|) log(p_ij)
                   i j

where:
- y_ij: true character label (one-hot)
- p_ij: predicted probability for character j at position i
- ε: label smoothing factor (typically 0.1)
- |V|: vocabulary size (407 characters)
```

**Rationale**: Label smoothing accounts for transcription ambiguities inherent in degraded historical texts, where expert annotators may disagree on character identity.

#### 2. Attribution Objectives (Transformer Model Only)

**Language Classification**: Cross-entropy over 10 medieval languages
```text
L_language = -∑ y_lang log(p_lang)
```

**Writing System Classification**: Cross-entropy over 7 scripts  
```text
L_script = -∑ y_script log(p_script)
```

**Dating Classification**: Cross-entropy over 160 temporal bins (1010-1715 CE)
```text
L_date = -∑ y_date log(p_date)  
```

**Combined Multi-Task Loss**:
```text
L_total = λ_trans × L_transcription + λ_lang × L_language + λ_script × L_script + λ_date × L_date

where λ_trans = 1.0, λ_lang = 0.3, λ_script = 0.3, λ_date = 0.2
```

### Training Strategies for Robustness

#### Modality Dropout for Missing Data Robustness

During training, we randomly drop modalities to simulate real-world deployment conditions:

```python
# Modality dropout probabilities
if self.training:
    if random.random() < 0.3:  # 30% chance
        korniienko_photo = None
    if random.random() < 0.3:  # 30% chance  
        korniienko_drawing = None
    if random.random() < 0.1:  # 10% chance (RTI is most critical)
        rti_images = None
```

This ensures models maintain performance when deployed with incomplete imaging pipelines.

#### Data Augmentation for Visual Robustness

Historical graffiti exhibits extreme variation in preservation quality. We apply targeted augmentations:

**Image-Specific Augmentations**:
- **Per-source noise**: Different noise patterns for each image source
- **Source dropout**: Randomly zero out specific image sources
- **Intensity scaling**: Simulate lighting variation in image capture

**General Visual Augmentations**:
- **Color jitter**: ±20% saturation/brightness to simulate aging
- **Gaussian noise**: σ=0.05 to simulate surface degradation  
- **Random crop/resize**: 224×224 → 256×256 → crop to 224×224

#### Curriculum Learning for Complex Scripts

We implement a three-stage curriculum:

1. **Stage 1 (Epochs 1-10)**: Train on high-quality, clear inscriptions only
2. **Stage 2 (Epochs 11-25)**: Introduce moderately degraded samples  
3. **Stage 3 (Epochs 26+)**: Full dataset including severely damaged inscriptions

This progressive difficulty schedule improves convergence and final performance.

### Optimization and Regularization

#### AdamW with Warmup and Decay

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    eps=1e-8
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=50,
    steps_per_epoch=len(dataloader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

#### Gradient Clipping for Stability

```python
# Clip gradients to prevent exploding gradients in attention mechanisms
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Dropout and Layer Normalization

- **Encoder dropout**: 0.1-0.2 in CNN layers, 0.1-0.3 in transformers
- **Decoder dropout**: 0.2-0.3 to prevent overfitting on limited character sequences
- **Attention dropout**: 0.1 in multi-head attention to maintain attention diversity

### Performance Metrics and Evaluation

#### Character Error Rate (CER)

Primary metric following historical OCR conventions:

```text
CER = (Substitutions + Insertions + Deletions) / Total_Characters

Target: <10% CER (state-of-the-art for historical texts)
```

#### Character Accuracy

```text
Accuracy = Correct_Characters / Total_Characters

Target: >90% accuracy
```

#### BLEU Score (Sequence-Level)

For evaluating complete inscription transcriptions:

```text
BLEU-4 = BP × exp(∑(1/4) × log(precision_n))
```

## Parameter Counts and Computational Requirements

Use the snippet below to report parameter counts for your current configurations.

```python
import torch
from models.models_transformer import create_sophia_transformer
from models.models_enhanced import EnhancedModel
from models.models_multichannel import MultiChannelModel

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

# Adjust hyperparameters to your experiment settings
tr = create_sophia_transformer(vocab_size=407, embed_dim=512, num_torso_layers=6, num_decoder_layers=3, num_heads=8)
en = EnhancedModel(vocab_size=407, vision_dim=512, hidden_dim=512, num_layers=8)
mc = MultiChannelModel(vocab_size=407, vision_dim=512, hidden_dim=512, num_layers=8)

for name, m in [("Transformer", tr), ("Enhanced", en), ("Multichannel", mc)]:
    t, trn = count_params(m)
    print(f"{name}: total={t:,} trainable={trn:,}")
```

**Typical Model Sizes**:
- Transformer: ~58.3M parameters
- Enhanced: ~45-55M parameters  
- Multichannel: ~50-60M parameters

## Practical tips and edge cases

- Missing modalities
  - All models tolerate missing Korniienko inputs; Enhanced/Multichannel also tolerate missing RTI by substituting zeros.
- Sequence handling
  - Ensure padding masks are correct when training with packed batches; the transformer model converts mask semantics internally.
- Vocab and special tokens
  - Reserve indices for PAD/BOS/EOS consistently across models and data pipeline.
- Generation
  - `MultiChannelModel.generate` uses multinomial sampling; for reproducible or more accurate decoding, implement greedy/beam search.

## Model Selection Guidelines

### When to Use Each Architecture

**Transformer Model (`SophiaTransformerModel`)**:
- ✅ **Complete multi-modal data** (RTI + Korniienko + spatial metadata + partial transcriptions)
- ✅ **Attribution tasks** required (dating, language identification, writing system classification)
- ✅ **Research scenarios** with full experimental pipeline
- ✅ **Cross-modal analysis** needed (understanding vision-text-spatial interactions)
- ❌ Computational constraints (largest model)
- ❌ Deployment without spatial metadata

**Enhanced Model (`EnhancedModel`)**:
- ✅ **Visual quality is primary bottleneck** (severely degraded inscriptions)
- ✅ **Language context is known** (script and language can be provided)
- ✅ **Deployment efficiency** important (fastest inference)
- ✅ **Standard images + optional Korniienko** data available
- ❌ No language/script metadata available
- ❌ Multi-task attribution required

**Multichannel Model (`MultiChannelModel`)**:
- ✅ **Multiple image sources are available** (different lighting, angles, reference materials)
- ✅ **Maximum visual information extraction** needed
- ✅ **Human-readable language inputs** preferred (string-based interface)
- ✅ **Autoregressive generation** workflows required
- ❌ Limited image data variety
- ❌ Computational constraints (complex spatial attention)

### Performance Trade-offs

| Metric | Transformer | Enhanced | Multichannel |
|---|---|---|---|
| **CER (Character Error Rate)** | 9.65% (best) | ~11-12% | ~10-11% |
| **Inference Speed** | Slowest | Fastest | Medium |
| **Memory Usage** | Highest | Lowest | Medium |
| **Data Requirements** | Complete multi-modal | Images + language context | Multiple images focused |
| **Deployment Complexity** | Complex | Simple | Medium |

## Limitations and Future Research Directions

### Current Limitations

1. **Limited Spatial Context Modeling**
   - Enhanced/Multichannel models lack explicit spatial encoders
   - No architectural understanding (pillar, wall, altar relationships)
   - Missing temporal layering of inscriptions

2. **Single-Scale Processing**
   - Current models process inscriptions at fixed resolution
   - No multi-resolution analysis for different inscription sizes
   - Limited handling of very long inscription sequences

3. **Language Modeling Constraints**
   - Character-level modeling limits understanding of historical abbreviations
   - No explicit modeling of scribal conventions and paleographic features
   - Limited cross-lingual transfer learning

### Future Research Directions

#### 1. Vision-Language Pre-training for Historical Texts

**Approach**: Self-supervised pre-training on large collections of historical manuscripts
- **Masked character modeling**: Predict masked characters from visual context
- **Image-text alignment**: Align manuscript images with transcribed text
- **Temporal modeling**: Learn chronological patterns in script evolution

#### 2. Hierarchical Multi-Resolution Architecture

**Innovation**: Process inscriptions at multiple scales simultaneously
- **Character-level**: Fine-grained feature extraction for individual letters
- **Word-level**: Capture spacing and linguistic patterns
- **Inscription-level**: Overall layout and structure understanding

#### 3. Paleographic Feature Learning

**Goal**: Explicitly model scribal hands and writing style evolution
- **Embedding scribal features**: Learn representations of individual scribes
- **Chronological regularization**: Enforce smooth evolution of script features
- **Cross-dataset transfer**: Apply learned features to other historical corpora

#### 4. Uncertainty Quantification and Active Learning

**Application**: Guide expert annotation efforts efficiently  
- **Epistemic uncertainty**: Model uncertainty in character predictions
- **Active learning**: Select most informative samples for human annotation
- **Confidence-aware transcription**: Highlight uncertain predictions for expert review

#### 5. 3D Geometric Integration

**Vision**: Incorporate cathedral 3D models for enhanced spatial understanding
- **3D spatial encoding**: Use cathedral mesh geometry for location context
- **Viewing angle normalization**: Account for photograph perspective distortion
- **Multi-view fusion**: Combine multiple viewpoints of same inscription

## Conclusion

The SOPHIA Epigraphic AI models represent a significant advance in automated historical text recognition, achieving 9.65% CER through innovative multi-modal fusion architectures. Our three-model approach addresses different deployment scenarios while maintaining state-of-the-art performance through:

1. **Domain-specific innovations**: Character-level tokenization, RTI-aware processing, language conditioning
2. **Robust attention mechanisms**: Cross-modal, spatial, and channel attention for degraded inscription analysis  
3. **Missing modality handling**: Graceful degradation when imaging pipelines are incomplete
4. **Multi-task learning**: Joint transcription and historical attribution for comprehensive analysis

These contributions establish a foundation for scaling automated epigraphic analysis to other historical sites and manuscript collections, accelerating digital humanities research and cultural heritage preservation.

## References

- **Assael et al.** (2022). "Restoring and attributing ancient texts using deep neural networks." *Nature* 603, 280-283. (ITHACA)
- **Google DeepMind** — Aeneas ("Predicting the Past"): Multi-modal transformer for ancient text restoration
- **Kenton & Toutanova** (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Dosovitskiy et al.** (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **Cultural Heritage Imaging** — Reflectance Transformation Imaging (RTI) methodology and best practices
