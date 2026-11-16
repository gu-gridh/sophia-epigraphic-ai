#!/usr/bin/env python3
"""
Synthetic incised-inscription generator for pretraining
=======================================================

Renders text with historical fonts and turns it into a *height map* of letters scratched
into a plaster/stone surface. From that one height map it derives every input view the
models use, so all of them get pretraining data:

    original  - lit surface photo (albedo x shading, dirt in the grooves, noise, JPEG)
    normal    - RGB normal map computed from the height map (like the RTI normal layer)
    blended   - mix of photo and raking-light shading
    texture   - high-pass detail map
    photo     - "Korniienko" archival photograph look (grey, contrast, grain)
    drawing   - clean tracing of the strokes (like a Korniienko drawing)

Text comes from transcriptions in one or more CSVs. To limit memorisation, lines are
re-composed: random word sequences, short fragments, and random character strings drawn
from each language's character distribution. Exact source lines are never reproduced
unless --verbatim_frac > 0, and lines in --exclude_csv are always skipped.

IMPORTANT for honest evaluation: build the text pool only from data the model may train
on (e.g. the training split), never from test transcriptions.

Example:
    python scripts/generate_synthetic.py \\
        --text_csv data/train_comprehensive.csv \\
        --exclude_csv data/test_comprehensive.csv \\
        --out_dir data/synthetic --data_dir data --num 20000 --workers 8
"""

import argparse
import csv
import glob
import html
import math
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features
from scipy import ndimage

try:
    from fontTools.ttLib import TTFont
except ImportError:  # pragma: no cover
    TTFont = None

SYSTEM_FONT_GLOBS = [
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
    '/System/Library/Fonts/Supplemental/Georgia.ttf',
    '/System/Library/Fonts/Supplemental/NotoSansGlagolitic-Regular.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
    '/usr/share/fonts/**/*.ttf',
    '/usr/share/fonts/**/*.otf',
]

# ----------------------------------------------------------------------------- text


def clean(text):
    text = str(text)
    for _ in range(3):
        new = html.unescape(text)
        if new == text:
            break
        text = new
    return unicodedata.normalize('NFC', text).strip()


def read_texts(paths, column='transcription_clean'):
    rows = []
    for p in paths:
        with open(p, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                t = clean(r.get(column) or r.get('target') or '')
                if t and t.lower() != 'nan':
                    rows.append((t, r.get('language_name') or 'unknown',
                                 r.get('writing_system_name') or 'unknown'))
    return rows


class TextSampler:
    """Re-composes lines per language from words, fragments and character statistics."""

    def __init__(self, rows, exclude=(), verbatim_frac=0.0, seed=0, renderable=None):
        """renderable: set of code points some font can draw (others are never sampled)."""
        self.rng = random.Random(seed)
        ok = (lambda ch: ch.isspace() or (ord(ch) in renderable and unicodedata.category(ch) != 'Co')) \
            if renderable is not None else (lambda ch: True)
        self.exclude = set(exclude)
        self.verbatim_frac = verbatim_frac
        self.by_lang = defaultdict(list)
        for t, lang, ws in rows:
            self.by_lang[(lang, ws)].append(t)
        self.keys = list(self.by_lang)
        self.weights = [len(self.by_lang[k]) for k in self.keys]
        self.words, self.chars, self.lengths = {}, {}, {}
        for k, texts in self.by_lang.items():
            self.words[k] = [w for t in texts for w in t.split() if w and all(ok(ch) for ch in w)]
            c = Counter(ch for t in texts for ch in t if not ch.isspace() and ok(ch))
            self.chars[k] = (list(c), list(c.values()))
            self.lengths[k] = [len(t) for t in texts]

    def sample(self):
        for _ in range(20):
            k = self.rng.choices(self.keys, self.weights)[0]
            target_len = max(1, self.rng.choice(self.lengths[k]) + self.rng.randint(-3, 3))
            mode = self.rng.random()
            if mode < self.verbatim_frac:
                text = self.rng.choice(self.by_lang[k])
            elif mode < 0.55 and self.words[k]:
                words = []
                while len(' '.join(words)) < target_len:
                    words.append(self.rng.choice(self.words[k]))
                text = ' '.join(words)
            elif mode < 0.75:
                src = self.rng.choice(self.by_lang[k])
                a = self.rng.randint(0, max(0, len(src) - 1))
                text = src[a:a + self.rng.randint(1, 24)].strip()
            else:
                chars, w = self.chars[k]
                n = min(target_len, 40)
                text = ''.join(self.rng.choices(chars, w, k=n))
                # sprinkle spaces like real word boundaries
                text = ''.join(ch if self.rng.random() > 0.12 else ' ' for ch in text).strip()
            text = ' '.join(text.split())[:160]
            if text and text not in self.exclude:
                return text, k[0], k[1]
        return text, k[0], k[1]


# ----------------------------------------------------------------------------- fonts


def find_fonts(font_dirs):
    paths = []
    for d in font_dirs:
        paths += glob.glob(os.path.join(d, '**', '*.otf'), recursive=True)
        paths += glob.glob(os.path.join(d, '**', '*.ttf'), recursive=True)
    for g in SYSTEM_FONT_GLOBS:
        paths += glob.glob(g, recursive=True)
    unique = {}
    for p in sorted(paths):
        unique.setdefault(os.path.basename(p).lower(), p)  # same font in several folders
    return sorted(unique.values())


def font_coverage(paths):
    cov = {}
    for p in paths:
        try:
            cmap = TTFont(p, lazy=True).getBestCmap() if TTFont else None
            cov[p] = set(cmap) if cmap else None
        except Exception:
            continue
    return cov


def is_private_use(ch):
    return unicodedata.category(ch) == 'Co'


def fonts_for(text, coverage):
    # Private Use Area code points mean different things in every font (the transcribers'
    # font vs. the rendering font), so such lines are never rendered.
    if any(is_private_use(c) for c in text):
        return []
    needed = {ord(c) for c in text if not c.isspace()}
    return [p for p, cps in coverage.items() if cps is not None and needed <= cps]


# ----------------------------------------------------------------------------- rendering


def fractal_noise(h, w, rng, octaves=5, persistence=0.55):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        sigma = max(h, w) / (2 ** (o + 1))
        out += amp * ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma)
        total += amp
        amp *= persistence
    out /= total
    return (out - out.mean()) / (out.std() + 1e-6)


def elastic(mask, rng, alpha, sigma):
    h, w = mask.shape
    dx = ndimage.gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return ndimage.map_coordinates(mask, [y + dy, x + dx], order=1, mode='constant')


def scratch_style(m, rng):
    """
    Turn typeset glyphs into hand-scratched single strokes: approximate the medial axis
    (ridge of the distance transform), then redraw it with an uneven 1-4 px tool and
    strong wobble, like letters scratched into plaster.
    """
    solid = m > 0.5
    dist = ndimage.distance_transform_edt(solid)
    ridge = solid & (dist >= ndimage.maximum_filter(dist, size=3)) & (dist > 0.5)
    ridge = ndimage.binary_dilation(ridge, iterations=1)  # reconnect the approximate skeleton
    line = ridge.astype(np.float32)
    width = rng.uniform(0.8, 2.2)
    line = ndimage.gaussian_filter(line, width) > rng.uniform(0.12, 0.25)
    return elastic(line.astype(np.float32), rng, alpha=rng.uniform(2.0, 5.0), sigma=rng.uniform(3, 6))


def render_text_mask(text, font_path, rng):
    """Grayscale [0,1] stroke mask of the text, possibly broken over 1-3 lines."""
    size = int(rng.integers(36, 72))
    layout = ImageFont.Layout.RAQM if features.check('raqm') else ImageFont.Layout.BASIC
    font = ImageFont.truetype(font_path, size, layout_engine=layout)
    words = text.split(' ')
    n_lines = 1 if len(text) < 14 else int(rng.choice([1, 1, 1, 1, 2, 3]))  # real graffiti: mostly one long line
    n_lines = min(n_lines, len(words))
    per = math.ceil(len(words) / n_lines)
    lines = [' '.join(words[i:i + per]) for i in range(0, len(words), per)]
    spacing = int(size * rng.uniform(0.15, 0.6))
    probe = ImageDraw.Draw(Image.new('L', (8, 8)))
    boxes = [probe.textbbox((0, 0), ln, font=font) for ln in lines]
    w = max(b[2] - b[0] for b in boxes) + 2 * size
    h = sum(b[3] - b[1] for b in boxes) + spacing * (len(lines) - 1) + 2 * size
    img = Image.new('L', (int(w), int(h)), 0)
    d = ImageDraw.Draw(img)
    y = size
    for ln, b in zip(lines, boxes):
        x = size + int(rng.integers(0, max(1, size // 2)))  # ragged left edge
        d.text((x - b[0], y - b[1]), ln, fill=255, font=font)
        y += (b[3] - b[1]) + spacing
    m = np.asarray(img, np.float32) / 255.0
    # scratched strokes: thin or thicken a little, then hand-made wobble
    k = int(rng.integers(-1, 2))
    if k < 0:
        m = ndimage.grey_erosion(m, size=(2, 2))
    elif k > 0:
        m = ndimage.grey_dilation(m, size=(2, 2))
    if rng.random() < 0.6:
        m = scratch_style(m, rng)
    m = elastic(m, rng, alpha=rng.uniform(1.0, 3.5), sigma=rng.uniform(4, 9))
    return np.clip(m, 0, 1)


def distractors(h, w, rng):
    """Random scratches / strokes that are not text (other graffiti, cracks)."""
    img = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(0, 12))):
        pts = [(float(rng.uniform(0, w)), float(rng.uniform(0, h)))]
        for _ in range(int(rng.integers(1, 5))):
            x, y = pts[-1]
            pts.append((x + float(rng.normal(0, w / 6)), y + float(rng.normal(0, h / 6))))
        d.line(pts, fill=int(rng.integers(80, 255)), width=int(rng.integers(1, 3)))
    return np.asarray(img, np.float32) / 255.0


def to_uint8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def synthesize(text, font_path, rng):
    m = render_text_mask(text, font_path, rng)
    # random rotation of the whole inscription
    m = ndimage.rotate(m, float(rng.normal(0, 4)), reshape=True, order=1)
    h, w = m.shape
    scratch = distractors(h, w, rng)

    # ---- height map: rough surface minus incisions
    surface = fractal_noise(h, w, rng) * rng.uniform(0.4, 1.2)
    surface += fractal_noise(h, w, rng, octaves=2) * rng.uniform(0.5, 2.0)    # broad undulation
    depth = rng.uniform(1.5, 4.0)
    groove = ndimage.gaussian_filter(m, rng.uniform(0.6, 1.4))
    height = surface - depth * groove - 0.6 * depth * ndimage.gaussian_filter(scratch, 0.8)

    # ---- normals from the height map
    strength = rng.uniform(1.5, 3.0)
    gy, gx = np.gradient(ndimage.gaussian_filter(height, 0.7))
    n = np.stack([-gx * strength, -gy * strength, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal_rgb = to_uint8((n + 1) / 2)

    # ---- lit photo: albedo x lambert shading, dirt in grooves
    az, el = rng.uniform(0, 2 * np.pi), rng.uniform(np.radians(20), np.radians(70))
    light = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    shade = np.clip(n @ light, 0, 1)
    base = np.array([rng.uniform(0.55, 0.85), rng.uniform(0.5, 0.78), rng.uniform(0.4, 0.7)])
    tint = 0.08 * fractal_noise(h, w, rng, octaves=3)[..., None]
    albedo = base[None, None, :] + tint
    if rng.random() < 0.35:   # remains of painted fresco
        paint = ndimage.gaussian_filter(rng.random((h, w)), max(h, w) / 8) > 0.5
        colour = rng.uniform(0.2, 0.8, size=3)
        albedo = np.where(paint[..., None], 0.6 * albedo + 0.4 * colour, albedo)
    dirt = rng.uniform(0.0, 0.35) * groove[..., None]
    photo = albedo * (0.35 + 0.75 * shade[..., None]) - dirt
    photo += rng.normal(0, rng.uniform(0.005, 0.03), photo.shape)
    photo = to_uint8(photo)

    # raking-light shading for 'blended'
    rake = np.array([np.cos(az), np.sin(az), 0.25]); rake /= np.linalg.norm(rake)
    rake_shade = np.clip(n @ rake, 0, 1)
    blended = to_uint8(0.5 * photo / 255.0 + 0.5 * rake_shade[..., None])

    grey = photo.mean(axis=-1) / 255.0
    texture = to_uint8(np.repeat((0.5 + 2.5 * (grey - ndimage.gaussian_filter(grey, 3)))[..., None], 3, -1))

    # Korniienko-style archival photo: monochrome, contrasty, grain, vignette
    k = grey * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)
    k = np.clip((k - 0.5) * rng.uniform(1.1, 1.7) + 0.5, 0, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1 - 0.35 * (((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    k = k * vign + rng.normal(0, 0.04, k.shape)
    sepia = np.array([1.0, 0.95, 0.85]) if rng.random() < 0.5 else np.ones(3)
    k_photo = to_uint8(k[..., None] * sepia)

    # Korniienko-style drawing: like the published tracings, mostly WHITE strokes on a black
    # ground (e.g. Korniienko_v1_137_3b_drawing_1002.png); some dark-on-light for variety.
    stroke = elastic(m, rng, alpha=1.0, sigma=6) > rng.uniform(0.35, 0.6)
    if rng.random() < 0.3:   # thick blobby spots, like damage marked in the tracings
        spots = ndimage.gaussian_filter(rng.random((h, w)), 2) > 0.62
        stroke |= spots & (ndimage.gaussian_filter(m, 6) > 0.05)
    ground = 0.04 * fractal_noise(h, w, rng, octaves=2)
    if rng.random() < 0.85:
        drawing = np.where(stroke, rng.uniform(0.85, 1.0), 0.05 + ground)
    else:
        drawing = np.where(stroke, rng.uniform(0.05, 0.25), 0.92 + ground)
    drawing = to_uint8(np.repeat(drawing[..., None], 3, -1))

    return {'original': photo, 'blended': blended, 'normal': normal_rgb, 'texture': texture,
            'photo': k_photo, 'drawing': drawing}


def save(arr, path, rng, jpeg=True):
    img = Image.fromarray(arr)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
    # keep files small: longest side at most 768 px
    s = 768 / max(img.size)
    if s < 1:
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    if jpeg:
        img.save(path, quality=int(rng.integers(70, 95)))
    else:
        img.save(path)


# ----------------------------------------------------------------------------- workers

_G = {}


def _init(coverage, out_dir, rel_root):
    _G.update(coverage=coverage, out_dir=out_dir, rel_root=rel_root)


def _work(job):
    idx, text, lang, ws, seed = job
    rng = np.random.default_rng(seed)
    candidates = fonts_for(text, _G['coverage'])
    if not candidates:
        return None
    font = candidates[int(rng.integers(len(candidates)))]
    try:
        views = synthesize(text, font, rng)
    except Exception as e:  # pragma: no cover - a broken glyph should not stop the run
        return ('error', str(e))
    row = {'id': f'syn{idx:07d}', 'transcription_clean': text, 'language_name': lang,
           'writing_system_name': ws, 'iiif_crop': '', 'font': os.path.basename(font)}
    sub = f'{idx // 1000:04d}'
    os.makedirs(os.path.join(_G['out_dir'], sub), exist_ok=True)
    for view, arr in views.items():
        name = f'syn{idx:07d}_{view}.' + ('png' if view == 'normal' else 'jpg')
        path = os.path.join(_G['out_dir'], sub, name)
        save(arr, path, rng, jpeg=view != 'normal')
        rel = os.path.relpath(path, _G['rel_root'])
        col = {'photo': 'korniienko_photo', 'drawing': 'korniienko_drawing'}.get(view, f'{view}_image')
        row[col] = rel
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text_csv', nargs='+', required=True, help='CSV(s) with transcription_clean / language_name')
    ap.add_argument('--exclude_csv', nargs='*', default=[], help='Never generate these exact lines (e.g. test set)')
    ap.add_argument('--out_dir', default='data/synthetic')
    ap.add_argument('--data_dir', default='data', help='Image paths in the CSV are relative to this')
    ap.add_argument('--font_dirs', nargs='+', default=['data/fonts'])
    ap.add_argument('--num', type=int, default=20000)
    ap.add_argument('--val_frac', type=float, default=0.05)
    ap.add_argument('--verbatim_frac', type=float, default=0.0,
                    help='Share of exact source lines (0 = only re-composed text)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rows = read_texts(args.text_csv)
    exclude = {t for t, _, _ in read_texts(args.exclude_csv)} if args.exclude_csv else set()
    print(f"Text pool: {len(rows)} lines, {len(exclude)} excluded lines")
    fonts = find_fonts(args.font_dirs)
    coverage = font_coverage(fonts)
    print(f"Fonts: {len(coverage)} usable ({', '.join(sorted(os.path.basename(p) for p in coverage))[:300]}...)")
    if not coverage:
        sys.exit("No fonts found. Put OFL fonts (e.g. CTAN fonts-churchslavonic) in --font_dirs.")

    renderable = set().union(*[c for c in coverage.values() if c])
    sampler = TextSampler(rows, exclude, args.verbatim_frac, args.seed, renderable)
    jobs = []
    for i in range(args.num):
        t, lang, ws = sampler.sample()
        jobs.append((i, t, lang, ws, args.seed * 10_000_000 + i))

    os.makedirs(args.out_dir, exist_ok=True)
    out_rows, skipped, errors = [], 0, 0
    with Pool(args.workers, initializer=_init, initargs=(coverage, args.out_dir, args.data_dir)) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=16), 1):
            if r is None:
                skipped += 1
            elif isinstance(r, tuple):
                errors += 1
            else:
                out_rows.append(r)
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)} done", flush=True)

    out_rows.sort(key=lambda r: r['id'])
    random.Random(args.seed).shuffle(out_rows)
    n_val = max(1, int(len(out_rows) * args.val_frac))
    cols = ['id', 'transcription_clean', 'language_name', 'writing_system_name', 'original_image',
            'blended_image', 'normal_image', 'texture_image', 'korniienko_photo', 'korniienko_drawing',
            'iiif_crop', 'font']
    for name, part in (('val.csv', out_rows[:n_val]), ('train.csv', out_rows[n_val:])):
        with open(os.path.join(args.out_dir, name), 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(part)
    print(f"Wrote {len(out_rows)} samples ({n_val} val). Skipped {skipped} lines with no covering font, "
          f"{errors} render errors. CSVs in {args.out_dir}")


if __name__ == '__main__':
    main()
