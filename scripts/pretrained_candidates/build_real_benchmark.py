#!/usr/bin/env python3
"""
Build a small real benchmark from the public Saint Sophia API for ZERO-SHOT model comparison.

For a random, language-stratified sample of transcribed inscriptions it downloads the
Korniienko drawing and photo and writes a CSV in the training-pipeline format.

These inscriptions are part of the training corpus: use this set only to compare
pretrained models *before* fine-tuning, never as extra training data or as a test set
for models that were fine-tuned on the full corpus.

    python scripts/pretrained_candidates/build_real_benchmark.py --n 60 --out data/benchmark_real
"""

import argparse
import csv
import html
import json
import os
import random
import re
import time
import unicodedata
import urllib.request
from collections import defaultdict

API = 'https://saintsophia.dh.gu.se/api/inscriptions'
UA = {'User-Agent': 'sophia-epigraphic-ai benchmark builder'}


def get_json(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'failed: {url}')


def paged(url):
    while url:
        page = get_json(url)
        yield from page['results']
        url = page.get('next')
        if url:
            url = url.replace('http://', 'https://')


def clean_html(text):
    """'<p>a</p>\\n<p>b</p>' -> 'a b' (lines joined by a space), entities decoded, NFC."""
    if not text:
        return ''
    parts = [html.unescape(re.sub(r'<[^>]+>', '', p)).strip()
             for p in re.split(r'</p>|<br\s*/?>', text)]
    joined = ' '.join(p.replace('\xa0', ' ').strip() for p in parts if p.strip())
    return unicodedata.normalize('NFC', ' '.join(joined.split()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--out', default='data/benchmark_real')
    ap.add_argument('--min_len', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    languages = {l['id']: l.get('text') or l.get('name') or str(l['id'])
                 for l in paged(f'{API}/language/?format=json&limit=100')}
    by_inscription = defaultdict(dict)
    for img in paged(f'{API}/korniienko-image/?format=json&depth=1&limit=500'):
        ins = img.get('inscription') or {}
        text = clean_html(ins.get('transcription'))
        url = img.get('url') or ''
        kind = 'drawing' if '_drawing_' in url else 'photo' if '_photo_' in url else None
        if not (kind and len(text) >= args.min_len):
            continue
        entry = by_inscription[ins['id']]
        entry.setdefault(kind, url)
        entry.update(id=ins['id'], title=ins.get('title'), text=text,
                     language=languages.get(ins.get('language'), 'unknown'))

    candidates = [e for e in by_inscription.values() if 'drawing' in e]
    print(f'{len(candidates)} transcribed inscriptions with a Korniienko drawing')
    rng = random.Random(args.seed)
    by_lang = defaultdict(list)
    for e in candidates:
        by_lang[e['language']].append(e)
    # proportional sample, but at least 2 per language where possible
    sample = []
    for lang, items in by_lang.items():
        k = max(min(2, len(items)), round(args.n * len(items) / len(candidates)))
        sample += rng.sample(items, min(k, len(items)))
    rng.shuffle(sample)
    sample = sample[:args.n]

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for e in sample:
        row = {'id': e['id'], 'title': e['title'], 'transcription_clean': e['text'],
               'language_name': e['language'], 'writing_system_name': 'unknown',
               'korniienko_drawing': '', 'korniienko_photo': ''}
        for kind in ('drawing', 'photo'):
            if kind in e:
                name = os.path.basename(e[kind])
                path = os.path.join(img_dir, name)
                if not os.path.exists(path):
                    req = urllib.request.Request(e[kind], headers=UA)
                    with urllib.request.urlopen(req, timeout=120) as r, open(path, 'wb') as f:
                        f.write(r.read())
                row[f'korniienko_{kind}'] = os.path.relpath(path, args.out)
        rows.append(row)
    with open(os.path.join(args.out, 'benchmark.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    langs = defaultdict(int)
    for r in rows:
        langs[r['language_name']] += 1
    print(f'Wrote {len(rows)} inscriptions to {args.out}/benchmark.csv: {dict(langs)}')


if __name__ == '__main__':
    main()
