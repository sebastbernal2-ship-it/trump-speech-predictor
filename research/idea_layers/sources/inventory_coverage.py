#!/usr/bin/env python3
"""Count retrospective source-date overlap, not verified availability or rights."""
import argparse
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('data_root', type=Path)
a = p.parse_args()
root = a.data_root
archives = root / 'source-archives'
days = {}
with (archives / 'abc-million-headlines-2003-2021/abcnews-date-text_train.csv').open() as f:
    days['abc'] = {date.fromisoformat(r['publish_date'][:4]+'-'+r['publish_date'][4:6]+'-'+r['publish_date'][6:8]) for r in csv.DictReader(f)}
with (archives / 'huffpost-news-category-2012-2018-v2/News_Category_Dataset_v2.json').open() as f:
    days['huffpost'] = {date.fromisoformat(json.loads(line)['date'][:10]) for line in f}
tweets = set()
for f in (archives / 'markhershey-complete-trump-tweets-8be27c439d0c').glob('*.csv'):
    with f.open(newline='') as h:
        tweets.update(date.fromisoformat(r[' Time'].strip()[:10]) for r in csv.DictReader(h))
days['twitter'] = tweets
with (archives / 'stiles-truth-social-cnn-feed-2026-11-05T0139Z/truth_archive.json').open() as f:
    days['truth'] = {date.fromisoformat(r['created_at'][:10]) for r in json.load(f) if r.get('created_at')}
views = ['rollcall-events.jsonl', 'events.jsonl', 'speeches.jsonl', 'rollcall-by-type/trump-long-speeches.jsonl']
for view in views:
    totals = Counter()
    years = {}
    with (root / view).open() as f:
        for line in f:
            row = json.loads(line)
            totals['rows'] += 1
            try:
                d = date.fromisoformat((row.get('event_date') or row.get('date') or '')[:10])
            except ValueError:
                totals['invalid_date'] += 1
                continue
            c = years.setdefault(d.year, Counter())
            c['rows'] += 1
            window = {d-timedelta(days=i) for i in range(1, 8)}
            hits = {name: bool(window & values) for name, values in days.items()}
            for name, hit in {**hits, 'any_news': hits['abc'] or hits['huffpost'], 'any_social': hits['twitter'] or hits['truth']}.items():
                c[name] += hit
                totals[name] += hit
    print(json.dumps({'view': view, 'total': dict(totals), 'years': {str(k): dict(v) for k,v in sorted(years.items())}}, sort_keys=True))
