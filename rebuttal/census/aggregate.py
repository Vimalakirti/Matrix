#!/usr/bin/env python3
"""Aggregate the E2 bounds CSVs (one per model run).

Writes <out>/summary.csv (per model x check kind) and <out>/rescale_exceptions.csv
(uncertified rescales grouped by pattern), and prints a report. For the
6B-8B LLMs, run at 1/2/3 layers, it checks that the per-layer increments are
identical and extrapolates to the Table 1 depth."""
import csv, sys, glob, os, re, collections
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
FULL = {'llama2': 32, 'llama3': 32, 'gptj': 28}
STAT = ['certified', 'field-exact', 'SUPPORT-ONLY', 'UNCERTIFIED', 'VIOLATED']

def load(tag):
    return list(csv.DictReader(open(os.path.join(out, tag + '.csv'))))

def summarize(rs):
    s = collections.defaultdict(lambda: collections.Counter())
    mins = {}
    for r in rs:
        s[r['check']][r['status']] += 1
        if r['status'] == 'certified' and r['margin_bits']:
            mins[r['check']] = min(mins.get(r['check'], 1e9), float(r['margin_bits']))
    return s, mins

def pattern(r):
    n = r['note']
    n = re.sub(r'e\d+', 'e#', n)
    n = re.sub(r'\(\d+ of \d+ entries\)', '', n)
    x = re.search(r'x from: (.*?); y fate', n); y = re.search(r'y fate: (.*)$', n)
    return (r['status'], x.group(1) if x else '?', y.group(1) if y else '?')

tags = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(out, '*.csv')) if not os.path.basename(f) in ('summary.csv', 'rescale_exceptions.csv'))
models = collections.OrderedDict()
for t in tags:
    m = t.split('_NUM_LAYERS')
    models.setdefault(m[0], {})[int(m[1]) if len(m) > 1 else None] = t

sw = csv.writer(open(os.path.join(out, 'summary.csv'), 'w'))
sw.writerow(['model', 'layers', 'basis', 'check', 'total'] + STAT + ['min_margin_bits_certified'])
ew = csv.writer(open(os.path.join(out, 'rescale_exceptions.csv'), 'w'))
ew.writerow(['model', 'layers', 'count', 'status', 'x_from', 'y_fate'])
for m, runs in models.items():
    if None in runs:
        rs = load(runs[None]); s, mins = summarize(rs)
        layers, basis, table = '', 'full graph', {k: dict(v) for k, v in s.items()}
        pats = collections.Counter(pattern(r) for r in rs if r['check'] in ('rescale', 'upscale', 'rational-rescale') and r['status'] not in ('certified',))
    else:
        L = sorted(runs); assert L[:3] == [1, 2, 3], (m, L)
        S = {l: summarize(load(runs[l])) for l in L}
        checks = set().union(*[S[l][0].keys() for l in L])
        table, mins = {}, {}
        stationary = True
        for k in checks:
            row = {}
            for st in STAT:
                c = [S[l][0][k][st] for l in L]
                d1, d2 = c[1] - c[0], c[2] - c[1]
                if d1 != d2: stationary = False
                row[st] = c[0] + (FULL[m] - 1) * d2
            table[k] = row
            mm = [S[l][1].get(k) for l in L]
            if mm[1] != mm[2]: stationary = False
            if mm[2] is not None: mins[k] = min(x for x in mm if x is not None)
        layers, basis = FULL[m], 'extrapolated from 1/2/3-layer graphs (per-layer increments %s)' % ('identical' if stationary else 'NOT identical')
        rs3 = load(runs[3]); rs2 = load(runs[2]); rs1 = load(runs[1])
        p3 = collections.Counter(pattern(r) for r in rs3 if r['check'] in ('rescale', 'upscale', 'rational-rescale') and r['status'] != 'certified')
        p2 = collections.Counter(pattern(r) for r in rs2 if r['check'] in ('rescale', 'upscale', 'rational-rescale') and r['status'] != 'certified')
        pats = collections.Counter({k: p3[k] + (FULL[m] - 3) * (p3[k] - p2[k]) for k in p3})
    print('== %s %s (%s)' % (m, layers, basis))
    for k in sorted(table):
        row = table[k]; tot = sum(row.get(st, 0) for st in STAT)
        sw.writerow([m, layers, basis, k, tot] + [row.get(st, 0) for st in STAT] + ['%.2f' % mins[k] if k in mins else ''])
        print('  %-17s %7d  ' % (k, tot) + '  '.join('%s %d' % (st, row.get(st, 0)) for st in STAT if row.get(st, 0)) + ('  min margin %.2f bits' % mins[k] if k in mins else ''))
    for (st, x, y), c in pats.most_common():
        ew.writerow([m, layers, c, st, x, y])
        print('     %5d %-12s x from %-26s y -> %s' % (c, st, x, y))
