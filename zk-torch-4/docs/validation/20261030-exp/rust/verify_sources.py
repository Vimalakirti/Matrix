#!/usr/bin/env python3
"""Verify recorded source files and original Rust test-body bytes (read-only)."""
import hashlib, json, pathlib, re, sys
root = pathlib.Path(sys.argv[1]).resolve()
here = pathlib.Path(__file__).resolve().parent
files = json.loads((here / 'source-sha256.json').read_text())
for rel, expected in files.items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, rel
records = json.loads((here / 'test-body-sha256.json').read_text())
for item in records:
    text = (root / item['source']).read_text()
    match = re.search(r'(?m)^( *)fn ' + re.escape(item['test']) + r'\(\) \{', text)
    assert match, item['test']
    start = match.end() - 1
    end = start + re.search(r'(?m)^' + match[1] + r'\}', text[start:]).end()
    assert hashlib.sha256(text[start:end].encode()).hexdigest() == item['body_sha256'], item['test']
print(f'PASS: {len(files)} source files and {len(records)} original test bodies match recorded SHA-256s')
