#!/usr/bin/env python3
"""Verify the self-contained evidence bundle and optionally all450 source files."""
import argparse,hashlib,json
from pathlib import Path

def safe_file(root,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'unsafe manifest path: {relative}')
    p=root/rel
    resolved=p.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f'path escapes root or is not a file: {relative}')
    return p

def check_rows(root,rows,bytes_required=False):
    seen=set()
    for row in rows:
        rel=row['path']
        if rel in seen:raise ValueError(f'duplicate manifest path: {rel}')
        seen.add(rel);data=safe_file(root,rel).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError(f'SHA256 mismatch: {rel}')
        if bytes_required and len(data)!=row['bytes']:
            raise ValueError(f'byte-count mismatch: {rel}')
    return seen

def verify_bundle(bundle):
    bundle=bundle.resolve(strict=True)
    manifest=json.loads((bundle/'export-manifest.json').read_text())
    expected=check_rows(bundle,manifest['files'],True)
    actual={str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()}
    if actual != expected|{'export-manifest.json'}:
        raise ValueError(f'bundle inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected-{"export-manifest.json"})}')
    return len(expected)

def verify_source(bundle,source):
    source=source.resolve(strict=True)
    return len(check_rows(source,json.loads((bundle/'full-source-manifest.json').read_text())))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='optional final reconstructed repository root')
    args=parser.parse_args();bundle=Path(__file__).resolve().parent
    print(f'PASS bundle: {verify_bundle(bundle)} files, exact inventory, SHA256 and byte counts')
    if args.source:print(f'PASS source: {verify_source(bundle,args.source)} required file SHA256 checks')
