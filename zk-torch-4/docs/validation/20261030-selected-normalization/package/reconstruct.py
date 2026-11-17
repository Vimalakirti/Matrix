#!/usr/bin/env python3
"""Reconstruct the450-file final source from the frozen InstanceNorm base.
Usage: python reconstruct.py BASE DEST
BASE is read-only. DEST must be new and outside resolved BASE and this bundle.
"""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
from verify_bundle import verify_bundle,verify_source,check_rows

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('base',type=Path);parser.add_argument('dest',type=Path)
args=parser.parse_args();base=args.base.resolve(strict=True);dest=args.dest.resolve(strict=False)
bundle=Path(__file__).resolve().parent
if not base.is_dir():raise SystemExit('BASE must be a directory')
for source,label in [(base,'BASE'),(bundle,'bundle')]:
    if dest==source or source in dest.parents:
        raise SystemExit(f'DEST must be outside resolved {label}')
if dest.exists():raise SystemExit('DEST must not already exist')
verify_bundle(bundle)
# Check the exact expected base bytes before copying or applying a patch.
full=json.loads((bundle/'full-source-manifest.json').read_text())
delta={x['path']:x for x in json.loads((bundle/'delta-manifest.json').read_text())}
expected=[]
for row in full:
    old=delta.get(row['path'])
    digest=old['base_sha256'] if old else row['sha256']
    if digest is None:
        if (base/row['path']).exists():raise SystemExit(f'BASE already contains new file: {row["path"]}')
    else:expected.append({'path':row['path'],'sha256':digest})
check_rows(base,expected)
shutil.copytree(base,dest,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
patch=bundle/'selected-normalization-after-instance.patch'
subprocess.run(['git','apply','--check',str(patch)],cwd=dest,check=True)
subprocess.run(['git','apply',str(patch)],cwd=dest,check=True)
count=verify_source(bundle,dest)
print(f'PASS reconstructed source: {count} required SHA256 checks; BASE unchanged; DEST={dest}')
