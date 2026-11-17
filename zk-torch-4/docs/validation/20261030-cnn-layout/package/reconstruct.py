#!/usr/bin/env python3
"""Read-only base reconstruction. DEST must be new, outside resolved BASE/bundle."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
from verify_bundle import verify_bundle,verify_source,check_rows,safe_file
p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('dest',type=Path);p.add_argument('--variant',choices=['joint','q20'],default='joint');a=p.parse_args()
base=a.base.resolve(strict=True);dest=a.dest.resolve(strict=False);bundle=Path(__file__).resolve().parent
if not base.is_dir():raise SystemExit('BASE must be a directory')
for source,label in [(base,'BASE'),(bundle,'bundle')]:
    if dest==source or source in dest.parents:raise SystemExit(f'DEST must be outside resolved {label}')
if dest.exists():raise SystemExit('DEST must not already exist')
verify_bundle(bundle)
rows=json.loads((bundle/f'{a.variant}-base-manifest.json').read_text());check_rows(base,rows)
for row in json.loads((bundle/f'{a.variant}-delta-manifest.json').read_text()):
    if row['base_sha256'] is None and (base/row['path']).exists():raise SystemExit(f'BASE already has new file {row["path"]}')
# Copy exactly the checked source manifest, not unlisted files/symlink targets.
dest.mkdir(parents=True)
for row in rows:
    source=safe_file(base,row['path']);target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
patch=bundle/f'{a.variant}-layout.patch'
subprocess.run(['git','apply','--check',str(patch)],cwd=dest,check=True)
subprocess.run(['git','apply',str(patch)],cwd=dest,check=True)
print('PASS reconstructed',verify_source(bundle,dest,a.variant),'source files; BASE unchanged; DEST',dest)
