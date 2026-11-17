#!/usr/bin/env python3
"""Losslessly decode the published text transport into a NEW evidence directory.
Usage: python3 decode_bundle.py /path/to/new-evidence-directory
No installation, build, test, network or source modification is performed.
"""
import argparse, hashlib, json
from pathlib import Path, PurePosixPath
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
a=p.parse_args();here=Path(__file__).resolve().parent;out=a.output.resolve()
if out.exists():p.error('output must not exist')
if out==here or here in out.parents:p.error('output must be outside the publication evidence directory')
manifest=json.loads((here/'transport-manifest.json').read_bytes())
assert manifest['format']=='morpheus-lossless-text-bundle-v1'
def safe(name):
 q=PurePosixPath(name)
 assert not q.is_absolute() and '..' not in q.parts and '.' not in q.parts
 return Path(*q.parts)
def digest(b):return hashlib.sha256(b).hexdigest()
seen=set();decoded=[]
for r in manifest['files']:
 rel=safe(r['path']);assert str(rel) not in seen;seen.add(str(rel))
 data=(here/safe(r['storage'])).read_bytes();assert digest(data)==r['storage_sha256'],r['storage']
 if r['encoding']=='utf8-line-dictionary-v1':
  encoded=json.loads(data);assert encoded['format']=='utf8-line-dictionary-v1'
  lines=encoded['lines'];assert all(isinstance(x,str) for x in lines)
  order=encoded['order'];assert all(isinstance(i,int) and 0<=i<len(lines) for i in order)
  assert sum(len(lines[i].encode('utf-8')) for i in order)==r['bytes'],r['path']
  data=''.join(lines[i] for i in order).encode('utf-8')
 else:assert r['encoding']=='identity'
 assert len(data)==r['bytes'] and digest(data)==r['sha256'],r['path']
 decoded.append((rel,data,r['executable']))
out.mkdir(parents=True)
for rel,data,executable in decoded:
 f=out/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
 if executable:f.chmod(0o755)
print(json.dumps({'status':'pass','files':len(decoded),'decoded_bytes':sum(len(b)for _,b,_ in decoded),'output':str(out),'scope':'lossless byte reconstruction only; no tests executed'},sort_keys=True))
