#!/usr/bin/env python3
"""Reconstruct exactly 455 source files from the authenticated 454-file CNN base.
Copies only manifest-listed source paths, not Git metadata or published evidence.
Uses an exact standard-library unified-patch applier: no shell or patch fuzz.
"""
import argparse,hashlib,json,re,sys
sys.dont_write_bytecode=True
from pathlib import Path,PurePosixPath
BUNDLE=Path(__file__).resolve().parent
BASE_MANIFEST_SHA='da3f4f1bae5927b06547d61bd061725a46634db1cb944df469da532084770113'
FINAL_MANIFEST_SHA='a57be8f20529556584fef5a97d8bf24c54d77809835890bd4e2678848c227579'
PATCH_SHA='6e755beca69b513a068b3337c3eae5b64c1ee5cc1435b62121967d8689b54666'
CHANGED={'zk-torch-4/src/morpheus/piop/cv.rs','zk-torch-4/src/morpheus/piop/mod.rs','zk-torch-4/src/morpheus/piop/cv_batch_tests.rs'}
def sha(v):return hashlib.sha256(v).hexdigest()
def safe_rel(s):
    p=PurePosixPath(s)
    if not s or not p.parts or p.is_absolute() or str(p)!=s or '..' in p.parts or '\\' in s or '\x00' in s:raise ValueError('unsafe relative path')
    return s

def manifest(name,expected):
    raw=(BUNDLE/'evidence'/name).read_bytes()
    if sha(raw)!=expected:raise ValueError('manifest identity mismatch: '+name)
    rows=json.loads(raw);result={}
    for x in rows:
        rel=safe_rel(x['path'])
        if rel in result or not re.fullmatch('[0-9a-f]{64}',x['sha256']):raise ValueError('duplicate or malformed manifest entry')
        result[rel]=x['sha256']
    return result

def source_bytes(root,rel):
    f=root/rel
    for p in [f,*f.parents]:
        if p==root:break
        if p.is_symlink():raise ValueError('source symlink is not supported: '+rel)
    if not f.is_file() or root not in f.resolve(strict=True).parents:raise ValueError('source missing or outside root: '+rel)
    return f.read_bytes()

def patched_files(original,text):
    lines=text.splitlines(keepends=True);i=0;out={}
    while i<len(lines):
        if not lines[i].startswith('--- '):raise ValueError('unexpected patch header')
        oldname=lines[i][4:].rstrip('\n');i+=1
        if i==len(lines) or not lines[i].startswith('+++ b/'):raise ValueError('unexpected patch target')
        rel=safe_rel(lines[i][6:].rstrip('\n'));i+=1
        if rel not in CHANGED or rel in out or oldname not in ('a/'+rel,'/dev/null'):raise ValueError('unexpected or duplicate patch path')
        if (oldname=='/dev/null') != (rel not in original):raise ValueError('patch create/update mismatch')
        src=original.get(rel,b'').decode().splitlines(keepends=True);cursor=0;result=[];hunks=0
        while i<len(lines) and lines[i].startswith('@@ '):
            m=re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n',lines[i]);i+=1
            if not m:raise ValueError('bad hunk header')
            a,ac,b,bc=int(m[1]),int(m[2] or 1),int(m[3]),int(m[4] or 1)
            pos=a-1 if ac else a
            if pos<cursor or pos>len(src):raise ValueError('overlapping/out-of-bounds hunk')
            result.extend(src[cursor:pos]);cursor=pos
            if len(result)!=(b-1 if bc else b):raise ValueError('new hunk offset mismatch')
            used_old=used_new=0
            while used_old<ac or used_new<bc:
                if i==len(lines) or lines[i][:1] not in (' ','+','-'):raise ValueError('missing hunk body')
                tag,payload=lines[i][0],lines[i][1:];i+=1
                if tag in (' ','-'):
                    if cursor>=len(src) or src[cursor]!=payload:raise ValueError('exact old-line mismatch')
                    cursor+=1;used_old+=1
                if tag in (' ','+'):
                    result.append(payload);used_new+=1
                if used_old>ac or used_new>bc:raise ValueError('hunk count overflow')
            hunks+=1
        if not hunks:raise ValueError('empty patch file')
        result.extend(src[cursor:]);out[rel]=''.join(result).encode()
    if set(out)!=CHANGED:raise ValueError('patch file set mismatch')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);args=p.parse_args()
    base=args.base.resolve(strict=True);dst=args.destination.resolve(strict=False)
    if not base.is_dir() or args.destination.is_symlink() or dst.exists():raise ValueError('base must exist; destination must be fresh')
    for protected in (base,BUNDLE):
        if dst==protected or protected in dst.parents or dst in protected.parents:raise ValueError('destination overlaps source/evidence')
    bm=manifest('base-source-manifest.json',BASE_MANIFEST_SHA);fm=manifest('final-source-manifest.json',FINAL_MANIFEST_SHA)
    raw_patch=(BUNDLE/'evidence/conv-overlay.patch').read_bytes()
    if sha(raw_patch)!=PATCH_SHA:raise ValueError('patch identity mismatch')
    if len(bm)!=454 or len(fm)!=455:raise ValueError('source manifest cardinality')
    original={}
    for rel,digest in bm.items():
        data=source_bytes(base,rel)
        if sha(data)!=digest:raise ValueError('base source hash mismatch: '+rel)
        original[rel]=data
    final=dict(original);final.update(patched_files(original,raw_patch.decode()))
    if set(final)!=set(fm) or any(sha(v)!=fm[k] for k,v in final.items()):raise ValueError('final reconstruction mismatch')
    # Validate everything before the first destination write. Source is never edited.
    dst.mkdir(parents=True)
    for rel,data in final.items():
        f=dst/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
    if any(sha(source_bytes(dst,k))!=v for k,v in fm.items()):raise ValueError('written source verification failed')
    print(json.dumps({'base_files_authenticated':454,'final_files_verified':455,'copied_only_manifest_sources':True,'final_manifest_sha256':FINAL_MANIFEST_SHA,'patch_sha256':PATCH_SHA,'destination':str(dst)}))
if __name__=='__main__':main()
