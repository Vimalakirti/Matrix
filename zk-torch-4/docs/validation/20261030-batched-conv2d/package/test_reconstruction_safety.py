#!/usr/bin/env python3
import argparse,importlib.util,json,subprocess,sys,tempfile
sys.dont_write_bytecode=True
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve(strict=True)
script=Path(__file__).with_name('reconstruct.py');spec=importlib.util.spec_from_file_location('reconstruction',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=0
with tempfile.TemporaryDirectory() as d:
    root=Path(d);exists=root/'existing';exists.mkdir();alias=root/'alias';alias.symlink_to(base,target_is_directory=True);broken=root/'broken';broken.symlink_to(root/'missing')
    cases=[(base,base),(base,base/'forbidden'),(base,base.parent),(base,exists),(alias,alias/'forbidden'),(base,broken),(base,script.parent/'forbidden')]
    for src,dst in cases:
        r=subprocess.run([sys.executable,str(script),'--base',str(src),'--destination',str(dst)],capture_output=True,text=True)
        assert r.returncode!=0,(src,dst);checks+=1
    fake=root/'tampered';fake.mkdir();(fake/'.gitignore').write_text('tampered base')
    dst=root/'must-not-exist';r=subprocess.run([sys.executable,str(script),'--base',str(fake),'--destination',str(dst)],capture_output=True,text=True)
    assert r.returncode!=0 and 'base source hash mismatch' in r.stderr and not dst.exists();checks+=1
    fake2=root/'symlinked';fake2.mkdir();(fake2/'.gitignore').symlink_to(base/'.gitignore')
    r=subprocess.run([sys.executable,str(script),'--base',str(fake2),'--destination',str(dst)],capture_output=True,text=True)
    assert r.returncode!=0 and 'source symlink' in r.stderr and not dst.exists();checks+=1
    for unsafe in ['../x','/absolute','a/../x','a//x','a\\x','./x','', '.']:
        try:m.safe_rel(unsafe)
        except ValueError:pass
        else:raise AssertionError(unsafe)
        checks+=1
assert not (base/'forbidden').exists() and not (script.parent/'forbidden').exists()
print(json.dumps({'rejection_checks':checks,'all_passed':True,'source_unchanged':True}))
