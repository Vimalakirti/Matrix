#!/usr/bin/env python3
import subprocess,sys,tempfile
from pathlib import Path
script=Path(__file__).with_name('prepare_harness.py');source=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as d:
    root=Path(d);existing=root/'existing';existing.mkdir();link=root/'source-link';link.symlink_to(source,target_is_directory=True)
    cases=[(source,source),(source,source/'forbidden'),(source,source.parent),(source,existing),(link,link/'forbidden'),(source,link)]
    for src,dst in cases:
        out=subprocess.run([sys.executable,str(script),str(src),str(dst)],capture_output=True,text=True)
        assert out.returncode!=0,(src,dst)
    assert not (source/'forbidden').exists()
print('PASS 6 unsafe/overlapping/existing destination cases rejected before writing')
