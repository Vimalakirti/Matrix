"""Regenerate fixed public CNN fixture in a NEW local directory, then verify pins."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('output',type=pathlib.Path);a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent;out=a.output.resolve()
if out.exists():raise SystemExit('Output must not exist: refuses overwrite or selection among old runs')
# Pinned versions are an experiment identity, not a claim of portable float bit identity.
import numpy,scipy,sklearn,joblib,threadpoolctl
actual={x.__name__:x.__version__ for x in [numpy,scipy,sklearn,joblib,threadpoolctl]}
expected={'numpy':'2.3.5','scipy':'1.17.0','sklearn':'1.8.0','joblib':'1.5.3','threadpoolctl':'3.6.0'}
if actual!=expected:raise SystemExit(f'Version mismatch: {actual}, expected {expected}')
out.mkdir(parents=True)
for f in ('training_plan.json','train_export.py','evaluate_frozen.py','integer_reference.py'):shutil.copy2(here/f,out/f)
env=os.environ.copy()
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
for script in ('train_export.py','evaluate_frozen.py'):
 with (out/(script+'.stdout')).open('wb') as log:subprocess.run([sys.executable,str(out/script)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
pins=json.loads((here/'expected_hashes.json').read_text());checks={}
for name,h in pins.items():
 actual=hashlib.sha256((out/name).read_bytes()).hexdigest();checks[name]={'expected':h,'actual':actual,'match':actual==h}
(out/'REPRODUCTION_CHECK.json').write_text(json.dumps(checks,indent=2)+'\n')
if not all(c['match'] for c in checks.values()):raise SystemExit('Reproduction hash mismatch; preserve output and investigate, never substitute it')
print('PASS: all seven frozen artifacts reproduced; diagnostics retained at',out)
