import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,pathlib,hashlib,time
import numpy as np
from sklearn.datasets import load_digits
import train_export as train
from integer_reference import infer
R=pathlib.Path(__file__).resolve().parent
def dump(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 if (R/'report.json').exists():raise RuntimeError('Refuse to overwrite prior evaluation')
 admission=json.load(open(R/'BOUND_ADMISSION.json'));assert admission['admitted']
 assert hashlib.sha256((R/'model.json').read_bytes()).hexdigest()==admission['model_sha256']
 assert hashlib.sha256((R/'float_model.json').read_bytes()).hexdigest()==admission['float_model_sha256']
 m=json.load(open(R/'model.json'));f=json.load(open(R/'float_model.json'));s=json.load(open(R/'split.json'))
 ds=load_digits(n_class=10);ids=s['test_indices'];assert ids[:4]==[1329,1379,1319,909]
 t=train.pack([np.asarray(f[n]) for n in train.NAMES]);fl=train.forward(t,(ds.images[ids]/16)[:,None,:,:]);fp=fl.argmax(axis=1)
 inputs=[];outputs=[];selected=[];start=time.time()
 for j,i in enumerate(ids):
  x=(ds.data[i].astype(np.int64)*64).tolist();v=infer(m,x,True)
  row={'dataset_index':i,'input_q10':x,'label':int(ds.target[i])};o={'dataset_index':i,**v}
  inputs.append(row);outputs.append(o)
  if j<4:selected.append({'input':row,'integer_output':o,'float_prediction':int(fp[j])})
 ip=np.array([o['prediction'] for o in outputs]);labels=ds.target[ids]
 dump('test_inputs.json',inputs);dump('integer_test_outputs.json',outputs);dump('selected_examples.json',selected)
 dump('dataset.json',{'pixels':ds.data.astype(np.int64).tolist(),'labels':ds.target.astype(np.int64).tolist()})
 report={'test_count':len(ids),'float_correct':int((fp==labels).sum()),'integer_correct':int((ip==labels).sum()),'float_accuracy':float((fp==labels).mean()),'integer_accuracy':float((ip==labels).mean()),'prediction_disagreements':int((fp!=ip).sum()),'integer_wrong_indices':[int(i) for i,ok in zip(ids,ip==labels) if not ok],'float_wrong_indices':[int(i) for i,ok in zip(ids,fp==labels) if not ok],'selected_examples':ids[:4],'scalar_integer_evaluation_seconds':time.time()-start,'model_sha256_before_evaluation':admission['model_sha256'],'float_model_sha256_before_evaluation':admission['float_model_sha256'],'all_test_cases_kept':True,'test_tuning':False}
 dump('report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
