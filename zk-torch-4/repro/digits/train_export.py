"""One declared fit; rerun into a new output directory to compare frozen artifacts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, platform, warnings
from pathlib import Path
import numpy as np
import scipy, sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from integer_reference import infer

def dump(root, name, data):
    (root/name).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(exist_ok=True,parents=True)
    plan_path=Path(__file__).with_name('training_plan.json')
    plan=json.loads(plan_path.read_text())
    d=load_digits(n_class=10); pixels=d.data.astype(np.uint8); labels=d.target.astype(np.uint8)
    train,test=train_test_split(np.arange(len(labels)),test_size=.25,random_state=20261030,stratify=labels)
    selected=test[:4].tolist() # fixed before fitting or any test predictions
    dump(root,'split.json',{'train_indices':train.tolist(),'test_indices':test.tolist(),'selected_indices':selected})
    dump(root,'dataset.json',{'pixels':pixels.tolist(),'labels':labels.tolist()})
    (root/'dataset_description.txt').write_text(d.DESCR)
    c=MLPClassifier(**{**plan['training'],'hidden_layer_sizes':(32,)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');c.fit(pixels[train]/16.,labels[train])
    quant=lambda v: np.floor(v*1024+.5).astype(np.int64).tolist()
    model={'format':'digits-mlp-q10-v1','architecture':[64,32,10],'scale':1024,'parameter_absolute_bound':16383,'weight_layout':'input-major [in][out]','weights':[quant(w) for w in c.coefs_],'biases':[quant(b) for b in c.intercepts_]}
    dump(root,'model.json',model)
    dump(root,'float_model.json',{'weights':[w.tolist() for w in c.coefs_],'biases':[b.tolist() for b in c.intercepts_]})
    maximum=max(abs(v) for layer in model['weights'] for row in layer for v in row)
    maximum=max(maximum,max(abs(v) for b in model['biases'] for v in b))
    if maximum>16383: raise RuntimeError(f'Frozen profile mismatch: {maximum} > 16383; no profile changed')
    rows=[{'dataset_index':int(i),'label':int(labels[i]),'input_q10':(pixels[i].astype(int)*64).tolist()} for i in test]
    dump(root,'test_inputs.json',rows)
    outputs=[{'dataset_index':r['dataset_index'],**infer(model,r['input_q10'])} for r in rows]
    dump(root,'integer_test_outputs.json',outputs)
    dump(root,'selected_examples.json',[{'input':r,'output':o} for r,o in zip(rows[:4],outputs[:4])])
    fp=c.predict(pixels[test]/16.).astype(int).tolist();ip=[o['prediction'] for o in outputs]; truth=labels[test].astype(int).tolist()
    ranges={k:[min(v for o in outputs for v in o['trace'][k]),max(v for o in outputs for v in o['trace'][k])] for k in outputs[0]['trace']}
    report={'plan_sha256':sha(plan_path.read_bytes()),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'dataset_hash_encoding':'pixels: uint8 C-order shape [1797,64]; targets: uint8 shape [1797]; indices: little-endian uint32 in returned order','pixels_sha256':sha(pixels.tobytes()),'targets_sha256':sha(labels.tobytes()),'train_indices_sha256':sha(train.astype('<u4').tobytes()),'test_indices_sha256':sha(test.astype('<u4').tobytes()),'train_count':len(train),'test_count':len(test),'selected_indices':selected,'training_iterations':int(c.n_iter_),'training_loss':float(c.loss_),'training_warnings':[str(w.message) for w in caught],'float_correct':sum(a==b for a,b in zip(fp,truth)),'integer_correct':sum(a==b for a,b in zip(ip,truth)),'float_accuracy':sum(a==b for a,b in zip(fp,truth))/len(test),'integer_accuracy':sum(a==b for a,b in zip(ip,truth))/len(test),'prediction_disagreement_indices':[int(test[i]) for i in range(len(test)) if fp[i]!=ip[i]],'float_failure_indices':[int(test[i]) for i in range(len(test)) if fp[i]!=truth[i]],'integer_failure_indices':[int(test[i]) for i in range(len(test)) if ip[i]!=truth[i]],'float_predictions':fp,'observed_heldout_ranges_NOT_domain_certificate':ranges,'maximum_absolute_quantized_parameter':maximum,'parameter_ranges':{'weights':[[int(np.min(w)),int(np.max(w))] for w in model['weights']],'biases':[[min(b),max(b)] for b in model['biases']]}}
    dump(root,'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='float_predictions'},indent=2))
if __name__=='__main__':main()
