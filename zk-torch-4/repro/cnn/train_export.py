"""Single predeclared CNN training run. Does not evaluate held-out examples."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import json, pathlib, hashlib, time, warnings
import numpy as np
import scipy, sklearn
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
ROOT=pathlib.Path(__file__).resolve().parent
SHAPES=((8,1,3,3),(8,),(16,8,2,2),(16,),(10,16),(10,))
NAMES=('conv1_weight','conv1_bias','conv2_weight','conv2_bias','fc_weight','fc_bias')
SIZES=[int(np.prod(s)) for s in SHAPES]
def unpack(t):
 out=[]; off=0
 for sh,n in zip(SHAPES,SIZES): out.append(t[off:off+n].reshape(sh)); off+=n
 assert off==len(t)==778
 return out
def pack(p): return np.concatenate([x.ravel() for x in p])
def conv(x,w,b):
 k=w.shape[-1]; patches=np.lib.stride_tricks.sliding_window_view(x,(k,k),axis=(2,3))
 y=np.einsum('nchwij,ocij->nohw',patches,w,optimize=True)+b[None,:,None,None]
 return y,(x,w,patches)
def conv_back(d,cache):
 x,w,patches=cache; k=w.shape[-1]; h,j=d.shape[-2:]
 dw=np.einsum('nohw,nchwij->ocij',d,patches,optimize=True)
 db=d.sum(axis=(0,2,3)); dx=np.zeros_like(x)
 for a in range(k):
  for b in range(k): dx[:,:,a:a+h,b:b+j]+=np.einsum('nohw,oc->nchw',d,w[:,:,a,b],optimize=True)
 return dx,dw,db
def pool(x):
 n,c,h,w=x.shape; assert h%2==w%2==0
 p=x.reshape(n,c,h//2,2,w//2,2).transpose(0,1,2,4,3,5).reshape(n,c,h//2,w//2,4)
 idx=p.argmax(axis=-1); y=np.take_along_axis(p,idx[...,None],axis=-1)[...,0]
 return y,(x.shape,idx)
def pool_back(d,cache):
 shape,idx=cache; n,c,h,w=shape
 p=np.zeros((n,c,h//2,w//2,4),dtype=d.dtype)
 np.put_along_axis(p,idx[...,None],d[...,None],axis=-1)
 return p.reshape(n,c,h//2,w//2,2,2).transpose(0,1,2,4,3,5).reshape(shape)
def forward(t,x,need_cache=False):
 w1,b1,w2,b2,w3,b3=unpack(t)
 z1,c1=conv(x,w1,b1); r1=np.maximum(z1,0); p1,k1=pool(r1)
 z2,c2=conv(p1,w2,b2); r2=np.maximum(z2,0); p2,k2=pool(r2)
 flat=p2.reshape(len(x),16); logits=flat@w3.T+b3
 if need_cache:return logits,(z1,c1,k1,z2,c2,k2,flat)
 return logits
def loss_grad(t,x,y,alpha=1e-4):
 ws=unpack(t); logits,cache=forward(t,x,True)
 lse=logsumexp(logits,axis=1); loss=float(np.mean(lse-logits[np.arange(len(y)),y]))
 probs=np.exp(logits-lse[:,None]); probs[np.arange(len(y)),y]-=1; probs/=len(y)
 z1,c1,k1,z2,c2,k2,flat=cache
 dw3=probs.T@flat; db3=probs.sum(axis=0)
 d=pool_back((probs@ws[4]).reshape(-1,16,1,1),k2); d*=z2>0
 d,dw2,db2=conv_back(d,c2); d=pool_back(d,k1); d*=z1>0
 _,dw1,db1=conv_back(d,c1)
 gs=[dw1,db1,dw2,db2,dw3,db3]
 for i in (0,2,4): loss+=alpha/2*float(np.sum(ws[i]**2));gs[i]+=alpha*ws[i]
 return loss,pack(gs)
def init(seed=1729):
 rng=np.random.default_rng(seed); out=[]
 for i,sh in enumerate(SHAPES):
  out.append(rng.normal(0,np.sqrt(2/np.prod(sh[1:])),sh) if i%2==0 else np.zeros(sh))
 return pack(out)
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def gradient_check():
 rng=np.random.default_rng(20261031);x=rng.uniform(.1,.9,(3,1,8,8));y=np.array([0,4,9])
 pp=[]
 for i,s in enumerate(SHAPES): pp.append(rng.uniform(.01,.1,s) if i%2==0 else np.full(s,.2))
 pp[4]=rng.normal(0,.1,SHAPES[4]);t=pack(pp)
 # Positive hidden weights/biases avoid ReLU kinks; check distinct pool maxima.
 _,cache=forward(t,x,True)
 assert cache[0].min()>0 and cache[3].min()>0
 gaps=[]
 for z in (cache[0],cache[3]):
  n,c,h,w=z.shape;p=z.reshape(n,c,h//2,2,w//2,2).transpose(0,1,2,4,3,5).reshape(n,c,h//2,w//2,4)
  ss=np.sort(p,axis=-1);gaps.append(float(np.min(ss[...,-1]-ss[...,-2])))
 assert min(gaps)>1e-7
 loss,g=loss_grad(t,x,y); eps=1e-5;checks=[]
 for i in rng.choice(778,100,replace=False):
  a=t.copy();b=t.copy();a[i]+=eps;b[i]-=eps
  fd=(loss_grad(a,x,y)[0]-loss_grad(b,x,y)[0])/(2*eps)
  err=abs(fd-g[i]);checks.append({'coordinate':int(i),'analytic':float(g[i]),'finite_difference':float(fd),'absolute_error':float(err)})
  assert err<2e-7+2e-4*abs(fd),(i,fd,g[i])
 dirs=[]
 for _ in range(10):
  v=rng.normal(size=778);v/=np.linalg.norm(v)
  fd=(loss_grad(t+eps*v,x,y)[0]-loss_grad(t-eps*v,x,y)[0])/(2*eps);an=float(g@v)
  assert abs(fd-an)<2e-7+2e-4*abs(fd),(fd,an)
  dirs.append({'analytic':an,'finite_difference':float(fd),'absolute_error':float(abs(fd-an))})
 # Additional mixed-sign ordinary-init check, away from small perturbation kinks.
 t=init();x=rng.uniform(0,1,(3,1,8,8));loss,g=loss_grad(t,x,y);mixed=[]
 for i in rng.choice(778,40,replace=False):
  a=t.copy();b=t.copy();a[i]+=eps;b[i]-=eps
  fd=(loss_grad(a,x,y)[0]-loss_grad(b,x,y)[0])/(2*eps);err=abs(fd-g[i])
  assert err<2e-6+1e-3*abs(fd),(i,fd,g[i])
  mixed.append({'coordinate':int(i),'absolute_error':float(err)})
 dump('gradient_check.json',{'passed':True,'positive_relu_pool_min_gaps':gaps,'coordinate_checks':checks,'direction_checks':dirs,'mixed_sign_checks':mixed})
 print('GRADIENT_CHECK_PASS',flush=True)
def main():
 plan=ROOT/'training_plan.json';assert plan.exists()
 if (ROOT/'float_model.json').exists(): raise RuntimeError('Refuse overwrite/retraining: frozen model exists')
 gradient_check()
 ds=load_digits(n_class=10);ids=np.arange(len(ds.target));tr,te=train_test_split(ids,test_size=.25,random_state=20261030,stratify=ds.target)
 assert list(te[:4])==[1329,1379,1319,909]
 dump('split.json',{'train_indices':tr.tolist(),'test_indices':te.tolist()})
 x=(ds.images[tr]/16)[:,None,:,:];y=ds.target[tr];t=init();hist=[];start=time.time()
 def fun(t):
  value,grad=loss_grad(t,x,y)
  if len(hist)%25==0: print('evaluation',len(hist),'loss',value,flush=True)
  hist.append(value);return value,grad
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');r=minimize(fun,t,method='L-BFGS-B',jac=True,options={'maxiter':400,'maxfun':50000,'maxls':50,'ftol':1e-10,'gtol':1e-6})
 elapsed=time.time()-start
 f={name:a.tolist() for name,a in zip(NAMES,unpack(r.x))}
 f.update({'architecture':'conv8k3-pool2-conv16k2-pool2-fc10','parameter_order':list(NAMES),'training_plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'test_evaluated':False})
 dump('float_model.json',f)
 q={name:np.floor(a*1024+.5).astype(np.int64).tolist() for name,a in zip(NAMES,unpack(r.x))}
 q.update({'architecture':f['architecture'],'scale':1024,'parameter_order':list(NAMES),'parameter_bound':16383,'training_plan_sha256':f['training_plan_sha256']})
 dump('model.json',q)
 maximum=max(int(np.max(np.abs(np.asarray(q[n])))) for n in NAMES)
 dump('training_diagnostics.json',{'success':bool(r.success),'status':int(r.status),'message':str(r.message),'iterations':int(r.nit),'function_evaluations':int(r.nfev),'gradient_evaluations':int(r.njev),'final_loss':float(r.fun),'gradient_max_abs':float(np.max(np.abs(r.jac))),'elapsed_seconds':elapsed,'warnings':[str(w.message) for w in ww],'loss_evaluations':hist,'test_evaluated':False,'quantized_max_abs':maximum,'parameter_bound_admitted':maximum<=16383,'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}})
 print('TRAINING_FROZEN',r.success,r.nit,maximum,elapsed,flush=True)
 if maximum>16383: raise RuntimeError('Frozen model exceeds public bound; STOP before test evaluation')
 dump('BOUND_ADMISSION.json',{'admitted':True,'max_abs':maximum,'bound':16383,'model_sha256':hashlib.sha256((ROOT/'model.json').read_bytes()).hexdigest(),'float_model_sha256':hashlib.sha256((ROOT/'float_model.json').read_bytes()).hexdigest(),'test_evaluated_before_admission':False})
if __name__=='__main__': main()
