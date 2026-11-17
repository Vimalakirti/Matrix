"""Independent scalar Python Q10 CNN reference; no training convolution code reused."""
import json,pathlib

def dot_round(d): return (d+512)//1024

def conv_scalar(x,w,b):
 out=[]; raw=[]; quot=[]; h=len(x[0]);wid=len(x[0][0]);k=len(w[0][0])
 for o,wo in enumerate(w):
  rows=[];ds=[];qs=[]
  for i in range(h-k+1):
   row=[];dr=[];qr=[]
   for j in range(wid-k+1):
    d=sum(x[c][i+a][j+z]*wo[c][a][z] for c in range(len(x)) for a in range(k) for z in range(k))
    q=dot_round(d);row.append(q+b[o]);dr.append(d);qr.append(q)
   rows.append(row);ds.append(dr);qs.append(qr)
  out.append(rows);raw.append(ds);quot.append(qs)
 return out,raw,quot

def relu(x):return [[[max(0,z) for z in row] for row in channel] for channel in x]
def pool_scalar(x):
 out=[];indices=[]
 for ch in x:
  rows=[];ix=[]
  for i in range(0,len(ch),2):
   row=[];ids=[]
   for j in range(0,len(ch[0]),2):
    v=[ch[i+a][j+b] for a in range(2) for b in range(2)];m=max(v)
    row.append(m);ids.append(v.index(m))
   rows.append(row);ix.append(ids)
  out.append(rows);indices.append(ix)
 return out,indices

def infer(model,input_q10,trace=False):
 if len(input_q10)==64:x=[[[int(input_q10[i*8+j]) for j in range(8)] for i in range(8)]]
 else:x=input_q10
 z1,d1,q1=conv_scalar(x,model['conv1_weight'],model['conv1_bias']);r1=relu(z1);p1,i1=pool_scalar(r1)
 z2,d2,q2=conv_scalar(p1,model['conv2_weight'],model['conv2_bias']);r2=relu(z2);p2,i2=pool_scalar(r2)
 flat=[v for c in p2 for row in c for v in row]
 d3=[sum(a*b for a,b in zip(w,flat)) for w in model['fc_weight']]
 q3=[dot_round(d) for d in d3];logits=[q+b for q,b in zip(q3,model['fc_bias'])]
 pred=max(range(10),key=lambda i:logits[i])
 result={'logits':logits,'prediction':pred}
 if trace:result['trace']={'conv1_dot':d1,'conv1_quotient':q1,'conv1_post_bias':z1,'relu1':r1,'pool1':p1,'pool1_first_indices':i1,'conv2_dot':d2,'conv2_quotient':q2,'conv2_post_bias':z2,'relu2':r2,'pool2':p2,'pool2_first_indices':i2,'flatten':flat,'fc_dot':d3,'fc_quotient':q3,'ranked_argmax_differences':[10*(logits[pred]-logits[j])+j-pred for j in range(10)]}
 return result

if __name__=='__main__':
 import sys
 model=json.load(open(sys.argv[1]));x=json.load(open(sys.argv[2]));print(json.dumps(infer(model,x,True),indent=2))
