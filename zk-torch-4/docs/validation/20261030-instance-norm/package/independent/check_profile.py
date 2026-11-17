#!/usr/bin/env python3
"""Independent integer/rational audit, not a Rust or cryptographic proof test."""
from math import isqrt
from fractions import Fraction
import random, struct
P=18446744069414584289
K=(1<<28)-1
G=(1<<24)-1
rng=random.Random(0x4d4f525048455553)
checks=0
maxes=dict(residual=0,product=0,quotient=0)
def rd(x,c): return (x+c//2)//c
def rec(m,q):
    if not m:return 0
    r=isqrt(q**3//m)
    # Independent square-root-plus-midpoint oracle, not binary search.
    return r+int(m*(2*r+1)**2<=4*q**3)
def bounds(s,e):
    q=1<<s;R=isqrt(q**3)+1;X=isqrt((K-e)*q)//2
    A=(G*R+q-1)//q;T=(A*X+q-1)//q
    return q,R,X,A,T
for s in range(21):
    for e in [0,1,K//2,K-1,K]+[rng.randrange(K+1) for _ in range(200)]:
        q,R,X,A,T=bounds(s,e)
        for raw,out in [(G*R,A),(A*X,T),((2*X)**2,K-e)]:
            env=raw+q*out+q//2+q-1
            assert env<P and env<(1<<59)
            maxes['residual']=max(maxes['residual'],env)
            maxes['product']=max(maxes['product'],raw)
        assert 2*(G+2*T)<1<<62
        assert ((1<<24)*X)<1<<48
        assert ((1<<24)*(K-e))<1<<52
        maxes['quotient']=max(maxes['quotient'],T)
        checks+=1
        for _ in range(8):
            n=rng.choice([1,2,3,4,7,15,33])
            # Include every endpoint/sign/constant-channel class.
            kind=rng.randrange(5)
            if kind==0: xs=[X]*n
            elif kind==1: xs=[-X]*n
            elif kind==2: xs=[X*(-1 if j%2 else 1) for j in range(n)]
            elif kind==3:xs=[0]*n
            else:xs=[rng.randint(-X,X) for _ in range(n)]
            gamma=rng.choice([-G,0,G,rng.randint(-G,G)])
            beta=rng.choice([-G,0,G,rng.randint(-G,G)])
            mu=rd(sum(xs),n);ds=[x-mu for x in xs]
            zs=[rd(d*d,q) for d in ds];v=rd(sum(zs),n);m=v+e
            r=rec(m,q);a=rd(gamma*r,q);t=rd(a*mu,q);o=beta-t
            hs=[rd(a*x,q) for x in xs];ys=[h+o for h in hs]
            assert abs(mu)<=X and all(abs(d)<=2*X for d in ds)
            assert all(0<=z<=K-e for z in zs) and 0<=v<=K-e and 0<=m<=K
            assert 0<=r<=R and abs(a)<=A and abs(t)<=T and abs(o)<=G+T
            assert all(abs(h)<=T for h in hs) and all(abs(y)<=G+2*T for y in ys)
            if not m:assert r==0 and all(y==beta for y in ys)
            elif r:assert m*(2*r-1)**2<=4*q**3<m*(2*r+1)**2
            else:assert m>4*q**3
            if len(set(xs))==1:assert all(y==beta for y in ys)
            for num,den,quot in [(sum(xs),n,mu),(sum(zs),n,v),(gamma*r,q,a),(a*mu,q,t)]+[(d*d,q,z) for d,z in zip(ds,zs)]+[(a*x,q,h) for x,h in zip(xs,hs)]:
                rho=num-den*quot+den//2
                assert 0<=rho<den
                for bad in [quot-1,quot+1]:assert not 0<=num-den*bad+den//2<den
                checks+=3
            for bad in {r-1,r+1,-r} - {r}:
                valid=(bad==0 and (not m or m>4*q**3)) or (bad>0 and m>0 and m*(2*bad-1)**2<=4*q**3<m*(2*bad+1)**2)
                assert not valid
                checks+=1
# Extended divisor domain: complete residual bounds at c<=2^24.
B=(1<<54)-1
for c in [1,2,3,768,1024,1<<20,(1<<20)+1,(1<<21)-1,1<<21,(1<<24)-1,1<<24]+[rng.randrange(1,(1<<24)+1) for _ in range(10000)]:
    Y=B//c+1
    envelope=B+c*Y+c//2+c-1
    assert envelope<(1<<56)<P
    for a in [-B,-c//2,-1,0,1,c//2,B]:
        y=rd(a,c);assert abs(y)<=Y and 0<=a-c*y+c//2<c
        for bad in [y-1,y+1]:assert not 0<=a-c*bad+c//2<c
        checks+=3
# Symbolic histograms audit huge logical reductions without claiming execution
# or allocating full 128^3/256^3 volumes.
for sf in range(21):
    for e in [0,1,K//2,K]:
        q,R,X,A,T=bounds(sf,e)
        for n in [1<<21,(1<<24)-1,1<<24]:
            for k in [0,1,n//2,n-1,n]:
                total=k*X-(n-k)*X;mu=rd(total,n)
                zz=k*rd((X-mu)**2,q)+(n-k)*rd((-X-mu)**2,q)
                v=rd(zz,n);r=rec(v+e,q)
                assert abs(total)<1<<48 and zz<1<<52
                assert abs(mu)<=X and 0<=v<=K-e and 0<=r<=R
                checks+=1
# Independent Fraction epsilon oracle against exact proposed IEEE decoding.
for s in range(21):
    values=[0.0,-0.0,1e-5,0.5/(1<<s),(K+0.49)/(1<<s),(K+0.5)/(1<<s),float('inf'),float('nan'),-1.0]
    values += [struct.unpack('>d',struct.pack('>Q',rng.randrange(1<<63)))[0] for _ in range(2000)]
    for eps in values:
        bits=struct.unpack('>Q',struct.pack('>d',eps))[0];exponent=(bits>>52)&2047;frac=bits&((1<<52)-1)
        if exponent==2047 or eps<0:continue
        f=Fraction.from_float(eps)*(1<<s);oracle=(2*f.numerator+f.denominator)//(2*f.denominator)
        mantissa=frac if exponent==0 else frac|(1<<52)
        power=(-1074 if exponent==0 else exponent-1023-52)+s
        decoded=(mantissa<<power) if power>=0 else ((mantissa+(1<<(-power-1)))>>(-power))
        assert decoded==oracle
        if power<0 and -power>=55:assert oracle==0
        checks+=1
print(f'PASS {checks} arithmetic assertion groups; all scales 0..20')
print(maxes)
print('Not Rust execution, PIOP replay, PCS verification, model-accuracy or performance evidence.')
