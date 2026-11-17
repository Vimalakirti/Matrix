#!/usr/bin/env python3
"""Independent integer-profile tests; not proof-system or model-accuracy evidence."""
from random import Random
P=0xfffffffeffffffe1
QE=32768
LS=(1,1,3,6,11,22,44,89,177,355,710,1420,2839,5678,11357,22713,45426,90852,181704,363409,726817,1453635,2907270,5814540,11629080,23258160,46516320,93032640,186065279,372130559,744261118)
def rd(n,d):
    assert d>0
    return (2*n+d)//(2*d)
def poly(r):
    a=rd(r*r,QE); c=16384+rd(5461*r,QE)
    z=rd(c*a,QE)
    return QE+r+z,z
def exp(x,s,b=8):
    l=LS[s]; k=(-x+l//2)//l
    assert x<=0 and 0<=k<1<<b
    r=x+l*k
    re=r*(1<<(15-s)) if s<=15 else rd(r,1<<(s-15))
    h,_=poly(re)
    return rd((QE>>k if k<=15 else 0)*h,1<<(30-s))
def budget(s,k):
    assert 0<=s<=30 and k>=1
    q=1<<s; dm=q*k; bits=(2*dm-1).bit_length(); t=(1<<bits)-1
    return max(2*q*q+dm,dm*(2*q-1)+t)<P and 2*t<P,dm,bits,max(2*q*q+dm,dm*(2*q-1)+t)
def softmax(xs,allowed,s):
    assert any(allowed)
    q=1<<s; m=max(x for x,a in zip(xs,allowed) if a)
    es=[exp(x-m,s) if a else 0 for x,a in zip(xs,allowed)]
    d=sum(es)
    assert q<=d<=q*sum(allowed)
    return [rd(q*e,d) for e in es],es,d,m
def accepts(e,d,w,s,k):
    ok,dm,bits,_=budget(s,k);q=1<<s
    rho=(2*q*e+d-2*d*w)%P; tau=(2*d-1-rho)%P
    return ok and 0<=e<=q and q<=d<=dm and 0<=w<=q and rho<1<<bits and tau<1<<bits

def main():
    checks=0
    for re in range(-16384,16385):
        h,z=poly(re);assert 0<h<2*QE
        if re<=0: assert 0<=z<=-re and h<=QE
        checks+=1
    print('PASS all32769 internal remainder values: h>0, h<2Qe and k0 h<=Qe; implies all31-scale E bounds0..Qs, E(0)=Qs.')
    rng=Random(20261030)
    for s in range(31):
        q=1<<s;l=LS[s]; assert exp(0,s)==q
        for _ in range(3000):
            k=rng.randrange(256);r=rng.randrange(1-(l+1)//2,l//2+1);x=r-l*k
            if x<=0: assert 0<=exp(x,s)<=q; checks+=1
        for k in [1,3,7,9,512,4096,1<<20]:
            ok,dm,bits,envelope=budget(s,k)
            if s<=20: assert ok
            if ok:
                for _ in range(100):
                    d=rng.randrange(q,dm+1);e=rng.randrange(min(q,d)+1);w=rd(q*e,d)
                    assert accepts(e,d,w,s,k)
                    assert not accepts(e,d,w-1,s,k) and not accepts(e,d,w+1,s,k)
                    checks+=3
    for s in [10,16,20]:
        q=1<<s;l=LS[s]
        # Additive -100 sentinel permits future scores to affect row0.
        old0=softmax([0,-100*q],[1,1],s)[0][0]
        old1=softmax([0,0],[1,1],s)[0][0]
        assert (old0,old1)==(q,q//2)
        a=softmax([0,-l,0],[1,1,0],s)
        b=softmax([0,-l,15*l],[1,1,0],s)
        assert a==b
        # Even selected-only denominator cannot repair a contaminated max.
        bad_e=[exp(-15*l,s),exp(-16*l,s)]
        if sum(bad_e):
            bad=[rd(q*e,sum(bad_e)) for e in bad_e]
            assert bad==[q,0] and a[0][:2]!=bad
        else:
            assert s==10  # contaminating max can instead destroy denominator positivity
        # All nonselected values, including huge field residues, are ignored.
        for v in [-(1<<60),P-1,100*q,15*l,1<<60]:
            assert softmax([0,-l,v],[1,1,0],s)==a
        # Free log normalizer is not fixed by uniformity/nonnegativity.
        assert [exp(0,s)]*2==[q,q]
        assert [exp(-l,s)]*2==[q//2,q//2]
        # Denominator-only contamination from one unmasked exp(0).
        assert rd(q*q,q)==q and rd(q*q,2*q)==q//2
        print(f'PASS SF{s} causal/padding/max-only/denominator-only/free-normalizer counterexamples.')
    q=1<<30;d=8*q;w=q;e=0
    rho=(2*q*e+d-2*d*w)%P;tau=2*d-1-rho
    assert (rho,tau)==(4*q-31,12*q+30)
    assert 0<=rho<2*d and 0<=tau<2*d and 0<=w<=q
    assert not budget(30,9)[0]
    print('PASS concrete SF30 row9 wrap counterfeit: eight e=Q, one e=0; D=8Q; false tail w=Q; rho=',rho,'tau=',tau,'; public budget rejects.')
    for s in [16,20,30]:
        for k in [7,9,512,4096,1<<20]: print('budget',s,k,budget(s,k))
    print('PASS independent assertions',checks,'; no GPU/fullproof/model-accuracy claim.')
if __name__=='__main__':main()
