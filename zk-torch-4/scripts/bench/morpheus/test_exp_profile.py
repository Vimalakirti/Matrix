#!/usr/bin/env python3
"""Independent scalar tests for canonical, non-saturating exp v1.

This is mathematical/scalar evidence, NOT Rust execution, proof verification,
model accuracy, or evidence that normalization/causality has been repaired.
No implementation helpers are imported. All divisions are integer floor.
"""
from decimal import Decimal, localcontext, ROUND_HALF_UP
import math
import re
from pathlib import Path

Q = 0xFFFFFFFEFFFFFFE1
QE = 32768
LS = (1,1,3,6,11,22,44,89,177,355,710,1420,2839,5678,11357,22713,
      45426,90852,181704,363409,726817,1453635,2907270,5814540,11629080,
      23258160,46516320,93032640,186065279,372130559,744261118)

def rd(n, d):
    assert d >= 1
    return (2*n+d)//(2*d)  # nearest, ties toward positive infinity

def bound(s, bits=8):
    return (1<<bits)*LS[s]-LS[s]//2-1

def decompose(x, s, bits=8):
    if not (0<=s<=30 and 1<=bits<=8):
        raise ValueError('unsupported profile')
    if not -bound(s,bits)<=x<=0:
        raise ValueError('outside nonpositive finite domain')
    k=(-x+LS[s]//2)//LS[s]
    return k,x+LS[s]*k

def remainder_internal(r,s):
    return r*(1<<(15-s)) if s<=15 else rd(r,1<<(s-15))

def polynomial(r,s,unit=QE):
    re_=remainder_internal(r,s)
    a=rd(re_*re_,QE)
    b=rd(5461*re_,QE)
    c=b+16384
    h=unit+re_+rd(c*a,QE)
    return h,(re_,a,b,c)

def result(x,s,bits=8):
    k,r=decompose(x,s,bits)
    h,_=polynomial(r,s)
    return rd((QE>>k if k<=15 else 0)*h,1<<(30-s))

def old_result(x,s,bits=4,unit=QE):
    # Existing old helper used f64::round (ties away from zero) on -x/L.
    # The inputs used here are nonpositive; math.floor(v+0.5) matches it.
    l=math.floor(math.log(2)*float(1<<s)+0.5)
    k=math.floor((-float(x))/l+0.5)
    r=x+k*l
    table_index=k if 0<=k<(1<<bits) else 0  # historical silent clamp
    h,_=polynomial(r,s,unit)
    return rd((QE>>table_index if table_index<=15 else 0)*h,1<<(30-s))

def accepts_constraints(x,k,r,s,bits=8):
    """Integer lifting of the four actual range roots plus decomposition."""
    l=LS[s]
    return (0<=k<(1<<bits) and x==r-l*k and
            0<=-x<=bound(s,bits) and 0<=l//2-r<l)

def rejects(fn,*args):
    try: fn(*args)
    except ValueError: return
    raise AssertionError(('unexpected acceptance',args))

def modular_and_padding_regressions():
    # Old graph ScaleDown only bound the remainder. Exhibit all four field
    # equations at x=0 with a counterfeit square quotient (SF10).
    inv=lambda d:pow(d,-1,Q)
    a=(-inv(QE))%Q
    cubic_n=(QE//2)*a%Q
    cubic=cubic_n*inv(QE)%Q
    h=(QE+cubic)%Q
    output_n=QE*h%Q
    output=output_n*inv(1<<20)%Q
    gates=[(0,QE,a,QE//2+1),(0,QE,0,QE//2),
           (cubic_n,QE,cubic,QE//2),(output_n,1<<20,output,1<<19)]
    for n,d,y,rho in gates:
        assert (n-d*y+d//2)%Q==rho and 0<=rho<d
    assert output!=1024 and not 0<=a<=8192
    # Paired new nonnegative roots cannot both be small for this quotient:
    # fake a is far outside[0,8192], and 8192-a is negative in signed lift.
    bits=8192 .bit_length()
    assert not (0<=a<(1<<bits) and 0<=(8192-a)%Q<(1<<bits))
    print('PASS modular quotient regression: old remainder-only constraints accept E(0)=',output,
          'with fake square quotient',a,'; new square interval[0,8192] rejects.')
    # The existing recipe zero-pads all three cubic coefficient literals.
    # Therefore padded output need not be nonnegative even for x<=0.
    c=-math.floor(math.log(5)*1024+0.5)
    k,r=decompose(c,10)
    re_=remainder_internal(r,10)
    padded_h=re_  # padded one_sixth=half=unit=0; b=cubic=0
    padded_y=rd((QE>>k)*padded_h,1<<20)
    assert (c,k,r,re_,padded_y)==(-1648,2,-228,-7296,-57)
    assert not 0<=padded_y<=2048 and -2048<=padded_y<=2048
    print('PASS padding regression: SF10 dim5 zero logits yield padded x=-1648, r=-228, output=-57; signed[-2Qs,2Qs] retains it, unsigned rejects.')
    # Canonical nonpositive profile intentionally cannot cover every old
    # padded-softmax input: old helper fills shifts into padded positions.
    c_negative_logits=2048+c
    assert c_negative_logits==400
    rejects(decompose,c_negative_logits,10)
    print('BEFORE PADDING REPAIR: SF10 dim5 logits=-2048 yield padded exp input+400, rejected by raw nonpositive helper; the following mask regression checks the repaired recipe.')

def masking_regressions():
    def mask(shape):
        padded=[1<<(extent-1).bit_length() for extent in shape]
        n=math.prod(padded)
        result_=[]
        for flat in range(n):
            q=flat; valid=True
            for extent,pad in zip(shape,padded):
                coord=q%pad; q//=pad
                valid &= coord<extent
            result_.append(int(valid))
        return result_
    assert mask([5])==[1,1,1,1,1,0,0,0]
    assert mask([3,3])==[1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0]
    # Include asymmetric axes, distinct logical values, and non-power-of-two
    # higher rank to detect accidental row-major interpretation/broadcasting.
    for s in [0,10,15,30]:
        for shape in [[5],[3,3],[3,5],[2,3,5],[4,4]]:
            m=mask(shape); n=len(m)
            logical=[-(i%7)*LS[s] for i in range(n)]
            expected=[result(v,s) if bit else 0 for v,bit in zip(logical,m)]
            for pad_value in [400,-5000,10000,10**12,-10**12,Q-1]:
                source=[v if bit else pad_value for v,bit in zip(logical,m)]
                # Actual mask relation is elementwise field multiplication;
                # the public mask has sf0, so no rounding/rescaling is added.
                masked=[(v%Q)*bit%Q for v,bit in zip(source,m)]
                signed=[v-Q if v>Q//2 else v for v in masked]
                assert all(v==0 for v,bit in zip(signed,m) if not bit)
                # Exp on logical coordinates; the existing zero padded
                # coefficient literals make every zero padded lane yield0.
                values=[result(v,s) if bit else 0 for v,bit in zip(signed,m)]
                output=[v*bit for v,bit in zip(values,m)]
                assert output==expected
                # The output mask also excludes arbitrary false pad data at
                # its input, independently of helper semantics.
                assert [v*bit for v,bit in zip(
                    [v if bit else 123456 for v,bit in zip(values,m)],m)]==expected
    # The old all-d_pad shift example is now admitted, without affecting any
    # logical exp value. Prior bad pad contributions to reductions change.
    logical_x=-1648; m=mask([5]); corrected=[result(logical_x,10)]*5+[0]*3
    old_zero_logits_padded=[result(logical_x,10)]*5+[-57]*3
    assert corrected[:5]==old_zero_logits_padded[:5]
    assert sum(corrected)-sum(old_zero_logits_padded)==171
    assert sum(corrected)==1025
    print('PASS authenticated-mask scalar semantics: [5], [3,3], asymmetric[3,5], [2,3,5], [4,4]; scales0/10/15/30; arbitrary positive/negative/field-residue pads; distinct logical outputs preserved and every pad0.')
    print('PADDING PROFILE CHANGE: SF10 dim5 padded+400 inputs are zeroed before helper and admitted; all logical values unchanged. Prior zero-logit padded outputs -57 become0; downstream sum854 becomes1025. No exact softmax mass or model-equivalence claim.')

def main():
    print('SCALAR ONLY: no Rust compiler, verifier, model, GPU, or task accuracy execution.')
    modular_and_padding_regressions()
    masking_regressions()
    with localcontext() as c:
        c.prec=100
        decimal_ls=tuple(int((Decimal(2).ln()*(1<<s)).to_integral_value(rounding=ROUND_HALF_UP)) for s in range(31))
    assert LS==decimal_ls
    old_ls=tuple(math.floor(math.log(2)*float(1<<s)+0.5) for s in range(31))
    assert old_ls==LS
    # SF_FLOAT is an f32 power of two, exact for every supported sf0..30;
    # its cast to f64 does not introduce a quantization difference here.
    src=(Path(__file__).resolve().parents[3] / 'src/basicblock/exp.rs').read_text()
    rust_ls=tuple(map(int,re.search(r'pub const EXP_LS: \[i128; 31\] = \[([^]]+)\]',src).group(1).split(',')))
    assert rust_ls==LS
    print('PASS: all 31 exact Ls constants independently match 100-digit Decimal and old f64 Ls.')
    print('PASS: old SF_FLOAT powers of two remain exactly representable at every supported scale.')
    max_bound=0
    for s,l in enumerate(LS):
        for bits in range(1,9):
            B=bound(s,bits); max_bound=max(max_bound,B)
            rb=max(1,B.bit_length()); rr=max(1,(l-1).bit_length())
            assert 2*((1<<rb)-1)<Q and 2*((1<<rr)-1)<Q
            # Signed input lifted via -x and B+x; canonical r via paired
            # rho and L-1-rho. All candidate decomposition intervals are tiny
            # compared with Q, excluding nonzero multiples of Q.
            assert B+(1<<bits)*l+l < (1<<40) < Q
            assert result(0,s,bits)==1<<s
            for x in [0,-1,-B,-max(0,B-1),-l//2]:
                if not -B<=x<=0: continue
                k,r=decompose(x,s,bits)
                assert accepts_constraints(x,k,r,s,bits)
                assert not accepts_constraints(x,k+1,r+l,s,bits)
                assert not accepts_constraints(x,k-1,r-l,s,bits)
            rejects(decompose,1,s,bits)
            rejects(decompose,-B-1,s,bits)
            rejects(decompose,-Q,s,bits)
            # Analytic polynomial budgets from canonical endpoint interval.
            lo=1-(l+1)//2; hi=l//2
            max_re=max(abs(remainder_internal(lo,s)),abs(remainder_internal(hi,s)))
            assert max_re<=16384
            max_a=rd(max_re*max_re,QE)
            max_b=max(abs(rd(5461*max_re,QE)),abs(rd(-5461*max_re,QE)))
            assert max_a<=8192 and max_b<=2731
            max_c=16384+max_b
            assert max_re*max_re<(1<<31) and 5461*max_re<(1<<31) and max_c*max_a<(1<<31)
            h_lo=QE-max_re
            h_hi=QE+max_re+rd(max_c*max_a,QE)
            assert 0<h_lo<=h_hi<2*QE
            assert QE*h_hi<(1<<31)
    for s in [-1,31]: rejects(decompose,0,s)
    for b in [0,9,64]: rejects(decompose,0,10,b)
    print('PASS: all scales x all 8 widths: endpoints, unique adjacent k/r, domain rejection, no-wrap and polynomial bounds.')
    print('largest accepted |x| =',max_bound,'; all candidate decomposition residual magnitudes <2^40<Q.')
    # Exact signed half ties are part of every rescale, including SF>15.
    for d in [2,4,8,32,32768,1<<20]:
        for q in [-17,-2,-1,0,1,16]:
            n=q*d+d//2
            assert rd(n,d)==q+1
            assert rd(n-1,d)==q
            for wrong in [q-1,q,q+2]:
                residual=2*n+d-2*d*wrong
                assert not 0<=residual<2*d
    print('PASS: positive/negative half ties; quotient +/-1 alternatives fail canonical residual interval.')
    # Full default domains at SF10/SF15: includes every old k4-boundary and
    # the existing -100 sentinel, with tail k>=16 fixed to table value zero.
    for s in [10,15]:
        B=bound(s); l=LS[s]; h=l//2
        mn=1<<100; mx=-1; old8_mismatch=0; old4_mismatch=0; oldunit_mismatch=0
        # Most tail rows have zero new output. Still enumerate every x and
        # test decomposition and shifted attack, not just output shortcuts.
        for mag in range(B+1):
            x=-mag; k=(mag+h)//l; r=x+l*k
            assert accepts_constraints(x,k,r,s)
            assert not accepts_constraints(x,k+1,r+l,s)
            assert not accepts_constraints(x,k-1,r-l,s)
            value=result(x,s)
            mn=min(mn,value); mx=max(mx,value)
            old8_mismatch += old_result(x,s,8,QE)!=value
            old4_mismatch += old_result(x,s,4,QE)!=value
            oldunit_mismatch += old_result(x,s,4,1)!=value
        assert mn==0 and mx==1<<s and old8_mismatch==0
        print(f'PASS exhaustive SF{s}: {B+1} integer inputs; output [{mn},{mx}]; corrected-old8 mismatches={old8_mismatch}; corrected-old4 mismatches={old4_mismatch}; original-unit1-old4 mismatches={oldunit_mismatch}.')
        print('columns: x, old4_unit1, old4_unit32768, old8_unit32768, new8_unit32768, (k,r)')
        samples=[0,-1,-l//2,-l,-15*l,-16*l,-100*(1<<s),-B]
        for x in samples:
            print(x,old_result(x,s,4,1),old_result(x,s,4,QE),old_result(x,s,8,QE),result(x,s),decompose(x,s))
    # Scalar only: no statement here about model precision, normalization,
    # causality, padded graph behavior, or proof acceptance.
    print('COMPLETE: unchanged corrected-old8 finite-domain scalar outputs; old4 silent-clamp outputs intentionally differ. No accuracy/equivalence claim outside that domain.')

if __name__=='__main__': main()
