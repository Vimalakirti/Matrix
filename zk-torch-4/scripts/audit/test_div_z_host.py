"""Compile extracted CUDA integer helpers as HOST C++; not a CUDA/GPU test.
Only CUDA qualifiers and two integer intrinsics are replaced. Python big integers
supply an independent quotient/remainder oracle. Requires g++, no third-party deps.
"""
import pathlib, random, subprocess, tempfile
root = pathlib.Path(__file__).resolve().parents[3]
header = (root / 'cuda_almost_goldilocks/morpheus_gauss.cuh').read_text()
body = header[header.index('template <int N> struct U'):header.index('__device__ U<3> ans_lo')]
preamble = r'''
#include <cmath>
#include <iostream>
#define __device__
#define __forceinline__ inline
using std::isfinite; using std::ldexp; using std::floor;
using u64 = unsigned long long;
u64 __umul64hi(u64 a,u64 b) { return (unsigned __int128(a)*b)>>64; }
int __clzll(u64 a) { return __builtin_clzll(a); }
'''.replace('unsigned __int128(a)', 'static_cast<unsigned __int128>(a)')
main = r'''
int main() {
 U<2> d; U<3> den;
 while (std::cin >> d.w[0] >> d.w[1] >> den.w[0] >> den.w[1] >> den.w[2]) {
  bool ie=false, exact_ie=false;
  auto q=div_z(d,den,ie); auto e=div_z_exact(d,den,exact_ie);
  std::cout<<q.w[0]<<" "<<q.w[1]<<" "<<ie<<" "<<e.w[0]<<" "<<e.w[1]<<" "<<exact_ie<<"\n";
 }
}
'''
M = 1 << 128
cases = [(d,den) for d in (0,1,2,(1<<64)-1,1<<64,M//2,M-2,M-1)
         for den in (M+1,M+2,2*M-1,2*M,3*M-1,4*M-1)]
rng = random.Random(903)
cases += [(rng.randrange(M),rng.randrange(M+1,4*M)) for _ in range(10000)]
def limbs(n,count): return [(n>>(64*i))&((1<<64)-1) for i in range(count)]
with tempfile.TemporaryDirectory() as td:
 p=pathlib.Path(td); (p/'test.cpp').write_text(preamble+body+main)
 subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined','-fno-sanitize-recover=all',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 result=subprocess.run([str(p/'test')],input=''.join(' '.join(map(str,limbs(d,2)+limbs(den,3)))+'\n' for d,den in cases),text=True,capture_output=True,check=True)
 lines=result.stdout.splitlines(); assert len(lines)==len(cases)
 for (d,den),line in zip(cases,lines):
  lo,hi,ie,elo,ehi,eie=map(int,line.split()); q,r=divmod(d*M,den)
  assert lo+(hi<<64)==q and elo+(ehi<<64)==q and ie==eie==bool(r),(d,den,line,q,r)
 print(f'PASS: {len(cases)} host-extracted div_z + div_z_exact cases, Python integer oracle; UBSan enabled. NOT a CUDA/GPU test.')
