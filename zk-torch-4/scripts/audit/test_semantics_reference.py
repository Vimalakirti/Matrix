"""Independent scalar audit regressions; does not import or execute repository code."""
import unittest

def old_scaleup_accept(x, y, s):
    return 0 <= s*x-y+s//2 < s

def exact_scaleup_accept(x, y, s):
    return y == s*x

def shift_ok(tokens, predictions, prompt_len):
    return 1 <= prompt_len <= len(tokens) and len(predictions) == len(tokens) and all(tokens[i+1] == predictions[i] for i in range(prompt_len-1, len(tokens)-1))

def lowest_argmax_ok(logits, selected):
    return 0 <= selected < len(logits) and all(logits[selected] >= v + int(i < selected) for i,v in enumerate(logits))

class Semantics(unittest.TestCase):
    def test_scaleup_counterexample(self):
        self.assertTrue(old_scaleup_accept(0, 1, 2))
        self.assertFalse(exact_scaleup_accept(0, 1, 2))
    def test_scaleup_exhaustive_small(self):
        for s in [1,2,4,8]:
            for x in range(-10,11):
                for y in range(-100,101):
                    self.assertEqual(exact_scaleup_accept(x,y,s), y==s*x)
    def test_shift_not_independent_argmax(self):
        self.assertFalse(shift_ok([2,3,4,5], [0,0,0,0], 2))
        self.assertTrue(shift_ok([2,3,0,0], [1,0,0,7], 2))
    def test_tie_break(self):
        self.assertTrue(lowest_argmax_ok([5,5],0))
        self.assertFalse(lowest_argmax_ok([5,5],1))
        self.assertTrue(lowest_argmax_ok([-2,-1,-1],1))
    def test_symmetric_mean_tie_is_not_deterministic(self):
        x,n=1,2
        admitted=[y for y in range(-2,3) if abs(x-n*y)<=n//2]
        self.assertEqual(admitted,[0,1])
        self.assertEqual((x+n//2)//n,1)

class CertifiedArgmaxBudget(unittest.TestCase):
    @staticmethod
    def budget(b, vocab, q):
        if not (vocab>0 and 0<b<62): raise ValueError("profile")
        bound=(1<<b)-1
        ranged=(1<<(b+1))-1
        if 2*ranged>=q: raise ValueError("range wrap")
        d=2*bound*vocab+vocab-1
        k=d.bit_length()
        if k>=64 or d+(1<<k)-1>=q: raise ValueError("comparison wrap")
        return bound,b+1,k
    def test_exhaustive_small_field_range_and_comparison(self):
        q=29
        bound,bits,k=self.budget(1,3,q)
        for residue in range(q):
            accepted=(residue+bound)%q < 1<<bits and (bound-residue)%q < 1<<bits
            signed=residue if residue<=q//2 else residue-q
            self.assertEqual(accepted, -bound<=signed<=bound)
        import itertools
        for logits in itertools.product(range(-bound,bound+1),repeat=3):
            expected=max(range(3),key=lambda i:(logits[i],-i))
            for selected in range(3):
                rank=[3*v-i for i,v in enumerate(logits)]
                admitted=all((rank[selected]-v)%q < 1<<k for v in rank)
                self.assertEqual(admitted,selected==expected)
    def test_large_bound_fails_closed(self):
        q=0xFFFFFFFEFFFFFFE1
        with self.assertRaises(ValueError): self.budget(60,128256,q)
        b,_,k=self.budget(16,128256,q)
        self.assertEqual(k,34)
        self.assertLess(2*b*128256+128255+(1<<k)-1,q)

if __name__ == '__main__': unittest.main()
