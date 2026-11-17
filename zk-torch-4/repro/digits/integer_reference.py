"""Pure-Python integer inference. No NumPy, sklearn, field code or floats."""
import json
from pathlib import Path
SCALE = 1024
BOUND = 16383

def rounded(dot):
    return (dot + SCALE // 2) // SCALE  # Python floor division, also for negatives

def infer(model, x):
    assert len(x) == 64 and all(type(v) is int and 0 <= v <= 1024 for v in x)
    values = list(x)
    trace = {}
    for layer, (n, m) in enumerate(((64, 32), (32, 10)), 1):
        w, b = model['weights'][layer-1], model['biases'][layer-1]
        assert len(w) == n and all(len(row) == m for row in w) and len(b) == m
        assert all(type(v) is int and abs(v) <= BOUND for row in w for v in row)
        assert all(type(v) is int and abs(v) <= BOUND for v in b)
        dot = [sum(values[i] * w[i][j] for i in range(n)) for j in range(m)]
        scaled = [rounded(v) for v in dot]
        biased = [scaled[j] + b[j] for j in range(m)]
        trace[f'dot{layer}'] = dot
        trace[f'rounded{layer}'] = scaled
        trace[f'biased{layer}'] = biased
        values = [max(0, v) for v in biased] if layer == 1 else biased
        if layer == 1: trace['relu1'] = values
    # max visits ascending indices; equal values retain the first.
    prediction = max(range(10), key=lambda j: values[j])
    trace['argmax_differences'] = [values[prediction] - v for v in values]
    return {'prediction': prediction, 'logits': values, 'trace': trace}

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    model = json.loads((root / 'model.json').read_text())
    rows = json.loads((root / 'test_inputs.json').read_text())
    expected = json.loads((root / 'integer_test_outputs.json').read_text())
    actual = [{'dataset_index': r['dataset_index'], **infer(model, r['input_q10'])} for r in rows]
    assert actual == expected
    assert [rounded(x) for x in [-1536,-512,512,1536]] == [-1,0,1,2]
    print(f'PASS: {len(actual)} held-out integer traces exactly reproduced; signed tie tests passed.')
