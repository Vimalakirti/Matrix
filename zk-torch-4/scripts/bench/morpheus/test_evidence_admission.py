"""Synthetic strings only: does not import/execute Morpheus or need GPUs."""
import unittest
from evidence_admission import admitted


def fixture(n=3, batches=()):
    b = ','.join(map(str, batches)) or '-'
    s = f'MORPHEUS_EXPECT protocol=3 profile=morpheus-ternary-v1 requests={n} batches={b}\n'
    s += f'MORPHEUS_POLICY protocol=3 profile=morpheus-ternary-v1 auth=true verify=true requests={n}\n'
    for i in range(n):
        s += f'request {i} prove: 1.00s\nMORPHEUS_AUTH request={i} present=true\nrequest {i} verify (true): 2.00s\n'
    for i, j in enumerate(batches):
        s += f'finalize batch {i} (J = {j}, 7 deferred rows, 90 proof bytes): 1.00s\nMORPHEUS_AUTH batch={i} present=true\nfinalize batch {i} verify (true): 2.00s\n'
    return s + 'proof per request 1300.0 MiB\nMORPHEUS_EXIT status=0\n'


class AdmissionTests(unittest.TestCase):
    def test_legacy_or_wrong_protocol_is_not_v3_evidence(self):
        self.assertFalse(admitted(fixture().replace('protocol=3 ', '')))
        for version in ('1', '2'):
            self.assertFalse(admitted(fixture().replace('protocol=3', 'protocol=' + version)))

    def test_oversized_integer_tokens(self):
        huge = '1' * 5000
        self.assertFalse(admitted(fixture().replace('requests=3', 'requests=' + huge)))
        self.assertFalse(admitted(fixture(10, (2, 8)).replace('batches=2,8', 'batches=' + huge)))

    def test_table_parser_rejects_without_conversion_crash(self):
        from paper_tables import parse
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'test.log'
            for bad in [fixture().replace('prove: 1.00s', 'prove: ..s'),
                        fixture().replace('requests=3', 'requests=' + '1' * 5000),
                        fixture() + 'proof parts (bytes): invalid\n',
                        fixture() + 'registration (one-time): ' + '1' * 5000 + ' sources committed in 1.0s\n']:
                p.write_text(bad)
                result = parse(p)
                self.assertFalse(result['ok'])
                self.assertIsNone(result['witness'])
                self.assertIsNone(result['reg'])
            p.write_text(fixture())
            self.assertTrue(parse(p)['ok'])

    def test_complete_mono(self): self.assertTrue(admitted(fixture()))
    def test_complete_deferred(self): self.assertTrue(admitted(fixture(10, (2, 8))))
    def test_partial(self): self.assertFalse(admitted(fixture().split('request 2 prove')[0]))
    def test_no_receipt(self): self.assertFalse(admitted('request 0 prove: 1.00s\nrequest 0 verify (true): 1.00s'))
    def test_nonzero_exit(self): self.assertFalse(admitted(fixture().replace('status=0', 'status=124')))
    def test_no_auth(self): self.assertFalse(admitted(fixture().replace('auth=true', 'auth=false')))
    def test_absent_wire_auth(self): self.assertFalse(admitted(fixture().replace('request=1 present=true', 'request=1 present=false')))
    def test_no_verify(self): self.assertFalse(admitted(fixture().replace('verify=true', 'verify=false')))
    def test_toy(self): self.assertFalse(admitted(fixture().replace('morpheus-ternary-v1', 'morpheus-toy')))
    def test_missing_verification(self): self.assertFalse(admitted(fixture().replace('request 1 verify (true): 2.00s\n', '')))
    def test_duplicate_id(self): self.assertFalse(admitted(fixture().replace('request 2 verify', 'request 1 verify')))
    def test_missing_final(self): self.assertFalse(admitted(fixture(10, (2, 8)).replace('finalize batch 1 verify (true): 2.00s\n', '')))
    def test_wrong_batch_size(self): self.assertFalse(admitted(fixture(10, (2, 8)).replace('(J = 8,', '(J = 7,')))
    def test_duplicate_run(self): self.assertFalse(admitted(fixture()+fixture()))
    def test_crash_after_verify(self): self.assertFalse(admitted(fixture()+'panicked'))
    def test_malformed_times(self):
        for token in ('..', 'NaN', 'inf', '-1.00', '1.2.3', '1e999', '9' * 400):
            with self.subTest(token=token):
                self.assertFalse(admitted(fixture().replace('prove: 1.00s', f'prove: {token}s')))
    def test_trailing_time_junk(self): self.assertFalse(admitted(fixture().replace('prove: 1.00s', 'prove: 1.00sjunk')))
    def test_malformed_final_time(self): self.assertFalse(admitted(fixture(10, (2, 8)).replace('proof bytes): 1.00s', 'proof bytes): ..s')))
    def test_malformed_component_time(self): self.assertFalse(admitted(fixture().replace('prove: 1.00s', 'prove: 1.00s [auth ..]')))
    def test_malformed_size(self): self.assertFalse(admitted(fixture().replace('1300.0 MiB', '.. MiB')))
    def test_invalid_count_checked_before_resources(self):
        from pathlib import Path
        source = Path(__file__).with_name('run_extra.sh').read_text()
        validation = source.index('[[ "${N_DEFER:-2}"')
        self.assertLess(validation, source.index('mkdir -p'))
        self.assertLess(validation, source.index('local watch=$!'))



if __name__ == '__main__': unittest.main()
