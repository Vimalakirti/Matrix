"""Fail-closed log admission, NOT cryptographic or provenance certification."""
import math
import re


def finite_nonnegative(token):
    # The producer uses decimal fixed-point; reject signs, exponent forms,
    # NaN/Inf, malformed decimals, and decimal values overflowing float.
    return (re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', token) is not None
            and math.isfinite(float(token)) and float(token) >= 0)


def timed_ids(text, pattern):
    rows = re.findall(pattern + r'(\S+)s(?= |$)', text, re.M)
    if any(not finite_nonnegative(row[-1]) for row in rows):
        return None
    return [row[0] for row in rows]


def admitted(text):
    headers = re.findall(r'^MORPHEUS_EXPECT protocol=3 profile=(\S+) requests=(\d+) batches=(\S+)$', text, re.M)
    policies = re.findall(r'^MORPHEUS_POLICY protocol=3 profile=(\S+) auth=(\S+) verify=(\S+) requests=(\d+)$', text, re.M)
    exits = re.findall(r'^MORPHEUS_EXIT status=(\d+)$', text, re.M)
    if len(headers) != 1 or len(policies) != 1 or exits != ['0']:
        return False
    profile, count, batch_text = headers[0]
    if profile != 'morpheus-ternary-v1' or policies[0] != (profile, 'true', 'true', count):
        return False
    if len(count) > 5:
        return False
    n = int(count)
    if not 1 <= n <= 10000:
        return False
    if batch_text == '-':
        batches = []
    elif re.fullmatch(r'[1-9]\d*(,[1-9]\d*)*', batch_text):
        tokens = batch_text.split(',')
        if len(tokens) > n or any(len(x) > 5 for x in tokens):
            return False
        batches = [int(x) for x in tokens]
        if sum(batches) != n:
            return False
    else:
        return False
    ids = [str(i) for i in range(n)]
    proves = timed_ids(text, r'^request (\d+) prove: ')
    verifies = timed_ids(text, r'^request (\d+) verify \(true\): ')
    auth = re.findall(r'^MORPHEUS_AUTH request=(\d+) present=true$', text, re.M)
    if proves != ids or verifies != ids or auth != ids:
        return False
    final_ids = [str(i) for i in range(len(batches))]
    fins = re.findall(r'^finalize batch (\d+) \(J = (\d+),', text, re.M)
    fvers = timed_ids(text, r'^finalize batch (\d+) verify \(true\): ')
    fauth = re.findall(r'^MORPHEUS_AUTH batch=(\d+) present=true$', text, re.M)
    if fins != list(zip(final_ids, map(str, batches))) or fvers != final_ids or fauth != final_ids:
        return False
    fproves = timed_ids(text, r'^finalize batch (\d+) \(J = \d+, [^\n]*\): ')
    if fproves != final_ids:
        return False
    sizes = re.findall(r'^proof per request\s+(\S+) MiB$', text, re.M)
    if len(sizes) != 1 or not finite_nonnegative(sizes[0]):
        return False
    # Reject malformed stage times too: these feed the breakdown table.
    for stages in re.findall(r'^request \d+ prove: .*?\[(.*)\]$', text, re.M):
        for item in stages.split(', '):
            if not finite_nonnegative(item.rsplit(' ', 1)[-1]):
                return False
    # Summary scalars with units are consumed by table generation.
    for token in re.findall(r'(?<!\S)(\S+) (?:s|MiB)(?=[;,)]|$| )', text, re.M):
        if not finite_nonnegative(token):
            return False
    return not re.search(r'panicked|verify \(false\)|present=false|AllocationFailed|out of memory', text, re.I)
