"""S48 gate: the SYNTHETIC THEOREM GENERATOR (src/python/synth) and the dataset it writes.

The generator is untrusted code that makes a CLAIM -- "these are theorems, each with a proof the kernel accepts, varied in depth, tidy, not decoration, not leaking the problems" -- and this gate protects
what makes the claim checkable: a record carries everything needed to check it, a checker that reads only the record catches every way a record can be wrong, and the data are a function of the
configuration and the index (so that a published dataset can be regenerated, and a shard remade alone).

  A. UNITS.         The polynomial normal form behind "tidy" (synth_poly) agrees with evaluation; `Vocab.tidy` refuses what nobody writes (x + 0, x - x, (a + b) - a, ...) and accepts what they do;
                    `random_tree` draws only tidy terms and the code of a tree parses back to it; `pretty` is the readable text of a statement (pins); `shape_of` forgets numerals and names and nothing
                    else; `split_of` is a function of the record, with the intended proportions.
  B. DETERMINISM.   The record of an index is the same whether it is made first, last or alone, by a new generator on the same world or by another process with another hash seed.  Shards of the
                    command line equal the records made here.  A pinned digest for every record of a fixed range (`tests/golden/s48_golden.json`).
  C. THE RECORDS.   Every record has the documented fields and types; the promises of the generator hold on each: depth at least the plan's, hypotheses within the limit, no decoration, no
                    hypothesis that is a truth, satisfiable, statements tidy (no x + 0 ...), the closable quota, the id and the recipe are functions of the data.
  D. FROM DATA.     `synth_verify.check_record` -- fresh engine, statement from the CODE, proof from the STEPS, kernel-checked, labels read back -- accepts every record, and the independent checker accepts
                    every proof (and refuses it against another statement).
  E. TAMPERING.     A record that is wrong in any one field is refused (with the reason): steps dropped, added, reversed, swapped between records, a lemma replaced; a code of another statement; digest,
                    readable text, counts and labels changed.  `verify_files` reports the id of the bad record and says the dataset is not ok.
  F. MERGE.         Exclusion of the library's, the bank's and the sub-theorems' statements; duplicates; the shape cap; splits (no recipe of `test_comp` in the other splits, a split is a function of the
                    record); the result does not depend on how the index space was cut into shards, nor on their order; the train target (exactly that many train records, a prefix of the dataset
                    without a target; a dry run counts the same and writes nothing); shards of another configuration are refused, shards of another range are made again.
  G. THE COMMAND LINE.  gen (two processes) -> merge -> verify -> card, end to end; the card says what the manifest says; `grow` tops up a dataset that is short of its train target (and the result is the
                    one a single large `gen` and `merge --train-target` give), says SHORT and exits 3 when it cannot, and `merge --train-target` fails the same way unless `--allow-short`.

Environment: S48_TRACE=1 prints timings, S48_ONLY=A,B selects sections, S48_UPDATE_GOLDEN=1 rewrites the pinned digests (when a change of the generator is intended).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import collections, copy, gzip, hashlib, json, os, random, re, shutil, subprocess, sys, tempfile, time, traceback
sys.setrecursionlimit(200000)
import imo_lib
import gym_env as G, gym_acts as A
import recheck
import synth_gen as SG, synth_merge as SM, synth_verify as SV, synth_check as SC, synth_poly as SPY, synth_terms as STM, synth_cli as CLI
from synth_tree import Reject

FAIL = []; CHECKS = [0]
TRACE = bool(os.environ.get('S48_TRACE'))
ONLY = set(os.environ['S48_ONLY'].split(',')) if os.environ.get('S48_ONLY') else None
GOLDEN = os.path.join(itp_paths.GOLDEN, 's48_golden.json')
SEED = 48
RANGE = (0, 240)                                          # the indices the gate generates in this process
SUBPROBLEMS = os.path.join(itp_paths.DATA, 'dataset_v2', 'subproblems.json')


def trace(msg):
    if TRACE: print('   [trace %s]' % msg, flush=True)


def expect(cond, what):
    CHECKS[0] += 1
    if not cond: FAIL.append(what)
    return bool(cond)


def raises(what, fn, exc=Exception):
    try: fn()
    except exc: return expect(True, what)
    except BaseException as e: return expect(False, '%s (it raised %s: %s)' % (what, type(e).__name__, str(e)[:100]))
    return expect(False, what + ' (it raised nothing)')


class Fixture:
    """the library's world, a generator on it, and the records of RANGE (made once, in order)"""
    def __init__(self):
        t0 = time.time()
        self.w = imo_lib.build_imo(); self.limit = self.w.gid['imo1959_p1']
        self.cfg = SG.Config(seed=SEED)
        self.gen = SG.Generator(self.cfg, world=self.w)
        self.eng = G.Engine(self.w, kinds=set(G.Kind), strict=True)
        trace('world %.1f s' % (time.time() - t0))
        t0 = time.time()
        self.recs = {}; self.rejects = {}
        for i in range(*RANGE):
            try:
                rec = self.gen.record(i); self.gen.self_check(rec); self.recs[i] = rec
            except Reject as e: self.rejects[i] = e.reason.split(':')[0]
        trace('%d records from %d indices, %.1f s' % (len(self.recs), RANGE[1] - RANGE[0], time.time() - t0))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ A. units
def eval_poly(p, sigma):
    return sum(c * _prod(sigma[v] ** e for v, e in m) for m, c in p.items())


def _prod(xs):
    r = 1
    for x in xs: r *= x
    return r


def raw_tree(rng, depth, nvars):
    """any tree of the vocabulary, tidy or not"""
    if depth == 0 or rng.random() < 0.3:
        return ('v', rng.randrange(nvars)) if rng.random() < 0.7 else ('n', rng.randint(0, 5))
    op = rng.choice(('add', 'mul', 'sub', 'neg'))
    return (op, raw_tree(rng, depth - 1, nvars)) if op == 'neg' else (op, raw_tree(rng, depth - 1, nvars), raw_tree(rng, depth - 1, nvars))


def sec_A(c):
    fx = c['fx']; w = fx.w; rng = random.Random(480)
    vocab = STM.Vocab(w, **STM.VOCABS['Z'])
    # A1  the polynomial of a tree is the tree's value, as a function of the variables
    bad = 0; n = 0
    for _ in range(400):
        t = raw_tree(rng, 4, 3); p = SPY.tree_poly(t)
        if p is None: bad += 1; continue
        for _ in range(3):
            sigma = [rng.randint(-6, 6) for _ in range(3)]; n += 1
            if eval_poly(p, sigma) != vocab.value(t, sigma): bad += 1
    expect(bad == 0 and n >= 1000, 'A1 tree_poly agrees with evaluation on %d (tree, assignment) pairs (%d disagree)' % (n, bad))
    v0, v1, v2 = ('v', 0), ('v', 1), ('v', 2); n0, n1, n2, n3 = ('n', 0), ('n', 1), ('n', 2), ('n', 3)
    expect(SPY.tree_poly(('mul', ('add', v0, v1), ('sub', v0, v1))) == SPY.padd(SPY.pmul(SPY.var(0), SPY.var(0)), SPY.pneg(SPY.pmul(SPY.var(1), SPY.var(1)))), 'A1 (a+b)(a-b) = a^2 - b^2')
    expect(SPY.tree_poly(('sub', ('add', v0, v1), v0)) == SPY.var(1), 'A1 (a+b)-a = b')
    # A2  tidy: what nobody writes is refused, what people write is accepted
    refused = {'x+0': ('add', v0, n0), '0+x': ('add', n0, v0), 'x*1': ('mul', v0, n1), '1*x': ('mul', n1, v0), 'x*0': ('mul', v0, n0), 'x-0': ('sub', v0, n0), '0-x': ('sub', n0, v0), 'x-x': ('sub', v0, v0),
               '--x': ('neg', ('neg', v0)), '-2': ('neg', n2), '2+3': ('add', n2, n3), '(a+b)-a': ('sub', ('add', v0, v1), v0), '(a-b)+b': ('add', ('sub', v0, v1), v1), 'ab-ba': ('sub', ('mul', v0, v1), ('mul', v1, v0)),
               '(a*b)*0': ('mul', ('mul', v0, v1), n0), 'a-(a+b)': ('sub', v0, ('add', v0, v1))}
    for what, t in refused.items(): expect(not vocab.tidy(t), 'A2 tidy refuses %s' % what)
    accepted = {'x': v0, '7': ('n', 4), 'x+y': ('add', v0, v1), 'xy+2': ('add', ('mul', v0, v1), n2), 'x*x+y': ('add', ('mul', v0, v0), v1), '-x-y': ('sub', ('neg', v0), v1), '(x+y)(x-y)': ('mul', ('add', v0, v1), ('sub', v0, v1)),
                '(x+1)*y': ('mul', ('add', v0, n1), v1), '3*x+y': ('add', ('mul', n3, v0), v1), '2-x': ('sub', n2, v0), 'x*y*z': ('mul', ('mul', v0, v1), v2)}
    for what, t in accepted.items(): expect(vocab.tidy(t), 'A2 tidy accepts %s' % what)
    # A3  random_tree: tidy, and its code parses back to it
    drawn = 0; untidy = 0; bad_code = 0
    for i in range(600):
        t = vocab.random_tree(rng, rng.randint(1, 4), ops=rng.randint(0, 4), p_op=rng.choice((0.4, 0.55, 0.8)), no_unit=bool(i % 2))
        if t is None: continue
        drawn += 1
        if not vocab.tidy(t): untidy += 1
        if vocab.parse(vocab.code(t)) != t: bad_code += 1
    expect(drawn >= 450 and untidy == 0 and bad_code == 0, 'A3 random_tree: %d drawn, %d untidy, %d whose code does not parse back to the tree' % (drawn, untidy, bad_code))
    expect(vocab.parse([('c', vocab.gid_of['add'], 2), ('v', 0, 0)]) is None and vocab.parse([('v', 0, 0), ('v', 1, 0)]) is None, 'A3 an incomplete or overlong code is not a term')
    # A4  pretty: the readable text of a statement (pins)
    pins = [('Z.of_N (Npos (XI (XO XH)))', '5'), ('Z.of_N N0', '0'), ('Z.of_N (Npos XH)', '1'), ('Z.of_N (Npos (XO (XO (XO XH))))', '8'), ('Z.of_N (Npos (XI (XI (XI (XO XH)))))', '23'),
            ('Z.neg x0', '-x0'), ('Z.dvd x0 (Z.neg (x1 + x2))', 'Z.dvd x0 (-(x1 + x2))'), ('forall (x0 : Z), forall (x1 : Z), x0 = x1', 'forall (x0 x1 : Z), x0 = x1'),
            ('forall (x0 : Z), forall (x1 : Z), forall (x2 : Z), Z.dvd x0 x1', 'forall (x0 x1 x2 : Z), Z.dvd x0 x1'), ('Z.zero < Z.one', '0 < 1'), ('Z.of_N N0 <= x0', '0 <= x0'),
            ('x0 + Z.of_N (Npos (XO XH))', 'x0 + 2')]
    for src, want in pins: expect(SC.pretty(src) == want, 'A4 pretty(%r) = %r (got %r)' % (src, want, SC.pretty(src)))
    # A5  shape_of forgets numerals and the names of variables, nothing else
    expect(SM.shape_of('forall (x0 x1 : Z), x0 + 3 <= x1 * 5') == SM.shape_of('forall (x9 x2 : Z), x9 + 4 <= x2 * 8'), 'A5 numerals and names are forgotten')
    expect(SM.shape_of('forall (x0 x1 : Z), x0 + 3 <= x1 * 5') == SM.shape_of('forall (x4 x7 : Z), x7 + 12 <= x4 * 2'), 'A5 a consistent renaming of the variables is the same shape')
    expect(SM.shape_of('forall (x0 x1 : Z), x0 + 3 <= x1 * 5') != SM.shape_of('forall (x0 x1 : Z), x0 + 3 <= x0 * 5'), 'A5 the pattern in which variables recur is part of the shape')
    expect(SM.shape_of('forall (x0 x1 : Z), x0 + 3 <= x1 * 5') != SM.shape_of('forall (x0 x1 : Z), x0 + 3 < x1 * 5'), 'A5 the relation is part of the shape')
    expect(SM.shape_of('forall (x0 x1 : Z), x0 + 3 <= x1 * 5') != SM.shape_of('forall (x0 x1 : Z), x0 * 3 <= x1 * 5'), 'A5 the operator is part of the shape')
    # A6  split_of: a function of the record; proportions as intended; the held-out recipes are a function of the recipe
    rng2 = random.Random(481); counts = collections.Counter(); rec_split = {}
    for i in range(20000):
        rec = {'recipe': hashlib.sha256(('r%d' % rng2.randrange(4000)).encode()).hexdigest()[:16], 'digest': hashlib.sha256(('d%d' % i).encode()).hexdigest()[:16]}
        s = SM.split_of(rec); counts[s] += 1; rec_split.setdefault(rec['recipe'], set())
        if s == 'test_comp': rec_split[rec['recipe']].add('comp')
    n = sum(counts.values())
    expect(abs(counts['test_comp'] / n - 0.06) < 0.025 and abs(counts['test_iid'] / n - 0.03 * 0.94) < 0.008 and abs(counts['dev'] / n - 0.03 * 0.94) < 0.008, 'A6 split proportions %s' % dict(counts))
    r1 = {'recipe': 'a' * 16, 'digest': 'b' * 16}
    expect(SM.split_of(r1) == SM.split_of(dict(r1, extra=1)) == SM.split_of(copy.deepcopy(r1)), 'A6 a split is a function of the recipe and the digest')
    expect(SM.split_of(r1, salt='other') in SM.SPLITS, 'A6 another salt, another split (a valid one)')
    # every record of one recipe is test_comp or none is
    by_recipe = collections.defaultdict(set)
    rng3 = random.Random(482)
    for i in range(3000):
        rc = hashlib.sha256(('rc%d' % rng3.randrange(300)).encode()).hexdigest()[:16]
        by_recipe[rc].add(SM.split_of({'recipe': rc, 'digest': hashlib.sha256(('dd%d' % i).encode()).hexdigest()[:16]}) == 'test_comp')
    expect(all(len(v) == 1 for v in by_recipe.values()), 'A6 test_comp holds whole recipes: a recipe is never split between test_comp and the rest')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ B. determinism
def _jsonl(recs): return [json.dumps(r, separators=(',', ':')) for r in recs]


def sec_B(c):
    fx = c['fx']; w = fx.w; gen = fx.gen
    ids = sorted(fx.recs)
    expect(len(ids) >= 25, 'B1 the range %s yields at least 25 records (got %d)' % (RANGE, len(ids)))
    # B1  the same record in another order, with the same generator
    sample = ids[::max(1, len(ids) // 10)][:10]
    again = {}
    for i in reversed(sample): again[i] = gen.record(i)
    expect(all(json.dumps(again[i]) == json.dumps(fx.recs[i]) for i in sample), 'B1 the record of an index does not depend on what was made before it (reversed order, %d records)' % len(sample))
    # B2  and with a NEW generator (a new grower, engine and caches) on the same kernel
    fresh = SG.Generator(fx.cfg, world=w)
    for i in sample[:4]: expect(json.dumps(fresh.record(i)) == json.dumps(fx.recs[i]), 'B2 index %d: a new generator makes the same record' % i)
    # B3  a record is a function of (configuration, index): another seed, another record
    other = SG.Generator(SG.Config(seed=SEED + 1), world=w)
    diff = 0
    for i in sample[:6]:
        try: r = other.record(i)
        except Reject: diff += 1; continue
        diff += r['digest'] != fx.recs[i]['digest']
    expect(diff >= 5, 'B3 another seed gives other theorems (%d of 6 differ)' % diff)
    expect(fx.cfg.digest() != SG.Config(seed=SEED + 1).digest() and fx.cfg.digest() != SG.Config(seed=SEED, closer_keep=0.5).digest() and fx.cfg.digest() == SG.Config(seed=SEED).digest(), 'B3 the configuration digest sees every field')
    expect(SG.Config.from_json(json.loads(json.dumps(fx.cfg.to_json()))).digest() == fx.cfg.digest(), 'B3 a configuration survives JSON')
    # B4  another process, other hash seeds: the same records (the generator must not depend on the order of a set of strings)
    code = ("import sys; sys.path.insert(0, %r)\nimport itp_paths, json\nimport synth_gen as SG\nfrom synth_tree import Reject\ngen = SG.Generator(SG.Config(seed=%d))\nout = {}\n"
            "for i in %r:\n    try: out[i] = gen.record(i)['digest']\n    except Reject: pass\nprint(json.dumps(out))\n") % (itp_paths.SRC_PY, SEED, list(range(0, 60)))
    procs = []
    for hs in ('0', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        procs.append(subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=tempfile.gettempdir()))
    want = {str(i): fx.recs[i]['digest'] for i in fx.recs if i < 60}
    for hs, p in zip(('0', '12345'), procs):
        out, err = p.communicate(timeout=600)
        try: got = json.loads(out.strip().split('\n')[-1])
        except Exception: got = None
        expect(got == want and len(want) >= 5, 'B4 another process (PYTHONHASHSEED=%s) makes the same %d theorems of the indices 0..59%s' % (hs, len(want), '' if got is not None else ' [' + err[-200:] + ']'))
    # B4b a derivation that a later derivation invalidates is undone (finding F74): these four indices of seed 1 ended in a failed replay before, and are now rejected by the quality rules, the undo having run once each
    g1 = SG.Generator(SG.Config(seed=1), world=w); st = g1.grower('Z').stats
    for i in (304, 613, 834, 1576):
        n0 = st['derive_undone']; why = None
        try: g1.record(i)
        except Reject as e: why = e.reason
        expect(why is not None and not why.startswith('replay') and st['derive_undone'] == n0 + 1, 'B4b seed 1, index %d: the undo of a derivation ran once and the attempt is not lost to a failed replay (%s)' % (i, why))
    # B5  pinned: the digests of the range (the generator and the library, as they are)
    pins = {str(i): fx.recs[i]['digest'] for i in sorted(fx.recs)}
    if os.environ.get('S48_UPDATE_GOLDEN'):
        with open(GOLDEN, 'w') as f: json.dump({'config': fx.cfg.digest(), 'range': list(RANGE), 'world': fx.gen.world_digest, 'digests': pins}, f, indent=0, sort_keys=True)
    gold = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {'config': None, 'range': None, 'digests': {}}
    diff = sorted(set(pins) ^ set(gold['digests'])) + sorted(k for k in pins if k in gold['digests'] and pins[k] != gold['digests'][k])
    expect(gold['config'] == fx.cfg.digest() and gold['range'] == list(RANGE), 'B5 the pinned configuration is the gate\'s (%s)' % os.path.basename(GOLDEN))
    same_world = gold.get('world') == fx.gen.world_digest
    why = ('' if same_world else ' [the LIBRARY is not the one of the pins (%s pinned, %s now): the pool of lemmas decides the theorems, so a library that has grown or changed makes other ones]' % (gold.get('world'), fx.gen.world_digest))
    expect(not diff, 'B5 the theorems of the indices %s..%s are those in %s: %d differ (%s)%s -- if the change is intended: S48_UPDATE_GOLDEN=1 python3 s48_synth_tests.py' % (RANGE[0], RANGE[1], os.path.basename(GOLDEN), len(set(diff)), sorted(set(diff))[:5], why))
    if not same_world: expect(False, 'B5 the pins were made in this library (a pin made in another library says nothing about the generator: re-pin with S48_UPDATE_GOLDEN=1 and read the difference first)')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ C. the records
FIELDS = {'id': str, 'index': int, 'seed': int, 'family': str, 'domain': str, 'statement': str, 'digest': str, 'pretty': str, 'code': list, 'steps': list, 'root_lemma': str, 'area': str, 'lemmas': list,
          'depth': int, 'n_steps': int, 'n_lemmas': int, 'n_hyps': int, 'n_vars': int, 'closers': dict, 'size': int, 'solved_by': list, 'semantic': dict, 'model': list, 'plan': dict, 'recipe': str}
UNTIDY = {'x + 0': r'\+ 0(?![\d.])', '0 + x': r'(?<![\w.])0 \+ ', 'x * 0': r'\* 0(?![\d.])', '0 * x': r'(?<![\w.])0 \* ', 'x * 1': r'\* 1(?![\d.])', '1 * x': r'(?<![\w.])1 \* ', 'x - 0': r'- 0(?![\d.])',
          'double negation': r'--'}            # (patterns of the printed text around a literal; the printer leaves out parentheses, so `x * y - x` is not one of `x - x`: the structural ones below read the AST)


def is_numeral(a): return a[0] == 'int' or (a[0] == 'prim' and a[1] == 'Z.neg' and is_numeral(a[2][0]))


def untidy_in(ast):
    """the structural untidiness in an AST of the evaluator: x - x, a double negation, arithmetic on two numerals (the others are patterns of the text)"""
    out = []
    def walk(a):
        if not isinstance(a, tuple): return
        if a and a[0] == 'prim' and a[1] in SPY.ARITH:
            kids = a[2]
            if a[1] == 'Z.sub' and kids[0] == kids[1]: out.append('x - x')
            if a[1] == 'Z.neg' and kids[0][0] == 'prim' and kids[0][1] == 'Z.neg': out.append('double negation')
            if a[1] != 'Z.neg' and all(is_numeral(x) for x in kids): out.append('arithmetic on two numerals')
        for x in a:
            if isinstance(x, tuple): walk(x)
            elif isinstance(x, list):
                for y in x: walk(y)
    walk(ast)
    return out


def sec_C(c):
    fx = c['fx']; cfg = fx.cfg; gen = fx.gen
    import semantic as SEM
    sem_ = SEM.Semantics(fx.w, window_ex=60)
    bad = collections.defaultdict(list)
    for i, r in sorted(fx.recs.items()):
        for f, ty in FIELDS.items():
            if f not in r or not isinstance(r[f], ty) or isinstance(r[f], bool): bad['field ' + f].append(r['id'])
        if set(r) != set(FIELDS): bad['extra or missing fields'].append(r['id'])
        _, _, plan = gen.plan(i)
        if r['id'] != 's%d-%07d' % (SEED, i) or r['index'] != i or r['seed'] != SEED: bad['id'].append(r['id'])
        if r['family'] != 'backward_tree' or r['domain'] != 'Z': bad['family/domain'].append(r['id'])
        if r['depth'] < plan.min_depth: bad['depth below the plan\'s minimum'].append(r['id'])
        if r['n_hyps'] > cfg.max_hyps: bad['too many hypotheses'].append(r['id'])
        if r['size'] > plan.max_total: bad['too big'].append(r['id'])
        if r['n_steps'] < 2 or r['n_steps'] != len(r['steps']): bad['steps'].append(r['id'])
        if not 2 <= r['n_vars'] <= 5: bad['variables'].append(r['id'])
        if r['n_lemmas'] != len(set(r['lemmas'])): bad['n_lemmas'].append(r['id'])
        sem = r['semantic']
        if sem['sat_distinct'] < cfg.min_sat: bad['hardly satisfiable'].append(r['id'])
        if sem['valid'] and r['n_hyps']: bad['hypotheses are decoration'].append(r['id'])
        if sem['exact'] and r['n_hyps'] and sem['n_needed'] < (r['n_hyps'] + 1) // 2: bad['hypotheses not needed'].append(r['id'])
        if ('Ex ' in r['statement']) != (sem['exact'] is False): bad['exact flag (an existential somewhere in the statement <=> the conclusion is not evaluated)'].append(r['id'])
        if r['solved_by'] and not SG.keep_closable(cfg, r['digest']): bad['closable beyond the quota'].append(r['id'])
        for what, rx in UNTIDY.items():
            if re.search(rx, r['pretty']): bad['untidy: ' + what].append(r['id'])
        stmt = SV.GT.statement_from_code(fx.eng, r['code'], fx.limit)
        for what in untidy_in(sem_.decode(stmt.n)): bad['untidy: ' + what].append(r['id'])
        if SC.digest(r['statement']) != r['digest']: bad['digest'].append(r['id'])
        recipe = sorted(set(r['lemmas']) | {'#' + k for k in r['closers']})
        if hashlib.sha256('|'.join(recipe).encode()).hexdigest()[:16] != r['recipe']: bad['recipe'].append(r['id'])
        if r['root_lemma'] != r['lemmas'][0] and r['lemmas']: bad['root lemma is the first lemma of the proof'].append(r['id'])
        if len(r['model']) != r['n_vars']: bad['model'].append(r['id'])
        if r['pretty'].count('forall') > 1 and not r['pretty'].startswith('forall'): bad['pretty'].append(r['id'])
    expect(not bad, 'C1 every record keeps the promises of the generator: %s' % {k: v[:3] for k, v in bad.items()})
    recs = list(fx.recs.values())
    depths = collections.Counter(r['depth'] for r in recs); areas = collections.Counter(r['area'] for r in recs)
    expect(len(depths) >= 4 and max(depths) >= 5, 'C2 depths vary (%s)' % sorted(depths.items()))
    expect(len(areas) >= 4, 'C2 several areas of root lemmas (%s)' % dict(areas))
    expect(any(r['closers'] for r in recs), 'C2 some proofs use a decision procedure')
    expect(len({r['digest'] for r in recs}) == len(recs), 'C2 the statements are distinct (%d records)' % len(recs))
    expect(len({r['root_lemma'] for r in recs}) >= 8 and len({l for r in recs for l in r['lemmas']}) >= 20, 'C2 a spread of lemmas (%d root lemmas, %d applied)' % (len({r['root_lemma'] for r in recs}), len({l for r in recs for l in r['lemmas']})))
    n_closed = sum(1 for r in recs if r['solved_by'])
    expect(n_closed <= 0.25 * len(recs), 'C3 decision procedures close at most a quarter of the statements at once (%d of %d)' % (n_closed, len(recs)))
    expect(sum(1 for r in recs if r['n_hyps'] >= 2) >= 0.5 * len(recs), 'C3 most theorems have two or more hypotheses')
    # C4  the generator's own accounting: every index is a record or has a counted reason
    expect(len(fx.recs) + len(fx.rejects) == RANGE[1] - RANGE[0], 'C4 every index is a record or a reject (%d + %d)' % (len(fx.recs), len(fx.rejects)))
    expect(all(isinstance(v, str) and v for v in fx.rejects.values()), 'C4 every reject has a reason')
    st = gen.grower('Z').stats
    expect(st.get('false_in_model', 0) >= 0, 'C4 the grower reports its statistics')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ D. from data
def sec_D(c):
    fx = c['fx']; w = fx.w
    eng = G.Engine(w, kinds=set(G.Kind), strict=True)                 # a fresh engine: nothing the generator computed
    proofs = []; bad = []
    for i, r in sorted(fx.recs.items()):
        try: proofs.append((r['id'], SV.check_record(w, eng, fx.limit, r)))
        except ValueError as e: bad.append((r['id'], str(e)[:100]))
    expect(not bad, 'D1 every record replays from its data alone (%d of %d fail: %s)' % (len(bad), len(fx.recs), bad[:3]))
    k = w.k
    second = recheck.SecondOpinion(k, needed_by=[x for _, pair in proofs for x in pair])
    refused = []
    for rid, (proof, stmt) in proofs:
        ok, msg = second.accepts(proof, stmt)
        if not ok: refused.append((rid, str(msg)[:80]))
    expect(not refused, 'D2 the independent checker accepts every proof (%d judged; refused: %s)' % (len(proofs), refused[:3]))
    # a checker that accepts everything checks nothing: the proof of one record against the statement of another
    (_, (p0, s0)), (_, (p1, s1)) = proofs[0], proofs[1]
    ok, _ = second.accepts(p0, s1)
    expect(not ok and s0.n != s1.n, 'D3 the independent checker refuses a proof of another statement')
    # the statement is the one the code says, and the code is what an agent can say: rebuilt and re-encoded it is the same code
    mism = []
    for i, r in sorted(fx.recs.items())[:12]:
        stmt = SV.GT.statement_from_code(eng, r['code'], fx.limit)
        code2 = SC.encode_statement(eng, stmt, fx.limit)
        if code2 != r['code']: mism.append(r['id'])
    expect(not mism, 'D4 encoding the statement of a code gives the code back (%s)' % mism[:3])
    # the count of binders read from the statement agrees with the record
    for i, r in sorted(fx.recs.items())[:12]:
        stmt = SV.GT.statement_from_code(eng, r['code'], fx.limit)
        expect(SV.count_binders(w, stmt) == (r['n_vars'], r['n_hyps']), 'D5 binders of %s' % r['id'])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ E. tampering
def sec_E(c):
    fx = c['fx']; w = fx.w; limit = fx.limit
    eng = G.Engine(w, kinds=set(G.Kind), strict=True)
    recs = [fx.recs[i] for i in sorted(fx.recs)]
    base = next(r for r in recs if r['n_steps'] >= 5 and not r['closers'] and r['n_hyps'] >= 2)
    other = next(r for r in recs if r['digest'] != base['digest'] and r['n_steps'] >= 3)
    expect(SV.check_record(w, eng, limit, base) is not None, 'E0 the untouched record is accepted (the control of every case below)')

    def refused(what, f, rec=base):
        t = copy.deepcopy(rec); f(t)
        try: SV.check_record(w, eng, limit, t)
        except ValueError as e: return expect(True, what + ' [%s]' % str(e)[:60])
        except Exception as e: return expect(False, '%s: raised %s instead of refusing' % (what, type(e).__name__))
        return expect(False, what + ' was ACCEPTED')

    refused('E1 the last step dropped', lambda t: (t['steps'].pop(), t.update(n_steps=t['n_steps'] - 1)))
    refused('E1 the first step dropped', lambda t: (t['steps'].pop(0), t.update(n_steps=t['n_steps'] - 1)))
    refused('E1 a step added', lambda t: (t['steps'].append(copy.deepcopy(t['steps'][-1])), t.update(n_steps=t['n_steps'] + 1)))
    refused('E1 the steps reversed', lambda t: t['steps'].reverse())
    refused('E1 the steps of another theorem', lambda t: t.update(steps=copy.deepcopy(other['steps']), n_steps=other['n_steps']))
    refused('E1 the count of steps changed', lambda t: t.update(n_steps=t['n_steps'] + 1))

    def replace_lemma(t):
        for s in t['steps']:
            if s.get('kind') == 'APPLY' and isinstance(s.get('source'), list) and s['source'][0] == 'c':
                gids = [g for g in range(len(w.gname) if hasattr(w.gname, '__len__') else 0) if g in w.gname and g != s['source'][1] and g < limit]
                s['source'] = ['c', next(g for g in gids if w.gname[g] not in t['lemmas']), s['source'][2]]; return
        raise AssertionError('no APPLY step')
    refused('E2 a lemma replaced by another', replace_lemma)
    def bad_apply(t):
        s = next(s for s in t['steps'] if s.get('kind') == 'APPLY'); s['source'] = ['c', 10 ** 6, 0]

    def bad_with(t):
        s = next(s for s in t['steps'] if s.get('kind') == 'APPLY_WITH'); s['codes'][0][0][1] = 10 ** 6
    refused('E2 an unknown global in an APPLY step', bad_apply)
    withs = next((r for r in recs if any(s.get('kind') == 'APPLY_WITH' for s in r['steps'])), None)
    if expect(withs is not None, 'E2 (a record with an APPLY_WITH step)'): refused('E2 an unknown global in an APPLY_WITH step', bad_with, rec=withs)
    refused('E2 a step of an unknown kind', lambda t: t['steps'][0].update(kind='NO_SUCH_KIND'))
    refused('E2 a step that is not a step', lambda t: t['steps'].__setitem__(0, 7))
    refused('E3 the code of another statement', lambda t: t.update(code=copy.deepcopy(other['code'])))
    refused('E3 a token of the code changed', lambda t: t['code'].__setitem__(len(t['code']) // 2, ['n', 99]))
    refused('E3 the code truncated', lambda t: t.update(code=t['code'][:-3]))
    refused('E3 a code that says nothing', lambda t: t.update(code=[]))
    refused('E3 a code that is not a list of tokens', lambda t: t.update(code=['a', 'b']))
    refused('E4 the printed statement of another theorem', lambda t: t.update(statement=other['statement']))
    refused('E4 the digest of another theorem', lambda t: t.update(digest=other['digest']))
    refused('E4 the readable text of another theorem', lambda t: t.update(pretty=other['pretty']))
    refused('E4 the readable text edited', lambda t: t.update(pretty=t['pretty'].replace('<=', '<', 1) if '<=' in t['pretty'] else t['pretty'] + ' '))
    refused('E5 n_hyps changed', lambda t: t.update(n_hyps=t['n_hyps'] + 1))
    refused('E5 n_vars changed', lambda t: t.update(n_vars=t['n_vars'] + 1))
    refused('E5 the lemma list changed', lambda t: t.update(lemmas=t['lemmas'][:-1]))
    refused('E5 the lemma list reordered is fine, a lemma renamed is not', lambda t: t.update(lemmas=t['lemmas'][:-1] + ['Z.no_such_lemma']))
    refused('E5 closers claimed that the proof does not use', lambda t: t.update(closers={'RING': 1}))
    closing = next((r for r in recs if r['closers']), None)
    if expect(closing is not None, 'E5 (a record whose proof uses a decision procedure)'):
        refused('E5 closers denied that the proof uses', lambda t: t.update(closers={}), rec=closing)
    refused('E5 the lemma list reordered', lambda t: t.update(lemmas=list(reversed(t['lemmas']))))
    refused('E5 the root lemma changed', lambda t: t.update(root_lemma=t['lemmas'][-1] if t['lemmas'][-1] != t['root_lemma'] else t['lemmas'][1]))
    refused('E5 n_lemmas changed', lambda t: t.update(n_lemmas=t['n_lemmas'] + 1))
    refused('E5 the recipe changed', lambda t: t.update(recipe='0' * 16))
    refused('E5 the id changed', lambda t: t.update(id='s%d-%07d' % (t['seed'], t['index'] + 1)))
    refused('E5 the size changed', lambda t: t.update(size=t['size'] + 1))
    refused('E5 an unknown family', lambda t: t.update(family='no_such_family'))
    refused('E5 a witness model that does not satisfy the statement', lambda t: t.update(model=[v + 1000 for v in t['model']]))
    refused('E5 a witness model of the wrong length', lambda t: t.update(model=t['model'][:-1]))
    # the label `solved_by` is found again by running the decision procedures (deep checks)
    closed = next((r for r in recs if r['solved_by']), None)
    if expect(closed is not None, 'E5 (a record that a decision procedure closes at once)'):
        t = copy.deepcopy(closed); t['solved_by'] = []
        try: SV.check_record(w, eng, limit, t, deep=True); expect(False, 'E5 solved_by denied: ACCEPTED by the deep check')
        except ValueError: expect(True, 'E5 a denied solved_by is refused by the deep check')
        try: SV.check_record(w, eng, limit, closed, deep=True); expect(True, 'E5 the deep check accepts the true solved_by')
        except ValueError as e: expect(False, 'E5 the deep check accepts the true solved_by (%s)' % e)
    t = copy.deepcopy(base); t['solved_by'] = ['RING']
    try: SV.check_record(w, eng, limit, t, deep=True); expect(False, 'E5 solved_by invented: ACCEPTED by the deep check')
    except ValueError: expect(True, 'E5 an invented solved_by is refused by the deep check')
    # E6  a file: verify_files reports the id of the bad record, and says the dataset is not ok
    tmp = tempfile.mkdtemp(prefix='s48_')
    try:
        good = recs[:6]; badrec = copy.deepcopy(base); badrec['steps'].pop(); badrec['n_steps'] -= 1
        path = os.path.join(tmp, 'x.jsonl.gz')
        with gzip.open(path, 'wt') as f:
            for r in good + [badrec]: f.write(json.dumps(r) + '\n')
        rep = SV.verify_files([path], second_opinion=False, build=lambda: w)
        expect(not rep['ok'] and rep['n_failed'] == 1 and rep['failed'][0]['id'] == badrec['id'], 'E6 verify_files names the bad record (%s)' % rep['failed'][:1])
        path2 = os.path.join(tmp, 'y.jsonl.gz')
        with gzip.open(path2, 'wt') as f:
            for r in good: f.write(json.dumps(r) + '\n')
        rep = SV.verify_files([path2], second_opinion=True, build=lambda: w)
        expect(rep['ok'] and rep['records'] == 6 and rep['second_opinion_accepted'] == 6, 'E6 verify_files accepts the good ones (%s)' % {k: rep[k] for k in ('ok', 'records', 'second_opinion_accepted', 'n_refused')})
    finally: shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ F. merge
def write_shard(path, recs, a, b):
    with gzip.open(path, 'wt') as f:
        for r in recs: f.write(json.dumps(r, separators=(',', ':')) + '\n')
    json.dump({'format': SG.FORMAT, 'config': 'x', 'generator': 'g', 'world': 'wd', 'range': [a, b], 'records': len(recs), 'rejects': {'too shallow': (b - a) - len(recs)}, 'errors': [], 'n_errors': 0, 'disagreements': [],
               'seconds': 1.0, 'stats': {}}, open(path + '.meta.json', 'w'))


def read_all(d):
    return {s: [l for l in gzip.open(os.path.join(d, s + '.jsonl.gz'), 'rt')] for s in SM.SPLITS}


def sec_F(c):
    fx = c['fx']; w = fx.w
    recs = [fx.recs[i] for i in sorted(fx.recs)]
    tmp = tempfile.mkdtemp(prefix='s48_')
    try:
        # F1  the cut into shards, and their order, do not matter
        d1 = os.path.join(tmp, 'one'); d2 = os.path.join(tmp, 'two'); os.makedirs(d1); os.makedirs(d2)
        a = RANGE[0]; b = RANGE[1]; mid = (a + b) // 2
        write_shard(os.path.join(d1, 'shard_0.jsonl.gz'), recs, a, b)
        lo = [r for r in recs if r['index'] < mid]; hi = [r for r in recs if r['index'] >= mid]
        write_shard(os.path.join(d2, 'shard_b.jsonl.gz'), hi, mid, b); write_shard(os.path.join(d2, 'shard_a.jsonl.gz'), lo, a, mid)
        m1 = SM.merge([os.path.join(d1, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o1'), fx.cfg, world_digest='wd')
        m2 = SM.merge([os.path.join(d2, 'shard_b.jsonl.gz'), os.path.join(d2, 'shard_a.jsonl.gz')], os.path.join(tmp, 'o2'), fx.cfg, world_digest='wd')
        expect(read_all(os.path.join(tmp, 'o1')) == read_all(os.path.join(tmp, 'o2')), 'F1 the merged splits do not depend on how the index space was cut, nor on the order of the shards')
        expect(m1['counts'] == m2['counts'] and m1['records'] == m2['records'] == len(recs) - sum(m1['dropped_in_merge'].values()), 'F1 counts agree (%s)' % m1['counts'])
        # F2  the splits: whole recipes in test_comp; digests unique; every record in the split its rule says
        data = {s: [json.loads(l) for l in lines] for s, lines in read_all(os.path.join(tmp, 'o1')).items()}
        comp = {r['recipe'] for r in data['test_comp']}; rest = {r['recipe'] for s in ('train', 'dev', 'test_iid') for r in data[s]}
        expect(not (comp & rest), 'F2 no recipe of test_comp occurs in train, dev or test_iid (%d recipes held out)' % len(comp))
        allr = [r for s in SM.SPLITS for r in data[s]]
        expect(len({r['digest'] for r in allr}) == len(allr), 'F2 the statements are distinct across the splits')
        expect(all(SM.split_of(r) == r['split'] == s for s in SM.SPLITS for r in data[s]), 'F2 every record is in the split its rule gives')
        expect(all(r['shape'] == SM.shape_of(r['pretty']) for r in allr), 'F2 the shape of every record is the shape of its text')
        # F3  exclusion, duplicates, shape cap
        leak = CLI.leak_digests(w, SUBPROBLEMS if os.path.exists(SUBPROBLEMS) else None)
        lemma_stmt = SC.digest(w.show(w.T(w.k.global_info(w.gid['Z.dvd_trans'])[1])))
        expect(lemma_stmt in leak, 'F3 the digests to exclude include the statement of the library lemma Z.dvd_trans')
        expect(SC.digest(w.show(w.T(w.k.global_info(w.gid['imo1959_p1'])[1]))) in leak, 'F3 ... and of the bank problem imo1959_p1')
        if os.path.exists(SUBPROBLEMS):
            sub = json.load(open(SUBPROBLEMS))['examples']
            expect(all(e['statement_digest'] in leak for e in sub) and len(sub) > 100, 'F3 ... and of every sub-theorem of the bank (%d)' % len(sub))
        expect(not any(r['digest'] in leak for r in recs), 'F3 no generated theorem is a library lemma, a bank problem or a sub-theorem')
        victim = recs[0]; clone = copy.deepcopy(recs[1]); clone['digest'] = victim['digest']; clone['id'] += 'x'
        twin = copy.deepcopy(recs[2]); twin['pretty'] = twin['pretty'].replace('x0', 'x0', 1)       # the same shape as recs[2] (equal text)
        triple = [copy.deepcopy(recs[2]) for _ in range(3)]
        for j, t in enumerate(triple): t['digest'] = '%016x' % (0xabc000 + j); t['id'] = 'dup%d' % j
        d3 = os.path.join(tmp, 'three'); os.makedirs(d3)
        mixed = [victim, clone] + triple + [recs[3]]
        write_shard(os.path.join(d3, 'shard_0.jsonl.gz'), mixed, 0, 10)
        m3 = SM.merge([os.path.join(d3, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o3'), fx.cfg, exclude={recs[3]['digest']}, world_digest='wd', shape_cap=2)
        dm = m3['dropped_in_merge']
        expect(dm.get('duplicate statement') == 1 and dm.get('shape cap') == 1 and dm.get('statement of a library lemma, bank problem or sub-theorem') == 1, 'F3 a duplicate, a third of one shape and an excluded statement are dropped and counted (%s)' % dm)
        expect(m3['records'] == 3, 'F3 and the rest is kept (%d)' % m3['records'])
        # the skeleton cap: records whose proofs have the same skeleton (the same steps, other terms)
        skel = [copy.deepcopy(recs[4]) for _ in range(5)]
        for j, t in enumerate(skel): t['digest'] = '%016x' % (0xdef000 + j); t['id'] = 'sk%d' % j; t['pretty'] = t['pretty'] + ' /\\ ' + ' + '.join(['x0'] * (j + 1)) + ' = 0'     # (another shape each)
        d5 = os.path.join(tmp, 'five'); os.makedirs(d5)
        write_shard(os.path.join(d5, 'shard_0.jsonl.gz'), skel, 0, 10)
        m5 = SM.merge([os.path.join(d5, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o5'), fx.cfg, world_digest='wd', shape_cap=99, skeleton_cap=3)
        expect(m5['records'] == 3 and m5['dropped_in_merge'].get('skeleton cap') == 2, 'F3 at most skeleton_cap records have the same skeleton (%s)' % m5['dropped_in_merge'])
        expect(SM.skeleton_of(recs[4]) == SM.skeleton_of(skel[0]) and len({SM.skeleton_of(r) for r in recs}) > len(recs) // 2, 'F3 the skeleton is a function of the steps (%d skeletons in %d records)' % (len({SM.skeleton_of(r) for r in recs}), len(recs)))
        t = copy.deepcopy(recs[4]); t['steps'] = t['steps'][:-1]
        expect(SM.skeleton_of(t) != SM.skeleton_of(recs[4]), 'F3 ... and another proof is another skeleton')
        m4 = SM.merge([os.path.join(d1, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o4'), fx.cfg, world_digest='wd', target=10)
        expect(m4['records'] == 10 and m4['dropped_in_merge'].get('beyond the target count', 0) > 0, 'F3 the target count caps the dataset (%d)' % m4['records'])
        # F5  the train target: exactly that many train records, and the dataset is the prefix (in index order) of the one without a target; a dry run counts the same and writes nothing
        full = {s_: [json.loads(l) for l in lines] for s_, lines in read_all(os.path.join(tmp, 'o1')).items()}
        k = max(2, len(full['train']) // 2)
        m6 = SM.merge([os.path.join(d1, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o6'), fx.cfg, world_digest='wd', train_target=k)
        part = {s_: [json.loads(l) for l in lines] for s_, lines in read_all(os.path.join(tmp, 'o6')).items()}
        expect(m6['counts']['train'] == k == len(part['train']) and m6['targets'] == {'records': None, 'train': k}, 'F5 the train target gives exactly that many train records (%d)' % k)
        expect(all(part[s_] == full[s_][:len(part[s_])] for s_ in SM.SPLITS), 'F5 every split of the targeted dataset is the beginning of the same split without a target (more shards never change it)')
        expect(len(part['dev']) + len(part['test_iid']) + len(part['test_comp']) < len(full['dev']) + len(full['test_iid']) + len(full['test_comp']), 'F5 the held-out records after the last train record are not kept')
        dry = SM.merge([os.path.join(d1, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o7'), fx.cfg, world_digest='wd', train_target=k, dry_run=True)
        expect(dry['counts'] == m6['counts'] and dry['records'] == m6['records'] and dry['dropped_in_merge'] == m6['dropped_in_merge'] and not os.path.exists(os.path.join(tmp, 'o7')), 'F5 a dry run counts what the merge keeps and writes nothing')
        m8 = SM.merge([os.path.join(d1, 'shard_0.jsonl.gz')], os.path.join(tmp, 'o8'), fx.cfg, world_digest='wd', train_target=10 ** 6)
        expect(m8['counts'] == m1['counts'], 'F5 a train target beyond what exists keeps everything (the caller decides whether that is short)')
        # F6  shards: those of another configuration or generator are refused (never mixed), those of another range are made again, those asked for are not remade
        sd = os.path.join(tmp, 'sh'); os.makedirs(sd)
        sp = os.path.join(sd, 'shard_0.jsonl.gz')
        meta = SG.run_range(fx.cfg, 0, 12, sp, world=fx.w)
        expect(meta['range'] == [0, 12] and meta['config'] == fx.cfg.digest() and meta['generator'] == SG.generator_digest(), 'F6 run_range names the range, the configuration and the generator')
        stamp = os.path.getmtime(sp)
        again = SG.run_range(fx.cfg, 0, 12, sp, world=fx.w)
        expect(again == meta and os.path.getmtime(sp) == stamp, 'F6 a shard that is the one asked for is not remade')
        raises('F6 a shard of another configuration is an error', lambda: SG.run_range(SG.Config(seed=SEED + 1), 0, 12, sp, world=fx.w), RuntimeError)
        mt = json.load(open(sp + '.meta.json')); mt['generator'] = 'someoneelse'; json.dump(mt, open(sp + '.meta.json', 'w'))
        raises('F6 a shard of another generator is an error', lambda: SG.run_range(fx.cfg, 0, 12, sp, world=fx.w), RuntimeError)
        mt['generator'] = SG.generator_digest(); json.dump(mt, open(sp + '.meta.json', 'w'))
        expect(meta['world'] == fx.gen.world_digest == SG.world_digest_of(fx.w, fx.gen.limit), 'F6 run_range names the library the shard was made in (the digest of the world below the limit)')
        mt['world'] = 'another library'; json.dump(mt, open(sp + '.meta.json', 'w'))
        raises('F6 a shard made in another library is an error (the pool of lemmas decides the theorems: the same index makes another one)', lambda: SG.run_range(fx.cfg, 0, 12, sp, world=fx.w), RuntimeError)
        raises('F6 the merge refuses the shard of another library', lambda: SM.merge([sp], os.path.join(tmp, 'o9'), fx.cfg, world_digest=fx.gen.world_digest), RuntimeError)
        del mt['world']; json.dump(mt, open(sp + '.meta.json', 'w'))
        raises('F6 a shard that does not say its library (an older generator) is refused as well', lambda: SG.run_range(fx.cfg, 0, 12, sp, world=fx.w), RuntimeError)
        raises('F6 ... and by the merge', lambda: SM.merge([sp], os.path.join(tmp, 'o9'), fx.cfg, world_digest=fx.gen.world_digest), RuntimeError)
        mt['world'] = fx.gen.world_digest; json.dump(mt, open(sp + '.meta.json', 'w'))
        wider = SG.run_range(fx.cfg, 0, 20, sp, world=fx.w)
        expect(wider['range'] == [0, 20] and [r['index'] for r in SG.read_shard(sp)] == [i for i in sorted(fx.recs) if i < 20], 'F6 a shard of another range is made again, and holds the records of the new range')
        # F4  the manifest: what the card and the paper quote
        man = json.load(open(os.path.join(tmp, 'o1', 'manifest.json')))
        st = man['statistics']
        expect(man['records'] == st['n'] == sum(man['counts'].values()) and man['attempts'] == RANGE[1] - RANGE[0], 'F4 the manifest counts agree (%s)' % man['counts'])
        expect(sum(st['depth'].values()) == st['n'] and sum(st['area'].values()) == st['n'] and sum(st['n_hyps'].values()) == st['n'], 'F4 the distributions add up')
        expect(man['config_digest'] == fx.cfg.digest() and man['split_rule']['salt'] and man['limit'] == 'imo1959_p1', 'F4 the manifest names the configuration, the split rule and the library limit')
        import synth_card as SK
        card = SK.card(man)
        expect('%d kernel-checked theorems' % man['records'] in card and all(s in card for s in ('## Splits', '## Ledger', '## Limitations')), 'F4 the card says what the manifest says')
    finally: shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ G. command line
def sec_G(c):
    fx = c['fx']
    tmp = tempfile.mkdtemp(prefix='s48_')
    py = [sys.executable, os.path.join(itp_paths.SRC_PY, 'synth', 'synth_cli.py')]
    try:
        out = os.path.join(tmp, 'ds')
        def run(*args, timeout=900):
            r = subprocess.run(py + list(args), capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir())
            return r
        r = run('gen', '--out', out, '--seed', str(SEED), '--attempts', '120', '--shard', '60', '--jobs', '2')
        expect(r.returncode == 0 and 'done: 2 shards' in r.stdout, 'G1 gen (two shards, two processes): %s' % (r.stdout + r.stderr)[-300:])
        shards = sorted(os.listdir(os.path.join(out, 'shards')))
        expect(len([s for s in shards if s.endswith('.jsonl.gz')]) == 2 and len([s for s in shards if s.endswith('.meta.json')]) == 2, 'G1 two shards with their meta files (%s)' % shards)
        got = [r for s in sorted(s for s in shards if s.endswith('.jsonl.gz')) for r in SG.read_shard(os.path.join(out, 'shards', s))]
        want = [fx.recs[i] for i in sorted(fx.recs) if i < 120]
        expect(_jsonl(got) == _jsonl(want) and len(want) >= 8, 'G1 the records of the command line are the records made in this process (%d)' % len(want))
        meta = [json.load(open(os.path.join(out, 'shards', s))) for s in shards if s.endswith('.meta.json')]
        expect(all(m['generator'] == SG.generator_digest() and m['config'] == fx.cfg.digest() and m['n_errors'] == 0 for m in meta), 'G1 the meta files name the generator and the configuration, and report no errors')
        # a shard that exists is not remade
        t0 = time.time(); r = run('gen', '--out', out, '--seed', str(SEED), '--attempts', '120', '--shard', '60', '--jobs', '2')
        expect(r.returncode == 0 and time.time() - t0 < 30, 'G1 gen is resumable: a second run reuses the shards (%.0f s)' % (time.time() - t0))
        r = run('merge', '--out', out, '--subproblems', SUBPROBLEMS) if os.path.exists(SUBPROBLEMS) else run('merge', '--out', out)
        expect(r.returncode == 0 and all(os.path.isfile(os.path.join(out, f)) for f in ('train.jsonl.gz', 'dev.jsonl.gz', 'test_iid.jsonl.gz', 'test_comp.jsonl.gz', 'manifest.json')), 'G2 merge: %s' % (r.stdout + r.stderr)[-300:])
        r = run('verify', '--out', out)
        ver = json.load(open(os.path.join(out, 'verification.json'))) if os.path.exists(os.path.join(out, 'verification.json')) else {}
        expect(r.returncode == 0 and ver.get('ok') and ver.get('second_opinion_accepted') == ver.get('records') and ver.get('records', 0) >= 8, 'G3 verify: %s' % (r.stdout + r.stderr)[-300:])
        man = json.load(open(os.path.join(out, 'manifest.json')))
        r = run('card', '--out', out)
        card = open(os.path.join(out, 'DATASET_CARD.md')).read() if os.path.exists(os.path.join(out, 'DATASET_CARD.md')) else ''
        expect(r.returncode == 0 and ('%d kernel-checked theorems' % man['records']) in card and ('**%d of %d records**' % (ver['checked'], ver['records'])) in card and 'World digest' in card, 'G4 card (it quotes the manifest and the verification)')
        # G6  grow: starts from shards that exist, tops up until the train split has the target, and the result is what one large gen and `merge --train-target` give
        w = fx.w; leak = CLI.leak_digests(w, None)
        def count_at(n):
            d = os.path.join(tmp, 'cnt%d' % n); os.makedirs(d)
            write_shard(os.path.join(d, 'shard_0.jsonl.gz'), [fx.recs[i] for i in sorted(fx.recs) if i < n], 0, n)
            return SM.merge([os.path.join(d, 'shard_0.jsonl.gz')], os.path.join(tmp, 'cnt%d_o' % n), fx.cfg, exclude=leak, world_digest='wd', dry_run=True)['counts']['train']
        n120, n180 = count_at(120), count_at(180)
        k = n180
        if expect(n120 < k, 'G6 (the fixture: %d train records in 120 attempts, %d in 180; the test wants a target between)' % (n120, n180)):
            out2 = os.path.join(tmp, 'ds2'); os.makedirs(os.path.join(out2, 'shards'))
            for f in os.listdir(os.path.join(out, 'shards')): shutil.copy(os.path.join(out, 'shards', f), os.path.join(out2, 'shards', f))
            r = run('grow', '--out', out2, '--seed', str(SEED), '--train-target', str(k), '--attempts', '120', '--max-attempts', '1200', '--shard', '60', '--jobs', '2')
            expect(r.returncode == 0 and 'round 1: 120 attempts' in r.stdout and 'round 2:' in r.stdout and 'done: the target is met' in r.stdout, 'G6 grow: a first round that falls short, a second that meets the target: %s' % (r.stdout + r.stderr)[-500:])
            r = run('merge', '--out', out2, '--train-target', str(k))
            m = json.load(open(os.path.join(out2, 'manifest.json')))
            expect(r.returncode == 0 and m['counts']['train'] == k and m['targets']['train'] == k, 'G6 merge --train-target after grow gives exactly the target (%s)' % m['counts'])
            # the same dataset from one large gen: records in index order, whatever was generated beyond
            out3 = os.path.join(tmp, 'ds3')
            top = int(max(f for f in os.listdir(os.path.join(out2, 'shards')) if f.endswith('.jsonl.gz')).split('_')[1].split('.')[0]) + 60
            r = run('gen', '--out', out3, '--seed', str(SEED), '--attempts', str(top), '--shard', '60', '--jobs', '2')
            r = run('merge', '--out', out3, '--train-target', str(k))
            ids = lambda d, s_: [json.loads(l)['id'] for l in gzip.open(os.path.join(d, s_ + '.jsonl.gz'), 'rt')]
            same = all(ids(out2, s_) == ids(out3, s_) for s_ in SM.SPLITS) and len(ids(out2, 'train')) == k
            expect(r.returncode == 0 and same, 'G6 grow + merge --train-target = one gen + merge --train-target (the same records in the same splits)')
            # G7  too short: grow says so and exits 3; merge --train-target fails the same way unless --allow-short
            r = run('grow', '--out', out2, '--seed', str(SEED), '--train-target', '100000', '--attempts', '120', '--max-attempts', '120', '--shard', '60', '--jobs', '1')
            expect(r.returncode == 3 and 'SHORT' in r.stderr, 'G7 grow: --max-attempts reached before the target: SHORT, exit 3 (exit %d: %s)' % (r.returncode, r.stderr[-200:]))
            r = run('merge', '--out', out2, '--train-target', '100000')
            expect(r.returncode == 3 and 'SHORT' in r.stderr, 'G7 merge --train-target with too few train records: SHORT, exit 3')
            r = run('merge', '--out', out2, '--train-target', '100000', '--allow-short')
            expect(r.returncode == 0, 'G7 ... unless --allow-short')
        # a tampered dataset fails `verify` with a non-zero exit code
        path = os.path.join(out, 'train.jsonl.gz')
        lines = [json.loads(l) for l in gzip.open(path, 'rt')]
        if expect(len(lines) >= 3, 'G5 (a train file with records)'):
            lines[1]['steps'] = lines[1]['steps'][:-1]; lines[1]['n_steps'] -= 1
            with gzip.open(path, 'wt') as f:
                for r_ in lines: f.write(json.dumps(r_) + '\n')
            r = run('verify', '--out', out, '--no-second-opinion')
            expect(r.returncode != 0 and lines[1]['id'] in r.stdout, 'G5 verify fails on a tampered dataset and names the record (exit %d)' % r.returncode)
    finally: shutil.rmtree(tmp, ignore_errors=True)


SECTIONS = (('A', sec_A), ('B', sec_B), ('C', sec_C), ('D', sec_D), ('E', sec_E), ('F', sec_F), ('G', sec_G))


def main():
    t_start = time.time()
    c = {'fx': Fixture()}
    for name, fn in SECTIONS:
        if ONLY is not None and name not in ONLY: continue
        t0 = time.time(); n0 = len(FAIL)
        try: fn(c)
        except Exception as e:
            FAIL.append('section %s raised %s: %s' % (name, type(e).__name__, str(e)[:300])); traceback.print_exc(limit=8)
        trace('section %s: %.1f s, %d new failures' % (name, time.time() - t0, len(FAIL) - n0))
    print('S48: %d checks, %d failures, %.1f s' % (CHECKS[0], len(FAIL), time.time() - t_start))
    for f in FAIL[:60]: print('  FAIL:', f)
    print('S48 SYNTHETIC GENERATOR GATE: %s' % ('PASS' if not FAIL else 'FAIL'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
