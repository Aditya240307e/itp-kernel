"""synth_verify.py -- check a synthetic dataset from its DATA ALONE: nothing the generator computed is believed.

    report = verify_files(paths, limit_name='imo1959_p1', second_opinion=True, batch=500)
    report['ok']                       # True iff every record passed every check

The generator is a program that can have bugs; the data are the claim. For every record this module, in a fresh world (the library, rebuilt) and a fresh engine, with none of the generator's code:
  1. says the statement again from its CODE (the typed term builder's tokens) and compares it with the printed statement and its digest;
  2. replays the proof from its portable STEPS -- each action refused or accepted by the engine, the proof term checked by the kernel against that statement;
  3. checks the labels that can be read off the data: the id, the number of steps, of hypotheses and variables, the lemmas the steps apply (by name, in order), the root lemma, the proof's closers, the
     recipe (a hash of both), the size of the statement, and that the witness model satisfies the hypotheses and the conclusion (a statement without a model would be vacuous or false);
  3'. (deep) the decision procedures that close the statement at once -- the label `solved_by` -- are found again by running them;
  4. (second_opinion) hands every proof and its statement to the independent checker (`recheck.SecondOpinion`, which shares no code with the kernel and fails closed on anything it is not equipped for).
A record that fails is reported with the reason; the dataset's ledger (`manifest.json`) is not part of the check.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import collections, gzip, hashlib, json, time
import gym_env as G
import gym_acts as A
import gym_tree as GT
import semantic as SEM
import synth_check as SC


def read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def step_lemmas(step):
    """the global id a portable step applies, or None (a decision procedure, a hypothesis, ...)"""
    if step.get('kind') == 'APPLY' and isinstance(step.get('source'), list) and step['source'] and step['source'][0] == 'c': return step['source'][1]
    if step.get('kind') == 'APPLY_WITH' and step.get('codes'):
        c = step['codes'][0]
        if len(c) == 1 and c[0][0] == 'c': return c[0][1]
    return None


def count_binders(w, stmt):
    """(leading variables, hypotheses) of a closed statement `forall vars, hyps -> concl` (as the generator builds it: first the variables, then the hypotheses): a binder is a hypothesis when its domain is a proposition"""
    k = w.k; doms, concl = G.open_binders(k, getattr(stmt, 'n', stmt), 0); nv = 0; nh = 0
    for j, d in enumerate(doms):
        ok, sort = k.infer_open(d, doms[:j])
        if ok and sort == w.PROP.n: nh += 1
        else:
            if nh: raise ValueError('a variable binder after a hypothesis')
            nv += 1
    return nv, nh


FAMILIES = ('backward_tree',)
DOMAINS = ('Z',)


def tree_size(k, node, numeral, memo):
    """the size of a term as a tree: operators, binders and leaves, a shared subterm counted every time it is written, a binary numeral (`Z.of_N <literal>`) as one (the same measure as the generator's, written again)"""
    r = memo.get(node)
    if r is not None: return r
    v = k.view(node); t = v[0]
    if t == 'app':
        r = 1 if numeral is not None and k.view(v[1]) == ('const', numeral) else tree_size(k, v[1], numeral, memo) + tree_size(k, v[2], numeral, memo)
    elif t in ('lam', 'pi'): r = tree_size(k, v[1], numeral, memo) + tree_size(k, v[2], numeral, memo) + 1
    else: r = 1
    memo[node] = min(r, 10_000)
    return memo[node]


_SEM = {}


def model_satisfies(w, stmt, model):
    """does the record's witness model satisfy every hypothesis and the conclusion?  (the reference semantics of semantic.py; an existential gets a wide window of witnesses.) None if the evaluator cannot read the statement"""
    hit = _SEM.get(id(w))
    if hit is None or hit[0] is not w: hit = _SEM[id(w)] = (w, SEM.Semantics(w, window_ex=60))
    sem = hit[1]
    try:
        ast = sem.decode(getattr(stmt, 'n', stmt)); kinds, hyps, concl = sem.parts(ast)
    except SEM.Unsupported:
        return None
    if len(kinds) != len(model): return False
    env = tuple(model)
    return all(bool(sem.compile(h)(env)) for h in hyps) and bool(sem.compile(concl)(env))


def check_record(w, eng, limit, rec, deep=False):
    """replay one record; returns (proof, statement) or raises ValueError with the reason. `deep`: also find the label `solved_by` again by running the decision procedures"""
    try:
        stmt = GT.statement_from_code(eng, rec['code'], limit)
    except Exception as e:
        raise ValueError('the code does not say a statement: %s' % str(e)[:100])
    text = w.show(stmt)
    if text != rec['statement']: raise ValueError('the code says another statement than the printed one')
    if SC.digest(text) != rec['digest']: raise ValueError('the digest is not the statement\'s')
    if SC.pretty(text) != rec['pretty']: raise ValueError('the readable text is not the statement\'s')
    steps = rec['steps']
    if len(steps) != rec['n_steps']: raise ValueError('n_steps is %d, the proof has %d steps' % (rec['n_steps'], len(steps)))
    try:
        proof = A.replay_portable(eng, stmt, steps, premises=limit)
    except (A.ActionError, G.AuthorError, RuntimeError) as e:
        raise ValueError('the proof does not replay: %s' % str(e)[:100])
    nv, nh = count_binders(w, stmt)
    if nv != rec['n_vars']: raise ValueError('n_vars is %d, the statement has %d variables' % (rec['n_vars'], nv))
    if nh != rec['n_hyps']: raise ValueError('n_hyps is %d, the statement has %d hypotheses' % (rec['n_hyps'], nh))
    applied = [w.gname.get(step_lemmas(s)) for s in steps if step_lemmas(s) is not None]
    if applied != rec['lemmas']: raise ValueError('the lemmas the steps apply are not the lemmas of the record')
    if rec['n_lemmas'] != len(set(applied)): raise ValueError('n_lemmas is %d, the steps apply %d lemmas' % (rec['n_lemmas'], len(set(applied))))
    if applied and rec['root_lemma'] != applied[0]: raise ValueError('the root lemma is not the first lemma the proof applies')
    closers = collections.Counter(s['kind'] for s in steps if s.get('kind') in ('RING', 'LINARITH', 'NLINARITH'))
    if dict(closers) != rec['closers']: raise ValueError('the closers of the record are not the proof\'s')
    recipe = sorted(set(applied) | {'#' + c for c in closers})
    if hashlib.sha256('|'.join(recipe).encode()).hexdigest()[:16] != rec['recipe']: raise ValueError('the recipe is not the hash of the proof\'s lemmas and closers')
    if rec['id'] != 's%d-%07d' % (rec['seed'], rec['index']): raise ValueError('the id is not the one of the seed and the index')
    if rec['family'] not in FAMILIES or rec['domain'] not in DOMAINS: raise ValueError('unknown family or domain')
    of_n = w.gid.get('Z.of_N') if rec['domain'] == 'Z' else None
    if tree_size(w.k, getattr(stmt, 'n', stmt), of_n, {}) != rec['size']: raise ValueError('the size of the record is not the size of the statement')
    sat = model_satisfies(w, stmt, rec['model'])
    if sat is not True: raise ValueError('the witness model does not satisfy the statement' if sat is False else 'the evaluator cannot read the statement')
    if deep:
        if SC.solver_profile(eng, stmt, limit) != rec['solved_by']: raise ValueError('the decision procedures that close the statement at once are not those of the record')
    return proof, stmt


def verify_files(paths, limit_name='imo1959_p1', second_opinion=True, batch=500, build=None, sample=None, progress=None, deep_every=20):
    """see the module documentation. `sample`: check every `sample`-th record only (None: all); `deep_every`: run the decision procedures again on every that-many-th record (0: never)"""
    import imo_lib
    w = (build or imo_lib.build_imo)()
    limit = w.gid[limit_name]
    eng = G.Engine(w, kinds=set(G.Kind), strict=True)
    t0 = time.time(); n = 0; bad = []; accepted = 0; refused = []; pend = []; recs_pend = []
    so_seconds = 0.0
    def flush():
        nonlocal accepted, so_seconds
        if not pend or not second_opinion: pend.clear(); recs_pend.clear(); return
        import recheck as RC
        t1 = time.time()
        so = RC.SecondOpinion(w.k, needed_by=[x for pair in pend for x in pair])
        for (proof, stmt), rec in zip(pend, recs_pend):
            ok, msg = so.accepts(proof, stmt)
            if ok: accepted += 1
            else: refused.append({'id': rec['id'], 'message': str(msg)[:200]})
        so_seconds += time.time() - t1; pend.clear(); recs_pend.clear()
    for path in paths:
        for j, rec in enumerate(read_records(path)):
            n += 1
            if sample and (n % sample): continue
            try:
                proof, stmt = check_record(w, eng, limit, rec, deep=bool(deep_every) and n % deep_every == 0)
            except ValueError as e:
                bad.append({'id': rec['id'], 'error': str(e)}); continue
            pend.append((proof, stmt)); recs_pend.append(rec)
            if len(pend) >= batch: flush()
            if progress and n % 1000 == 0: progress(n, len(bad), time.time() - t0)
    flush()
    checked = n if not sample else n // sample
    return {'ok': not bad and not refused, 'records': n, 'checked': checked, 'failed': bad[:50], 'n_failed': len(bad), 'second_opinion': second_opinion, 'second_opinion_accepted': accepted,
            'second_opinion_refused': refused[:50], 'n_refused': len(refused), 'seconds': round(time.time() - t0, 1), 'second_opinion_seconds': round(so_seconds, 1)}
