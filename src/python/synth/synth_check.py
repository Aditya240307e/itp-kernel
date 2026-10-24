"""synth_check.py -- what is known about a generated theorem besides its kernel-checked proof: its meaning on small integers and what the decision procedures do with it.

    code = encode_statement(eng, stmt)                       # the statement in the typed term builder's tokens (what an agent starts an episode from), or None
    prof = semantic_profile(sem, stmt, rng, samples=600)     # satisfiable? each hypothesis needed? valid without hypotheses?  (on a box of small integers)
    sol = solver_profile(eng, stmt, premises)                # which decision procedures close the theorem at once

THE SEMANTIC PROFILE is a sanity layer under the kernel, never a replacement for it. The statement is read into the reference semantics of semantic.py (the meaning of the library's symbols written
down once) and evaluated on random integer assignments:
    sat_rate        the share of assignments that satisfy all hypotheses (0 would mean a vacuous theorem: it is rejected by the generator; the generator also keeps a WITNESS MODEL)
    agreement       among the assignments that satisfy the hypotheses, the conclusion holds -- the kernel proved the theorem, so a counterexample would mean that the evaluator and the library disagree
    valid           the conclusion holds for all sampled assignments whatever the hypotheses say: the hypotheses are decoration
    needed          for each hypothesis, an assignment on which all the OTHER hypotheses hold, this one fails and so does the conclusion: that hypothesis cannot be dropped. (An estimate from below:
                    failing to find such an assignment among the samples does not show the hypothesis is superfluous.)
THE SOLVER PROFILE tries ring, linarith and nlinarith on the theorem's goal once the hypotheses are introduced: a theorem that one of them closes is a calculation, not a search, and is labelled so
(`closer_solvable`); a training set needs to know how many of those it holds.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import hashlib, re
import gym_env as G
import gym_build as GB
import semantic as SEM
from gym_env import Kind


def digest(text): return hashlib.sha256(text.encode()).hexdigest()[:16]


def _literal(text, i):
    """at `i` (just after `Z.of_N (`): the value of a binary literal `N0)` / `Npos <positive>)` and the index after it, or None"""
    if text.startswith('N0)', i): return 0, i + 3
    if not text.startswith('Npos ', i): return None
    i += 5; bits = []
    while True:
        if text.startswith('(XO ', i): bits.append(0); i += 4
        elif text.startswith('(XI ', i): bits.append(1); i += 4
        elif text.startswith('XH', i): i += 2; break
        else: return None
    for _ in range(len(bits) + 1):
        if i >= len(text) or text[i] != ')': return None
        i += 1
    v = 1
    for b in reversed(bits): v = 2 * v + b                 # the outermost constructor is the lowest bit
    return v, i


def pretty(text):
    """the text of a statement for reading: binary literals in decimal (`Z.of_N (Npos (XI (XO XH)))` -> `5`), `Z.zero` / `Z.one` as 0 / 1, `Z.neg x` as `-x`, one `forall` for a run of variables. Digests and
    codes are those of the raw text: this is only for people."""
    out = []; i = 0; head = 'Z.of_N ('; zero = 'Z.of_N N0'
    while i < len(text):
        if text.startswith(head, i):
            got = _literal(text, i + len(head))
            if got is not None: out.append(str(got[0])); i = got[1]; continue
        if text.startswith(zero, i) and (i + len(zero) == len(text) or not (text[i + len(zero)].isalnum() or text[i + len(zero)] in "_.'")):
            out.append('0'); i += len(zero); continue
        out.append(text[i]); i += 1
    t = ''.join(out)
    t = re.sub(r'\bZ\.zero\b', '0', t); t = re.sub(r'\bZ\.one\b', '1', t)
    t = t.replace('Z.neg ', '-')
    t = re.sub(r'\((\d+)\)', r'\1', t)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r'forall \(((?:x\d+ )*x\d+) : Z\), forall \(((?:x\d+ )*x\d+) : Z\),', r'forall (\1 \2 : Z),', t)
    return t


def encode_statement(eng, stmt, premises):
    """the closed statement as a code of the typed term builder (None if the builder cannot say it)"""
    eng.premises = premises
    try:
        space = GB.TermSpace(eng, [], [], roots=[])
        return [list(t) for t in space.encode(getattr(stmt, 'n', stmt), eng.k.mk_type(0))]
    except (GB.BuildError, G.AuthorError, RuntimeError):
        return None


def _has(a, tag):
    """does the AST contain a node of this tag (a tuple whose first element is `tag`)?"""
    if isinstance(a, tuple):
        if a and a[0] == tag: return True
        return any(_has(x, tag) for x in a)
    return False


def semantic_profile(sem, stmt, rng, samples=600, window=6, model=None):
    """see the module docstring; None if the statement uses a symbol the evaluator does not know. Keys:
        sat_rate, sat_count   the share and the number of the sampled assignments (the witness model included) that satisfy all hypotheses; sat_distinct: how many different ones
        exact                 False when the conclusion has an existential: its witnesses can lie outside the evaluator's window, so the conclusion is not evaluated (`agree`, `valid`, `needed` are then None)
        disagreements         assignments that satisfy the hypotheses and falsify the conclusion (always 0 for a correct pair of kernel and evaluator)
        valid                 the conclusion holds on every sample
        needed, n_needed      per hypothesis: found an assignment where only this hypothesis fails and the conclusion fails with it
        hyp_always            per hypothesis: true on every sample (a hypothesis that is a truth)
    Half of the assignments are drawn uniformly from a box of small integers (small values and zero over-represented), half are NEIGHBOURS of the witness model (one or two coordinates moved by up to 3):
    the neighbours are where a hypothesis fails alone."""
    try:
        ast = sem.decode(getattr(stmt, 'n', stmt))
        kinds, hyps, concl = sem.parts(ast)
    except SEM.Unsupported:
        return None
    if not all(kd == 'Z' for kd in kinds): return None
    n = len(kinds); exact = not _has(concl, 'ex')
    fh = [sem.compile(h) for h in hyps]; fc = sem.compile(concl) if exact else None
    envs = []
    if model is not None: envs.append(tuple(model))
    special = (-2, -1, 0, 1, 2, 3)
    for j in range(samples):
        if model is not None and j % 2 == 1:
            env = list(model)
            for _ in range(rng.choice((1, 1, 2))):
                i = rng.randrange(n); env[i] += rng.choice((-3, -2, -1, 1, 2, 3))
            envs.append(tuple(env))
        else:
            envs.append(tuple(rng.choice(special) if rng.random() < 0.4 else rng.randint(-window, window) for _ in range(n)))
    sat = 0; agree_bad = 0; valid = exact; need = [False] * len(hyps); always = [True] * len(hyps); sat_envs = set()
    for env in envs:
        hv = [bool(f(env)) for f in fh]
        for i, v in enumerate(hv):
            if not v: always[i] = False
        cv = bool(fc(env)) if exact else True
        if not cv: valid = False
        if all(hv):
            sat += 1; sat_envs.add(env)
            if not cv: agree_bad += 1
        elif not cv:
            bad = [i for i, v in enumerate(hv) if not v]
            if len(bad) == 1: need[bad[0]] = True              # all the other hypotheses hold, this one fails, and so does the conclusion
    return dict(sat_rate=round(sat / len(envs), 4), sat_count=sat, sat_distinct=len(sat_envs), exact=exact, disagreements=agree_bad, valid=valid if exact else None, needed=need if exact else None,
                n_needed=sum(need) if exact else None, hyp_always=always, n_samples=len(envs))


CLOSERS = (Kind.RING, Kind.LINARITH, Kind.NLINARITH)


def solver_profile(eng, stmt, premises, level=2):
    """which decision procedures close the theorem right after its hypotheses are introduced (with a work budget of the given fuel level): a list of kind names"""
    eng.premises = premises
    ep = G.Episode(eng, stmt, premises=premises)
    if ep.n_goals == 0: return ['SETTLED']
    g = ep.goal(); menu = ep.menu(); out = []
    for kind in CLOSERS:
        acts = [a for a in menu.actions if a.kind == kind]
        if not acts: continue
        kids, status = eng.probe_status(g, acts[0], menu, fuel=eng.fuel_levels(kind)[level])
        if kids is not None and not kids: out.append(Kind(kind).name)
    return out
