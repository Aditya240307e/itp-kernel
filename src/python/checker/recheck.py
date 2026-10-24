"""recheck.py -- a second opinion: hand a finished world to the independently written checker.

    import recheck
    count, problems = recheck.recheck_world(w)        # w: an author.World (or a bare kernel);  problems == []  iff the second checker accepts every global

The kernel (src/c++/hybrid) decides what is accepted.  indep_checker.py is a second implementation of the same typing rules, written separately in Python, that shares no code with the kernel and
reads the kernel's exported terms (`dump_raw`).  Neither is trusted blindly: a proof is believed when both accept it.  This module is the glue that every gate used to repeat: parse the exported
terms, fill an environment, run `Checker.check_definitions()` over it.  `parse_raw` understands the HYBRID kernel's export format (de Bruijn indices `b0`, context levels `f0`).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import re
import indep_checker as ic

_RAW = re.compile(r'\s*(\(fix<\d+>|\(lam|\(pi|\(match|\(let|\bin\b|b\d+|f\d+|T\d+|C\d+|\?\d+|<other>|\(|\)|\[|\]|\||:=|\.)')


def parse_raw(s, new=True):
    """the kernel's text export of a term -> the tuple tree the independent checker reads:  ('bvar',i) ('fvar',l) ('type',n) ('const',gid) ('app',f,a) ('lam',T,B) ('pi',D,C) ('let',T,V,B) ('match',S,M,(b..)) ('fix',decreasing,T,B)"""
    if not new: raise ValueError('only the hybrid kernel\'s export format is understood here')
    toks = [m.group(1) for m in _RAW.finditer(s)]; pos = [0]
    def eat(): t = toks[pos[0]]; pos[0] += 1; return t
    def peek(): return toks[pos[0]]
    def term():
        t = eat()
        if t[0] == 'b' and t[1:].isdigit(): return ('bvar', int(t[1:]))
        if t[0] == 'f' and t[1:].isdigit(): return ('fvar', int(t[1:]))
        if t[0] == 'T': return ('type', int(t[1:]))
        if t[0] == 'C': return ('const', int(t[1:]))
        if t[0] == '?': return ('meta', int(t[1:]))
        if t == '<other>': return ('other',)
        if t in ('(lam', '(pi'):
            a = term(); assert eat() == '.'; b = term(); assert eat() == ')'; return (t[1:], a, b)
        if t == '(let':
            ty = term(); assert eat() == ':='; v = term(); assert eat() == 'in'; b = term(); assert eat() == ')'; return ('let', ty, v, b)
        if t.startswith('(fix<'):
            d = int(t[5:-1]); ty = term(); assert eat() == ':='; b = term(); assert eat() == ')'; return ('fix', d, ty, b)
        if t == '(match':
            s_ = term(); assert eat() == '['; m = term(); assert eat() == ']'; br = []
            while peek() == '|': eat(); br.append(term())
            assert eat() == ')'; return ('match', s_, m, tuple(br))
        if t == '(':
            f = term(); a = term(); assert eat() == ')'; return ('app', f, a)
        raise ValueError(t)
    r = term(); assert pos[0] == len(toks), (s[:80], pos[0], len(toks)); return r


_CONST = re.compile(r'\bC(\d+)')


def consts_of(text):
    """the global ids a kernel text export mentions (a scan, no parse)"""
    return {int(m) for m in _CONST.findall(text)}


def needed_globals(k, roots=()):
    """the global ids the independent checker needs in order to judge the closed terms `roots` (author.Terms or kernel nodes): the globals they mention, and -- transitively -- those mentioned by the type and
    the value of each of them, the constructors of an inductive type, and the globals that give the quotient its meaning (the checker looks for them in the whole environment).  Found in the kernel's text
    export, without parsing it: the expensive part of an environment is the parse, and a world that holds the proofs of a whole problem bank spends nearly all of it on proofs that a given term has nothing to
    do with.  Two links need no code: the family of a constructor is the head of the constructor's type, and the family a recursive function recurses on is in the function's type (or in the value of a
    definition that the type mentions); the gate S47 checks that the set is closed under both.  The constructors are the link that a scan cannot find: a `match` names none of them."""
    seen = set(); stack = []
    def want(gid):
        if gid not in seen: seen.add(gid); stack.append(gid)
    for r in roots:
        for g in consts_of(k.dump_raw(getattr(r, 'n', r))): want(g)
    for gid in range(k.num_globals()):
        if k.quot_role(gid): want(gid)
    while stack:
        gid = stack.pop()
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        for g in consts_of(k.dump_raw(ty)): want(g)
        if val is not None:
            for g in consts_of(k.dump_raw(val)): want(g)
        for g in ctors: want(g)
    return seen


def build_env(k, gids=None):
    """an independent-checker environment holding every global of the kernel `k` (or only those in `gids`)"""
    env = ic.Env()
    for gid in range(k.num_globals()):
        if gids is not None and gid not in gids: continue
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty)), None if val is None else parse_raw(k.dump_raw(val)), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    return env


class SecondOpinion:
    """the independent checker over every global of a kernel as it is NOW (make it after the proofs it will judge: a replay may add derived globals).
    `needed_by`: closed terms (author.Terms or kernel nodes); then the environment holds only the globals that they need (`needed_globals`) -- seconds instead of the minute that a world with a bank's proofs
    in it costs -- and `accepts` judges only terms that mention nothing else (it says so and refuses otherwise); `problems()` re-checks only those globals."""
    def __init__(self, k, fuel=200_000_000, needed_by=None):
        self.k = k; self.gids = None if needed_by is None else needed_globals(k, needed_by)
        self.checker = ic.Checker(build_env(k, self.gids), params=True, fuel=fuel, prop_elim=True)

    def accepts(self, proof, statement):
        """(ok, message): is `proof` a proof of `statement` for the independent checker?  Both are closed terms (author.Term objects or kernel nodes)"""
        raw = [self.k.dump_raw(getattr(x, 'n', x)) for x in (proof, statement)]
        if self.gids is not None:
            missing = consts_of(raw[0]) | consts_of(raw[1])
            missing -= self.gids
            if missing: return False, 'the term mentions %d global(s) that this environment was not made for (%s)' % (len(missing), sorted(missing)[:5])
        return self.checker.check_closed(parse_raw(raw[0]), parse_raw(raw[1]))

    def problems(self):
        """every global re-checked: a list of (gid, name, message), empty iff the independent checker accepts the whole environment"""
        return self.checker.check_definitions()


def recheck_world(w, fuel=200_000_000):
    """(number of globals, problems): every global of the world `w` (an author.World, or a bare kernel) re-checked from scratch by the independent checker;  problems == []  iff all is accepted"""
    so = SecondOpinion(getattr(w, 'k', w), fuel)
    bad = so.problems()
    return len(so.checker.g), bad
