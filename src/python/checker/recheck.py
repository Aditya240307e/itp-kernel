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


def build_env(k):
    """an independent-checker environment holding every global of the kernel `k`"""
    env = ic.Env()
    for gid in range(k.num_globals()):
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        env.add(gid, name, parse_raw(k.dump_raw(ty)), None if val is None else parse_raw(k.dump_raw(val)), cf, ci, ctors, k.quot_role(gid), k.is_inductive(gid),
                recursion=k.recursive_info(gid), reserved=k.reserved_kind(gid))
    return env


class SecondOpinion:
    """the independent checker over every global of a kernel as it is NOW (make it after the proofs it will judge: a replay may add derived globals)"""
    def __init__(self, k, fuel=200_000_000):
        self.k = k; self.checker = ic.Checker(build_env(k), params=True, fuel=fuel, prop_elim=True)

    def accepts(self, proof, statement):
        """(ok, message): is `proof` a proof of `statement` for the independent checker?  Both are closed terms (author.Term objects or kernel nodes)"""
        tree = lambda x: parse_raw(self.k.dump_raw(getattr(x, 'n', x)))
        return self.checker.check_closed(tree(proof), tree(statement))

    def problems(self):
        """every global re-checked: a list of (gid, name, message), empty iff the independent checker accepts the whole environment"""
        return self.checker.check_definitions()


def recheck_world(w, fuel=200_000_000):
    """(number of globals, problems): every global of the world `w` (an author.World, or a bare kernel) re-checked from scratch by the independent checker;  problems == []  iff all is accepted"""
    so = SecondOpinion(getattr(w, 'k', w), fuel)
    bad = so.problems()
    return len(so.checker.g), bad
