"""named.py -- a small NAMED term language and the verdict helpers shared by the gates that write terms by hand (S16 quotients, S17 proof irrelevance).

Terms are position independent: ('Lam', x, T, b) ('Pi', x, T, b) ('app', f, a) ('Match', s, motive, branches) ('Fix', d, self_name, fn_type, body) ('const', gid) ('type', k) and bare strings for
variables. `lv` turns one into the absolute-level trees of harness.py, `HY` into a hybrid tree. `kernel_ok` / `indep_ok` give the verdicts of the C++ kernel and of the independent checker on
`proof : statement`; a gate that demands both to agree is a differential test of the two implementations."""
import harness as H
from harness import *
from s3_tests import V
import indep_checker as ic

# ---------------------------------------------------------------------------------------------------------------- a small NAMED term language (position independent)
def lam(x, T, b): return ('Lam', x, T, b)
def pi(x, T, b): return ('Pi', x, T, b)
def arrow(A, B): return ('Pi', '_', A, B)
def ap(f, *xs):
    for x in xs: f = ('app', f, x)
    return f
def match(s, motive, *branches): return ('Match', s, motive, tuple(branches))

def lv(t, ctx=()):
    """named term -> the level trees of harness.py (absolute levels; ctx = names of the enclosing binders, outermost first)"""
    if isinstance(t, str):
        for i in range(len(ctx) - 1, -1, -1):
            if ctx[i] == t: return V(i)
        raise KeyError('unbound name ' + t)
    tag = t[0]
    if tag == 'Lam': return ('lam', lv(t[2], ctx), lv(t[3], ctx + (t[1],)))
    if tag == 'Pi': return ('pi', lv(t[2], ctx), lv(t[3], ctx + (t[1],)))
    if tag == 'app': return ('app', lv(t[1], ctx), lv(t[2], ctx))
    if tag == 'Match': return ('match', lv(t[1], ctx), lv(t[2], ctx), tuple(lv(b, ctx) for b in t[3]))
    if tag == 'Fix': return ('fix', t[1], lv(t[3], ctx), lv(t[4], ctx + (t[2],)))       # ('Fix', decreasing_arg, self_name, fn_type, body): self is the first binder of the body
    if tag in ('const', 'type'): return t
    raise ValueError(tag)
HY = lambda t: l2h(lv(t), 0)                       # closed named term -> hybrid tree


def gmap(k): return {name: gid for gid, name, _, _ in k.list_globals()}

def kernel_type_ok(k, stmt):
    try: return bool(k.infer_open(build_new(k, HY(stmt)), [])[0])
    except Exception: return False

def kernel_ok(k, proof, stmt):
    try:
        s = build_new(k, HY(stmt))
        return bool(k.infer_open(s, [])[0]) and bool(k.check_open(build_new(k, HY(proof)), s, [])[0])
    except Exception: return False

def indep_ok(chk, proof, stmt): return chk.check_closed(HY(proof), HY(stmt))[0]
def indep_type_ok(chk, stmt): return chk.run(lambda: chk.sort_of({}, ic.from_hybrid(HY(stmt))))[0]

FAIL = []
def expect(cond, what):
    if not cond: FAIL.append(what)
    return cond

def refuses(thunk, needle):
    """the call must raise, with `needle` in the message"""
    try: thunk()
    except Exception as e: return needle in str(e)
    return False
