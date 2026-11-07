import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths
import nat_lib, recheck

w = nat_lib.build_nat()
Nat, Zero, False_ = w['Nat'], w['Zero'], w['False']
B2 = w.inductive('B2', w.TYPE, [('b2f', []), ('b2t', [])])         # a second data type, unrelated to Nat

def attempt(label, make, name=None):
    try: t = make(); ty = w.type_of(t)
    except Exception as e: print('%s: refused (%s)' % (label, str(e).split('\n')[0])); return
    print('%s: the kernel infers %s' % (label, w.show(ty)))
    if name:                                                         # the kernel's checked entry point
        try: w.k.declare_checked(ty.n, t.n, name); print('    declared: accepted')
        except Exception as e: print('    declared: refused (%s)' % str(e).split('\n')[0])

# (a) match on a Nat with a motive whose binder is a B2
attempt('(a) motive over B2', lambda: w.fun([Nat], lambda n: w.match(n, w.fun([B2], lambda z: Nat), [Zero, lambda k: Zero])), 'probe_a')
# (b) a motive whose codomain is a term (Zero) and not a sort
attempt('(b) motive into Zero', lambda: w.fun([False_], lambda e: w.match(e, w.fun([False_], lambda z: Zero), [])), 'probe_b')
# (c) a motive without a binder
attempt('(c) motive Nat', lambda: w.fun([Nat], lambda n: w.match(n, Nat, [Zero, lambda k: Zero])))
n, bad = recheck.recheck_world(w)
print('independent checker: %d globals re-checked, problems: %s' % (n, bad))
