import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths
import nat_lib, recheck

w = nat_lib.build_nat()
TYPE, PROP, False_ = w.TYPE, w.PROP, w['False']

def attempt(label, make):
    try: make(); print('%s: accepted' % label)
    except Exception as e: print('%s: refused (%s)' % (label, str(e).split('rejected: ')[-1]))

# control: the family on the left of an arrow in a field is refused
attempt('B0, field (B0 -> False) -> B0', lambda: w.inductive('B0', TYPE, [('mk0', [lambda *_: w.arrow(w.arrow(w['B0'], False_), w['B0'])])]))
# the family inside the index of a recursive field
attempt('Q A, field Q (Q A -> False)', lambda: w.inductive('Q', w.arrow(PROP, TYPE), [('mkq', [PROP, lambda A: w['Q'](w.arrow(w['Q'](A), False_))], lambda A, h: [A])]))
# the family inside a result index of a constructor
attempt('R A, result index R A -> False', lambda: w.inductive('R', w.arrow(PROP, TYPE), [('mkr', [PROP], lambda A: [w.arrow(w['R'](A), False_)])]))
n, bad = recheck.recheck_world(w)
print('independent checker: %d globals re-checked, problems: %s' % (n, bad))
