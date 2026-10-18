"""Finding F30: a wrongly instantiated `apply` used to run for minutes (F29); with the work budget it fails cleanly.

    python3 repros/wrong_apply_budget.py [fuel]          (default 3e8 steps, about 10 s)

The statement is Z.le_of_lt, `x < y -> x <= y`. The call `apply Z.le_trans (x + 1)` is a mistake: an explicit argument binds the FIRST quantified variable of the lemma (`a` in
`le_trans a b c`), not the middle one that was meant, so the lemma's other variables stay unknown and unifying its conclusion with the goal drives `conv` into the shared-term blow-up of
conv_exponential_shared.py. The call ends in FuelExhausted, the proof state is exactly what it was, and the script goes on.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import os, sys, time
sys.setrecursionlimit(100000)
import bin_lib, quot_lib, int_lib
from author import AuthorError
w = bin_lib.build_bin_defs()
quot_lib.declare_quotients(w); quot_lib.declare_prod(w); quot_lib.quot_lift2(w)
for stage in (int_lib.bool_lib, int_lib.int_core, int_lib.int_ops, int_lib.int_pow, int_lib.int_nat, int_lib.int_ring, int_lib.int_order): stage(w)
Z = w['Z']
zle, zlt, zadd, zone = w['Z.le'], w['Z.lt'], w['Z.add'], w['Z.one']
w.k.set_fuel(int(sys.argv[1]) if len(sys.argv) > 1 else 300000000)
def script(p):
    p.intro('x', 'y', 'h'); t0 = time.time()
    try: p.apply('Z.le_trans', (zadd, 'x', zone))
    except AuthorError as e: print('apply failed after %.1f s: %s' % (time.time() - t0, str(e).split('\n')[0][:260]))
    raise AuthorError('stop')
try: w.prove('t.x', w.forall_([Z, Z, lambda x, y: zlt(x, y)], lambda x, y, h: zle(x, y)), script)
except AuthorError as e: print('done:', str(e)[:60])
