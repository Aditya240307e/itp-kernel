"""Probes (checkpoint -> try a tactic -> rewind) must survive the kernel lazily synthesising its own derived
globals (the `same_ctor` discriminator, __Unit, __Empty) inside the probe, and rewinding must still REFUSE when a
genuine (non-derived) global was declared since the checkpoint."""
import curriculum as cur, itp_env

def build():
    k, refs = cur.build_base_kernel(); refs.update(cur.setup_nat_order(k, refs)); return k, refs

def refutable_goal(k, refs):
    """Pi n. Le (Succ n) Zero -> Eq Nat Zero Zero.  cases on the Le hypothesis refutes le_S (Succ m vs Zero) and
    therefore exploits it, which builds same_ctor(Nat) on first use."""
    nat, zero, succ = refs['nat_ty'], refs['zero'], refs['succ_c']
    le = lambda a, b: cur.le_app(k, refs, a, b)
    ty = k.mk_pi(nat, k.mk_pi(le(k.mk_app(succ, k.mk_var(0)), zero), cur.eq_app(k, refs, nat, zero, zero)))
    return ty

def probe(k, refs, ty):
    ps = k.start_proof(ty)
    ps.intro('n', target_meta_id=ps.goal_meta_id(0)); ps.intro('h', target_meta_id=ps.goal_meta_id(0))
    ps.cases(1, target_meta_id=ps.goal_meta_id(0))      # hyp 1 = h : Le (Succ n) Zero
    return ps.num_goals()

k, refs = build(); ty = refutable_goal(k, refs)
size0 = k.library_size()
cp = k.checkpoint()
n_goals = probe(k, refs, ty)
size_in_probe = k.library_size()
assert size_in_probe > size0, "expected the probe to synthesise derived globals (same_ctor / Unit / Empty)"
k.rewind(cp)                                            # used to raise: 'a global was declared since this checkpoint'
assert k.library_size() == size0, (k.library_size(), size0)
print("PASS  probe that lazily builds same_ctor rewinds cleanly (library %d -> %d -> %d)" % (size0, size_in_probe, k.library_size()))

cp = k.checkpoint(); n_goals2 = probe(k, refs, ty); k.rewind(cp)
assert n_goals2 == n_goals and k.library_size() == size0
print("PASS  the same probe is repeatable after rewind (derived globals rebuilt identically; %d goals both times)" % n_goals)

# a real (non-derived) declaration since the checkpoint must still block rewind
cp = k.checkpoint()
nat = refs['nat_ty']
k.declare_checked(k.mk_pi(nat, nat), k.mk_lambda(nat, k.mk_var(0)), 'my_id')
try:
    k.rewind(cp)
except Exception as e:
    assert 'global was declared' in str(e); print("PASS  rewind still refuses when a genuine global was declared since the checkpoint")
else:
    raise AssertionError("rewind silently accepted a kept global -- the safety contract is broken")

# and a kept derived global (no rewind) keeps working: same_ctor built once, reused
k2, refs2 = build(); ty2 = refutable_goal(k2, refs2)
probe(k2, refs2, ty2); s1 = k2.library_size(); probe(k2, refs2, ty2)
assert k2.library_size() == s1, "same_ctor must be built once and reused"
print("PASS  without rewind, derived globals are built once and reused")
print("ALL REWIND-DERIVED-GLOBAL TESTS PASSED")
