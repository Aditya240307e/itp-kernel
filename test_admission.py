"""F1: inductive admission checker (well-typed fields/indices, universe rule). Negatives must be rejected
with a specific reason; legitimate shapes must be accepted."""
import curriculum, itp_env

def fresh():
    k, refs = curriculum.build_base_kernel()
    return k, refs, itp_env.CtorSpec

def rejects(thunk, needle):
    try:
        thunk()
    except Exception as e:
        assert needle in str(e), "rejected, but for the wrong reason: %s (wanted %r)" % (e, needle)
        return
    raise AssertionError("was ACCEPTED but must be rejected (%s)" % needle)

# ---- negatives ----
k, r, CS = fresh(); T0 = r['type0']
U = k.reserve_inductive_type('U', T0)
rejects(lambda: k.finish_inductive(U, [CS('mk', [T0])]), "larger than the family's own universe")        # Girard-style: mk : Type0 -> U
print("PASS  U : Type0 with constructor field Type0 is rejected (paradox shape)")

k, r, CS = fresh(); T0 = r['type0']
Uf = k.reserve_inductive_type('Uf', T0)       # field is a large function type: Nat -> Type0 lives in Type1
rejects(lambda: k.finish_inductive(Uf, [CS('mk', [k.mk_pi(r['nat_ty'], T0)])]), "larger than the family's own universe")
print("PASS  field of type Nat -> Type0 in a Type0 inductive is rejected")

k, r, CS = fresh(); T0 = r['type0']
Bad = k.reserve_inductive_type('Bad', T0)     # field type is a TERM (Zero : Nat), not a type
rejects(lambda: k.finish_inductive(Bad, [CS('mk', [r['zero']])]), "not a type")
print("PASS  constructor field whose 'type' is a value (not a type) is rejected")

k, r, CS = fresh(); T0 = r['type0']
Ub = k.reserve_inductive_type('Ub', T0)
rejects(lambda: k.finish_inductive(Ub, [CS('mk', [k.mk_var(5)])]), "ill-typed")
print("PASS  unbound variable in a field type is rejected")

k, r, CS = fresh(); T0 = r['type0']
I = k.reserve_inductive_type('I', k.mk_pi(r['nat_ty'], T0))       # I : Nat -> Type0
rejects(lambda: k.finish_inductive(I, [CS('mk', [], [])]), "do not fully apply")
print("PASS  constructor that omits a required index is rejected")

k, r, CS = fresh(); T0 = r['type0']
I = k.reserve_inductive_type('I', k.mk_pi(r['nat_ty'], T0))
rejects(lambda: k.finish_inductive(I, [CS('mk', [], [r['zero'], r['zero']])]), "ill-typed")
print("PASS  constructor with too many indices is rejected")

k, r, CS = fresh(); T0 = r['type0']
I = k.reserve_inductive_type('I', k.mk_pi(r['nat_ty'], T0))
rejects(lambda: k.finish_inductive(I, [CS('mk', [], [T0])]), "ill-typed")
print("PASS  result index of the wrong type (Type0 where Nat expected) is rejected")

k, r, CS = fresh(); T0 = r['type0']
W = k.reserve_inductive_type('W', r['zero'])                         # declared 'sort' is a value, not a type
rejects(lambda: k.finish_inductive(W, [CS('mk', [])]), "declared type")
print("PASS  ill-typed declared family type is rejected")

k, r, CS = fresh(); T0 = r['type0']
Ng = k.reserve_inductive_type('Ng', T0)                              # strict positivity still enforced
rejects(lambda: k.finish_inductive(Ng, [CS('mk', [k.mk_pi(k.mk_const(Ng), r['nat_ty'])])]), "strict positivity")
print("PASS  strict positivity is still enforced")

# ---- positives ----
k, r, CS = fresh(); T0 = r['type0']
Box = k.reserve_inductive_type('Box', T0)       # field Pi A:Type0. A  is impredicative => lives in Type0: allowed
k.finish_inductive(Box, [CS('mk', [k.mk_pi(T0, k.mk_var(0))])])
print("PASS  small impredicative field (Pi A:Type0. A) is accepted")

k, r, CS = fresh(); T0 = r['type0']
Id2 = k.reserve_inductive_type('Id2', k.mk_pi(T0, T0))            # Id2 : Type0 -> Type0
k.finish_inductive(Id2, [CS('mk', [T0], [k.mk_var(0)])])         # field A is a bare result index: parameter-like, exempt
print("PASS  large parameter-like field (bare result index) is accepted")

k, r, CS = fresh(); T0 = r['type0']; nat = r['nat_ty']
succ = r['succ_c']
Vsort = k.mk_pi(T0, k.mk_pi(nat, T0))                              # V : Type0 -> Nat -> Type0
V = k.reserve_inductive_type('V', Vsort); Vc = k.mk_const(V)
vnil = CS('vnil', [T0], [k.mk_var(0), r['zero']])
vcons = CS('vcons', [T0, nat, k.mk_var(0), k.mk_app(k.mk_app(Vc, k.mk_var(0)), k.mk_var(1))],
           [k.mk_var(0), k.mk_app(succ, k.mk_var(1))])
k.finish_inductive(V, [vnil, vcons])
print("PASS  Vec-style indexed family (parameter + data fields + recursive field) is accepted")

k, r, CS = fresh(); T0 = r['type0']
# unsafe escape hatch still exists for deliberate experimentation (the flag skips admission entirely)
Uu = k.reserve_inductive_type('Uu', T0)
k.finish_inductive(Uu, [CS('mk', [T0])], True)
print("PASS  explicit unsafe flag still bypasses admission (documented, for experiments only)")
print("ALL ADMISSION TESTS PASSED")
