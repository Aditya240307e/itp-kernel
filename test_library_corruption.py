import itp_env
import copy

failures = 0
def check(cond, label):
    global failures
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

kA = itp_env.Kernel()
type0 = kA.mk_type(0)
prop0 = kA.mk_prop()

nat_id = kA.reserve_inductive_type('Nat', type0)
nat_ty = kA.mk_const(nat_id)
nat_h = kA.finish_inductive(nat_id, [itp_env.CtorSpec('Zero', []), itp_env.CtorSpec('Succ', [nat_ty])])
zero = kA.mk_const(nat_h.ctor_global_ids[0])
succ_c = kA.mk_const(nat_h.ctor_global_ids[1])

eq_sort = kA.mk_pi(type0, kA.mk_pi(kA.mk_var(0), kA.mk_pi(kA.mk_var(0), prop0)))
eq_id = kA.reserve_inductive_type('Eq', eq_sort)
eq_ty = kA.mk_const(eq_id)
eq_h = kA.finish_inductive(eq_id, [itp_env.CtorSpec('refl', [type0, kA.mk_var(0)], [kA.mk_var(0), kA.mk_var(1), kA.mk_var(1)])])
refl_c = kA.mk_const(eq_h.ctor_global_ids[0])

one = kA.mk_app(succ_c, zero)
two = kA.mk_app(succ_c, one)

# A valid lemma: one_eq_one : Eq Nat 1 1
lemma_type = kA.mk_app(kA.mk_app(kA.mk_app(eq_ty, nat_ty), one), one)
lemma_value = kA.mk_app(kA.mk_app(refl_c, nat_ty), one)
kA.declare_checked(lemma_type, lemma_value, 'one_eq_one')

events = kA.export_library()

# --- Corruption 1: tamper with the lemma's own recorded VALUE so it no
# longer matches its own recorded TYPE (swap the refl's own argument
# from 1 to 2, so it "proves" Eq Nat 2 2 while the type still claims
# Eq Nat 1 1) -- a type-correctness violation. ---
tampered1 = copy.deepcopy(events)
for e in tampered1:
    if e.get('name') == 'one_eq_one':
        # e['value'] is the serialized refl application: app(app(refl, Nat), one)
        # -- replace the innermost 'one' argument with a serialized 'two'.
        two_serialized = e['value']['arg']  # reuse the ALREADY-serialized 'one' node's own SHAPE as a template... no, need genuinely different content
        break

# Simpler, more direct corruption: replace the const name the value
# references entirely, pointing 'refl' at a name that will not exist
# yet at reconstruction time -- a clean, unambiguous corruption.
tampered2 = copy.deepcopy(events)
for e in tampered2:
    if e.get('name') == 'one_eq_one':
        e['value']['fn']['fn']['name'] = 'nonexistent_lemma_reference'
        break

kB = itp_env.Kernel()
try:
    kB.import_library(tampered2)
    check(False, "import correctly rejects a Const reference to a nonexistent name")
except itp_env.ITPError as ex:
    check('nonexistent_lemma_reference' in str(ex), "import correctly rejects a Const reference to a nonexistent name")

# --- Corruption 2: tamper with the lemma's own TYPE (not value), so
# value and type genuinely disagree -- value still proves Eq Nat 1 1,
# but the claimed type is changed to Eq Nat 1 2 (structurally valid,
# resolves cleanly, but is simply FALSE / does not type-check). ---
tampered3 = copy.deepcopy(events)
for e in tampered3:
    if e.get('name') == 'one_eq_one':
        # e['type'] is app(app(app(Eq, Nat), one), one) -- change the
        # SECOND 'one' argument to reference Succ(one) = two instead.
        # e['type']['arg'] is the outermost App's own second argument
        # (the second 'one'); wrap it in an extra Succ application.
        original_arg = e['type']['arg']
        e['type']['arg'] = {'t': 'app', 'fn': {'t': 'const', 'name': 'Succ'}, 'arg': original_arg}
        break

kB2 = itp_env.Kernel()
try:
    kB2.import_library(tampered3)
    check(False, "import correctly rejects a value that no longer type-checks against its own (tampered) type")
except itp_env.ITPError as ex:
    check(True, "import correctly rejects a value that no longer type-checks against its own (tampered) type")
    print(f"  (rejected with: {str(ex)[:80]})")

# --- Control: confirm the UNMODIFIED export still imports successfully
# (so we know the corruption tests above are genuinely testing
# rejection, not just a broken import path in general). ---
kB3 = itp_env.Kernel()
kB3.import_library(events)
check(kB3.library_size() == kA.library_size(), "control: the UNMODIFIED export still imports successfully")

if failures == 0:
    print("\nALL LIBRARY CORRUPTION-REJECTION CHECKS PASSED")
else:
    print(f"\n{failures} CHECK(S) FAILED")
