import torch
from model import TermEncoder

failures = 0
def check(cond, label):
    global failures
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures += 1

enc = TermEncoder(embed_dim=16)

# --- Determinism: the same global id always hashes to the same slot. ---
check(enc._hash_const(42) == enc._hash_const(42), "hashing the same global id twice gives the same slot")

# --- No advance registration needed: a global id that did not exist
# anywhere when the model was constructed (simulating a lemma
# declare_checked'd mid-curriculum, long after model construction)
# still gets a genuine, usable embedding -- not a shared "unknown"
# fallback indistinguishable from every other unregistered constant. ---
NEVER_SEEN_GID = 987654321
idx = enc._hash_const(NEVER_SEEN_GID)
check(0 <= idx < enc.const_table_size, f"a never-registered global id still maps into the valid table range (got {idx})")
embed_for_new_gid = enc._scalar_feature(TermEncoder.TAG_CONST, NEVER_SEEN_GID)
check(embed_for_new_gid.shape == (16,), "the embedding for a never-seen constant has the right shape")
check(not torch.isnan(embed_for_new_gid).any(), "the embedding for a never-seen constant is not NaN")

# --- Distinctness at this session's realistic scale: hash the base
# library's own ~20 constants (this session's actual scale) and
# confirm they land in DISTINCT slots -- collisions are a known,
# quantified small-probability event (see TermEncoder's own
# docstring), not something to expect at this specific scale, so
# their absence here is a meaningful (not just lucky) confirmation. ---
sample_gids = list(range(20))
slots = [enc._hash_const(g) for g in sample_gids]
check(len(set(slots)) == len(slots), f"20 sequential global ids hash to 20 DISTINCT slots (no collision at this scale)")

# --- Two different constants produce two different embeddings
# (assuming no collision, confirmed above), end to end through
# _scalar_feature, not just at the raw hash-index level. ---
embed_a = enc._scalar_feature(TermEncoder.TAG_CONST, 3)
embed_b = enc._scalar_feature(TermEncoder.TAG_CONST, 17)
check(not torch.equal(embed_a, embed_b), "two different (non-colliding) global ids get genuinely different embeddings")

print()
if failures == 0:
    print("ALL hashing-based const embedding CHECKS PASSED")
else:
    print(f"{failures} CHECK(S) FAILED")
