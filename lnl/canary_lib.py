"""canary_lib.py -- the random-proof soundness canary shared by the gates that add something to the trusted base (S15 classical axioms, S16 quotients).

Idea: a blind random tactic generator throws proof attempts at statements that are FALSE in the intended model. If the trusted base were inconsistent
(or a rule unsound), some attempt would eventually complete AND verify. A verified proof of a false statement is a soundness bug, so the canary must
return none. Because "no proof found" is also what a toothless generator produces, every canary is paired with a POWER run of the same generator on
TRUE statements: it must verify a minimum number of proofs, and every completed proof state must verify (completed == verified)."""
import random
from harness import build_new, l2h, parse_raw


def run(k, g, statements, consts, ind_gids, N, seed, stop_on_verified=True):
    """statements: {name: level tree}. consts: global ids the generator may `apply`. ind_gids: families the generator may `cases` / `induct`.
    Returns (tried, completed, verified, first_verified_name). With stop_on_verified the run ends at the first verified proof (the FALSE use)."""
    rng = random.Random(seed)
    head = lambda x: head(x[1]) if x[0] == 'app' else x
    tried = completed = verified = 0
    first = None
    for name, st in statements.items():
        stmt = build_new(k, l2h(st, 0))
        for _ in range(N):
            ps = k.start_proof(stmt); tried += 1
            for _step in range(9):
                if ps.num_goals() == 0: break
                nh = ps.goal_num_hyps(0); tgt = parse_raw(k.dump_raw(ps.goal_target(0)), True)
                hyps = [parse_raw(k.dump_raw(ps.goal_hyp_type(0, j)), True) for j in range(nh)]
                indh = [j for j, h in enumerate(hyps) if head(h)[0] == 'const' and head(h)[1] in ind_gids]; r = rng.random()
                try:
                    if tgt[0] == 'pi' and r < 0.4: ps.intro('x')
                    elif indh and r < 0.65:
                        j = rng.choice(indh)
                        if rng.random() < 0.6: ps.cases(j)
                        else: ps.induct(j, eq_gid=g['Eq'], refl_gid=g['refl'])
                    elif nh and r < 0.82: ps.apply(build_new(k, ('fvar', rng.randrange(nh))))
                    else: ps.apply(build_new(k, ('const', rng.choice(consts))))
                except Exception: ps = None; break
            if ps is not None and ps.num_goals() == 0 and ps.is_complete():
                completed += 1
                if k.check_open(ps.extract(), stmt, [])[0]:
                    verified += 1
                    if first is None: first = name
                    if stop_on_verified: return tried, completed, verified, first
    return tried, completed, verified, first
