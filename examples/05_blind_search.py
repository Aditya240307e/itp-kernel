"""05 -- blind proof search: iterative deepening over the engine's menus, no learning.

    python3 examples/05_blind_search.py                       (about 90 s)

The plain search that ships with the environment (src/python/environment/gym_search.py).  Goals are the nodes of an AND-OR graph, an action is an edge to the goals it leaves, and a goal is solved
when some action solves all of them.  The order in which actions are tried comes from `default_rank` (cheap closing actions first); any other order can be passed as `rank`.
What the search returns is a TREE of actions; `commit` replays it on a live proof and the kernel checks the result, so "solved" always means a kernel-checked proof.

  1. one problem, in detail:  the tree the search finds;
  2. a handful of problems:   how many probes (tried actions) each needs, in the strict regime (the lemmas written for the problem are hidden).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import bank_all, bank_baseline
import gym_env as G, gym_search as S

bank = bank_all.load()
engine = G.Engine(bank.w)

# ---------------------------------------------------------------------------------------------------------------------------------------- 1. one problem
name = 'quad_form_zero'
placed = bank.placed[name]
engine.premises = placed.premises('strict')
proof, tree, searcher = S.prove_statement(engine, placed.stmt, max_depth=6, max_probes=4000)
print('%s: %s' % (name, placed.problem.text))
print('solved: %s   probes: %d   tree: %d actions, depth %d, kinds %s' % (proof is not None, searcher.probes if searcher else 0, tree.size(), tree.depth(), [G.Kind(k).name if isinstance(k, int) else k for k in tree.kinds()]))
assert proof is not None                               # (`prove_statement` has already had the kernel check the proof against the statement)

# ---------------------------------------------------------------------------------------------------------------------------------------- 2. a handful of problems
names = ['ident_diff_squares', 'ident_sophie_germain', 'ineq_am_gm_two', 'coprime_consecutive', 'two_pow_gt_n']
rows = bank_baseline.baseline(bank, names, regimes=('strict',), probes=1500, time_limit=15.0, engine=engine)
print('\nblind search, strict regime, 1500 probes (what plain search cannot do: an inequality that needs the hint (a-b)^2, an induction, a witness):')
print(bank_baseline.table(rows))

print('EXAMPLE OK')
