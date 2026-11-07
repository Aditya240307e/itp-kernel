import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths
import collections
import nat_lib, gym_env as G, gym_search as S

w = nat_lib.build_nat()
Nat, Zero, Succ, Eq, forall = w['Nat'], w['Zero'], w['Succ'], w.Eq, w.forall_
w.recursive('twice', w.arrow(Nat, Nat),
            lambda f: w.fun([Nat], lambda n: w.match(n, lambda _: Nat, [Zero, lambda k: Succ(Succ(f(k)))])))
twice = w['twice']
statement = forall([Nat], lambda n: Eq(twice(n), n + n))

eng = G.Engine(w, kinds=set(G.Kind))                      # an engine that offers every kind of action
ep = G.Episode(eng, statement, premises=w.k.num_globals())  # one proof attempt; the agent may use every global so far
print(eng.render(ep.goal()))

menu = ep.menu()                                           # the legal actions for the first open goal
print(len(menu.actions), 'actions:', dict(collections.Counter(G.Kind(a.kind).name for a in menu.actions)))

try:                                                       # an action that does not apply is refused and changes nothing
    ep.step(G.Action(G.Kind.RING, 0))
except G.ActionError as e:
    print('refused:', e, '| goals open:', ep.n_goals, '| steps taken:', len(ep.steps))

induct = next(a for a in menu.actions if a.kind == G.Kind.INDUCT)
ep.step(induct)                                            # probed first, then taken; the base case is closed by computation
print(eng.render(ep.goal()))

proof, tree, searcher = S.prove_statement(eng, statement, max_depth=6, max_probes=2000)   # blind iterative deepening over the menus
print('search:', 'found a proof' if proof is not None else 'no proof', '| probes used:', searcher.probes)
