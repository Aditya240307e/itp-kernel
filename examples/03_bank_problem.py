"""03 -- a problem of the bank: what it says, whether the formalisation says that, and its recorded proof.

    python3 examples/03_bank_problem.py [problem_name]        (about a minute: the bank's world of ~580 globals is built from scripts; default problem: imo1959_p1)

The bank (src/python/problems/) holds formal olympiad problems.  Each one is a theorem statement in a library world, together with

  * its text in words and its source (`IMO 1959 P1`), topics and level;
  * its intended meaning as a Python predicate (`semantic.Spec`): the formal statement is READ BACK into a syntax tree, evaluated on a box of integers and compared with the predicate, and every mutated
    statement (one plausible slip each) must differ from it -- so a mis-formalised problem cannot enter unnoticed;
  * a reference proof written as a sequence of ACTIONS of the gym's action language, checked by the kernel (data/dataset_v1/bank.json holds them in a portable, JSON form).

Below: the statement, the faithfulness check, the replay of the recorded proof by the kernel, and the verdict of the independent checker on the resulting proof term.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import json, os, sys
import bank as B, bank_all
import gym_env as G, gym_acts as A
import recheck

NAME = sys.argv[1] if len(sys.argv) > 1 else 'imo1959_p1'
bank = bank_all.load()
w = bank.w
if NAME not in bank.placed: raise SystemExit('no such problem; the bank has %s ...' % ', '.join(list(bank.placed)[:6]))
placed = bank.placed[NAME]; problem = placed.problem
print('bank: %d problems in a world of %d globals' % (len(bank.placed), bank.k.num_globals()))
print('\n%s  (%s; level %d, %s; topics: %s)' % (NAME, problem.source, problem.level, B.LEVELS[problem.level], ', '.join(problem.topics)))
print('  in words:   ', problem.text)
print('  formalised: ', bank.statement_text(NAME)[:230] + (' ...' if len(bank.statement_text(NAME)) > 230 else ''))

# ---------------------------------------------------------------------------------------------------------------------------------------- is it the problem that was meant?
report = bank.check(NAME, demo=False)                    # the statement against its Python specification, and "can an agent say this statement with the term builder?"
sem = report.semantics
if sem is not None:
    print('\nfaithfulness: the statement agrees with its Python specification at %d points; %d of %d mutated statements are told apart from it (the rest are equivalent, by the spec\'s own account)'
          % (sem.points, sem.caught, sem.mutants))
print('the statement is sayable as a typed term of %d tokens: %s' % (len(report.code or ()), 'yes' if report.code else 'no'))
assert report.ok, report.errors

# ---------------------------------------------------------------------------------------------------------------------------------------- the recorded proof, replayed
data = json.load(open(os.path.join(itp_paths.DATA, 'dataset_v1', 'bank.json')))
record = next(p for p in data['problems'] if p['name'] == NAME)
steps = record['demonstration']['steps']
print('\nthe recorded proof: %d actions   (split: %s, family: %s)' % (len(steps), record['split'], record['family']))
for i, s in enumerate(steps):
    if 'source' in s:
        tag, ident, _ = s['source']
        what = (w.gname.get(ident) if tag == 'c' else 'hypothesis %d' % ident) + (' (right to left)' if s.get('rev') else '')
    else:
        what = json.dumps({k: v for k, v in s.items() if k != 'kind'})[:96]
    print('  %2d. %-9s %s' % (i + 1, s['kind'], what))

if any('codes' in s for s in steps): print('      (an argument is SAID as a code of the typed term builder, gym_build.py: ["n", -2] is the numeral -2)')
engine = G.Engine(w, kinds=set(G.Kind), strict=True)
proof = A.replay_portable(engine, placed.stmt, steps, placed.premises('prefix'))        # raises unless every action applies and the kernel accepts the finished term
print('\nthe kernel accepted the proof term (%d characters when printed raw)' % len(w.k.dump_raw(proof.n)))
print('premises: %d globals in the prefix regime (the library before the problem), %d in the strict regime (without the lemmas written for it)'
      % (placed.premises('prefix'), placed.premises('strict')))

# ---------------------------------------------------------------------------------------------------------------------------------------- the second opinion
ok, message = recheck.SecondOpinion(bank.k).accepts(proof, placed.stmt)
print('independent checker on the same term:', 'accepts' if ok else 'REJECTS (%s)' % message)
assert ok
print('EXAMPLE OK')
