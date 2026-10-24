"""synth_card.py -- the dataset card of the synthetic theorem dataset, written from its manifest (and its verification report)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401


def _table(rows, header):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(' --- ' for _ in header) + '|']
    for r in rows: out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def _dist(d, total, keyfmt=str):
    return [(keyfmt(k), v, '%.1f%%' % (100 * v / total)) for k, v in d.items()]


def card(man, ver=None):
    st = man['statistics']; n = st['n']; cfg = man['config']
    L = []
    L.append('# Synthetic theorem dataset (format %d)\n' % man['format'])
    L.append('%d kernel-checked theorems about integers (`Z`): statements with hypotheses, each with a machine-checked proof, generated from the lemma library by growing proof trees at random. '
             'They are **synthetic**: true by construction, varied in depth and in the lemmas they need, and not meant to be interesting mathematics -- they are the practice ground for a prover that is later '
             'trained and tested on olympiad problems (the problem bank).\n' % man['records'])
    L.append('## How it is made\n')
    L.append('1. A *root lemma* of the pool is drawn (an area first -- order, equality, divisibility, gcd, prime, existence, logic -- then a lemma of the area, so that the large areas do not crowd out the others). A '
             '**witness model** (integers for the variables) is drawn and arguments for the lemma are drawn until its premises hold at the model; the root goal is the lemma\'s conclusion at these arguments.')
    L.append('2. The proof tree is **grown backwards**: the engine applies library lemmas to open goals (the same unifier an agent uses; every step is validated by a probe), along a main line of the planned '
             'depth and a few side branches. Every goal of the tree is kept true at the witness model. A goal that fails somewhere cannot be proved without hypotheses, so it is **lifted**: it becomes a hypothesis '
             'of the theorem; a goal that holds everywhere is closed by a premise-free lemma or a decision procedure (ring, linarith, nlinarith).')
    L.append('3. The theorem is `forall variables, leaf_1 -> ... -> leaf_m -> root`, and the proof is the tree\'s steps in the order the engine meets the goals. Statements are required to be **tidy** (no `x + 0`, '
             '`x - x`, double negations, cancelling terms, division by the literal zero), small, with hypotheses that are satisfiable, not always true, and (where the evaluator can tell) each needed.')
    L.append('4. Every theorem is **replayed from its data** twice in the generating process (proof term checked by the kernel), and again by `synth_verify` in a fresh process and fresh world, '
             'with the independent checker\'s second opinion.\n')
    L.append('The index is the identity: theorem *i* of seed *s* is made from `random.Random("s/i")` alone, so any record can be regenerated and a shard can be remade in isolation.\n')
    L.append('## Files\n')
    L.append('`train.jsonl.gz`, `dev.jsonl.gz`, `test_iid.jsonl.gz`, `test_comp.jsonl.gz` (one JSON record per line), `manifest.json` (configuration, digests, ledger, statistics), `config.json`, `verification.json`.\n')
    L.append('A record: `id`, `index`, `seed`, `family`, `domain`; `statement` (printed by the library\'s printer; its digest is `digest`), `pretty` (numerals in decimal, for reading), `code` (the statement in the tokens '
             'of the typed term builder: what a policy starts an episode from); `steps` (the proof: portable actions, `gym_acts.replay_portable`); labels: `root_lemma`, `area`, `lemmas` (applied, in proof order), '
             '`depth` (longest chain of steps of the tree), `n_steps`, `n_lemmas`, `n_hyps`, `n_vars`, `closers`, `size` (tokens), `solved_by` (decision procedures that close the statement at once), '
             '`semantic` (satisfiability and the evaluator\'s findings), `model` (the witness model), `recipe` (hash of the set of lemmas and closers of the proof), `shape` (the statement without numerals and names), '
             '`skeleton` (the proof without terms: kinds and lemmas of its steps) and `split`.\n')
    L.append('## Splits\n')
    sr = man['split_rule']
    L.append('`test_comp`: records whose **recipe** (set of lemmas and decision procedures the proof uses) falls in the held-out %.1f%% of recipes -- no training record has that recipe, though its lemmas are '
             'seen individually: the test of putting known pieces together in a new way. `test_iid`: %.1f%% of the statements by digest; `dev`: %.1f%%; the rest is `train`. Salt `%s`.\n'
             % (sr['permille']['test_comp'] / 10, sr['permille']['test_iid'] / 10, sr['permille']['dev'] / 10, sr['salt']))
    L.append(_table([(s, man['counts'][s], '%.1f%%' % (100 * man['counts'][s] / max(1, man['records']))) for s in ('train', 'dev', 'test_iid', 'test_comp')], ('split', 'theorems', 'share')))
    L.append('\n## What is in it\n')
    L.append('Root-lemma areas:\n')
    L.append(_table(_dist(dict(sorted(st['area'].items(), key=lambda kv: -kv[1])), n), ('area', 'theorems', 'share')))
    L.append('\nDepth (longest chain of steps of the proof tree), hypotheses, steps, variables:\n')
    keys = sorted({int(k) for k in st['depth']} | {int(k) for k in st['n_hyps']} | {int(k) for k in st['n_steps']} | {int(k) for k in st['n_vars']})
    L.append(_table([(k, st['depth'].get(str(k), 0), st['n_hyps'].get(str(k), 0), st['n_steps'].get(str(k), 0), st['n_vars'].get(str(k), 0)) for k in keys], ('value', 'depth', 'hypotheses', 'steps', 'variables')))
    L.append('\n* statement size (tokens): mean %s, median %s, 90th percentile %s, max %s' % (st['size']['mean'], st['size']['p50'], st['size']['p90'], st['size']['max']))
    L.append('* a decision procedure (ring / linarith / nlinarith) **closes the statement at once** for %.1f%% of the theorems (these are kept at a quota: the rest need search); the proof uses a decision procedure somewhere for %.1f%%'
             % (100 * st['closed_at_once_by_a_decision_procedure'], 100 * st['uses_a_decision_procedure_in_the_proof']))
    L.append('* existential conclusions: %.1f%% (their conclusion is not evaluated by the semantic layer: the witnesses may lie outside its window)' % (100 * st['existential_conclusion']))
    L.append('* %d distinct lemmas applied (of the pool), %d distinct root lemmas, %d distinct recipes, %d distinct proof skeletons, %d distinct statement shapes (at most %d records per shape and %d per skeleton)\n'
             % (st['lemmas_used'], st['root_lemmas'], st['distinct_recipes'], st['distinct_skeletons'], st['distinct_shapes'], man['split_rule']['shape_cap'], man['split_rule']['skeleton_cap']))
    top = sorted(st['lemma_usage'].items(), key=lambda kv: (-kv[1], kv[0]))                 # (the manifest is written with sorted keys)
    L.append('Most used lemmas: ' + ', '.join('`%s` (%d)' % kv for kv in top[:12]) + '. Least used: ' + ', '.join('`%s` (%d)' % kv for kv in top[-8:]) + '.\n')
    L.append('## Ledger\n')
    L.append('%d indices were tried (seed %d, from the first index); %d produced a record; %d survived the merge. Why the others did not:\n' % (man['attempts'], cfg['seed'], man['records_before_merge'], man['records']))
    rj = dict(sorted(man['rejects'].items(), key=lambda kv: -kv[1]))
    L.append(_table([(k, v, '%.1f%%' % (100 * v / max(1, man['attempts']))) for k, v in rj.items()], ('rejected at generation', 'indices', 'share')))
    L.append('\n' + _table([(k, v) for k, v in man['dropped_in_merge'].items()], ('dropped in the merge', 'records')))
    L.append('\nGeneration errors (bugs of the generator, always counted): %d. Disagreements between the semantic evaluator and the kernel: %d. Generation time: %d s (single process equivalent).\n' % (man['errors'], man['disagreements'], man['generation_seconds']))
    if ver:
        L.append('## Verification\n')
        L.append('`synth_verify` rebuilt the library in a fresh process and, for **%d of %d records**: said the statement again from its code and compared it with the printed one, replayed the proof from its steps '
                 '(kernel-checked), checked the labels that can be read off the data (steps, hypotheses, variables, applied lemmas, closers)%s. Failures: %d%s.\n'
                 % (ver['checked'], ver['records'], ', and had the independent checker accept the proof (%d accepted, %d refused)' % (ver['second_opinion_accepted'], ver['n_refused']) if ver['second_opinion'] else '',
                    ver['n_failed'], '' if ver['ok'] else ' (see verification.json)'))
    L.append('## Provenance\n')
    L.append('World digest `%s` (the library below `%s`); generator `%s`; configuration `%s`.\n' % (man['world_digest'], man['limit'], man['generator'], man['config_digest']))
    L.append('## Limitations (read before using)\n')
    L.append('* The theorems are random combinations of library lemmas: their hypotheses and conclusions are true and proved, but often not what a person would ask. A prover that does well here has learned lemma application, '
             'unification and the decision procedures; it has not necessarily learned mathematics.')
    L.append('* All variables are integers; the vocabulary is the library\'s (order, equality, divisibility, gcd, primes, existence). Nothing here has the structure of a real olympiad problem (no sums, no recursion, no case analysis '
             'beyond what lemmas give) -- that is what the bank is for.')
    L.append('* The semantic layer (satisfiability, "needed" hypotheses) works on a box of small integers: it can reject a good theorem and cannot prove anything; the kernel proofs are the only claim.')
    L.append('* Depth is the depth of the tree that produced the proof, not a measure of difficulty: a deep tree of order lemmas is a transitivity chain.\n')
    return '\n'.join(L)
