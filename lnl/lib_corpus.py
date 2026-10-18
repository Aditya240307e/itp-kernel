"""lib_corpus.py -- the proofs of the LIBRARY as training data: every theorem of the base library (`imo_lib.build_imo`, the lemmas the bank's problems stand on) with its proof replayed as a sequence of
kernel-checked actions.

The bank is small (a few dozen problems a stage); the library is where the learner finds the examples of how a proof goes: ~400 lemmas about equality, naturals, binary numbers, integers (`Z`), divisibility,
gcd, primes, induction principles, ..., 2000 actions.  Each lemma is a problem like the bank's (a statement, the globals before it as premises) and has a reference demonstration; the same replay
machinery produces both (`recorder.replay`), so the actions have one format.

    corpus = lib_corpus.build()                        # {'format', 'globals', 'world_digest', 'lemmas': [{'name', 'gid', 'statement', 'steps', 'length', 'split'}, ..], 'unreplayed': [..]}
    lib_corpus.write(corpus, 'library.json')

`split` is `bank_split.lemma_split(name)`: a function of the lemma's name, the same for every version of the library.  A lemma whose statement is the statement of a problem of the bank is marked `bank_duplicate`
(a learner that trains on lemmas must not be handed the answer to a problem it is tested on).
"""
import os, sys, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itp_env_lnl as E
import bank as B, bank_lib, bank_split, recorder as REC
import gym_env as G


def build(progress=None, names=None, bank_digests=()):
    """replay the proofs of the base library: the corpus as a dict (see the module documentation). `names`: only these lemmas; `bank_digests`: statement digests of the bank's problems"""
    t0 = time.time()
    w, recs = REC.record_world(bank_lib.base)
    eng = G.Engine(w, kinds=set(G.Kind), strict=True)
    lemmas = []; bad = []
    for name, rec in recs.items():
        if names is not None and name not in names: continue
        r = REC.replay(eng, rec)
        if not r.ok:
            bad.append({'name': name, 'gid': rec.gid, 'gap': str(r.gap)[:300]}); continue
        dig = B.statement_digest(w, rec.stmt)
        lemmas.append({'name': name, 'gid': rec.gid, 'statement': w.show(rec.stmt), 'statement_digest': dig, 'steps': r.portable, 'length': len(r.portable),
                       'split': bank_split.lemma_split(name), 'bank_duplicate': dig in bank_digests})
        if progress: progress(len(lemmas), len(recs), name)
    return {'format': 1, 'library': 'imo_lib.build_imo', 'globals': w.k.num_globals(), 'world_digest': B.world_digest(w, w.k.num_globals()),
            'seconds': round(time.time() - t0, 1), 'lemmas': lemmas, 'unreplayed': bad}


def write(corpus, path):
    with open(path, 'w') as f: json.dump(corpus, f, indent=0)


def digest(corpus):
    """a hash of what the corpus says (not of the time it took): the names, statements and steps"""
    h = hashlib.sha256()
    for l in corpus['lemmas']: h.update(json.dumps([l['name'], l['gid'], l['statement'], l['steps']], sort_keys=True).encode())
    return h.hexdigest()[:16]


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'library.json'
    c = build(progress=lambda i, n, nm: print('\r%d/%d %s' % (i, n, nm[:40]), end='', flush=True) if i % 25 == 0 else None)
    write(c, out)
    print('\n%d lemmas, %d actions, %d not replayed, %.0f s, digest %s' % (len(c['lemmas']), sum(l['length'] for l in c['lemmas']), len(c['unreplayed']), c['seconds'], digest(c)))
