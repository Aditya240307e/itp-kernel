"""ml_lemmas.py -- a library that grows: lemmas the agent proved become premises that later episodes may use.

    lib = LemmaLibrary(bk, eng, ctx, forbid=forbidden_digests, held_out=names_of_dev_and_test_problems)
    rec, why = lib.add(candidate)            # candidate: a hindsight example (gym_tree.Example.to_dict() form); admitted -> a LemmaRecord, installed in this world; else (None, reason)
    lib.premises(task_premises)              # the premises of a task, widened by every installed lemma: what an episode on this world must be started with
    payload = lib.records_json(since=3)      # the records from the fourth on, as plain data (a worker process installs them with `lib.install_records(payload)`)
    lib.sync(nets=[net], runners=[runner])   # after installing: the global features of the world again, in the networks and in the policy runners

WHAT IS A LEMMA HERE.  A closed statement with a proof that the engine can replay from the LIBRARY ALONE (the globals below the first problem of the bank) and the lemmas admitted before it.  `add` replays the
candidate's portable actions on a fresh proof state under exactly those premises (nothing of the problem it came from is visible: its helper lemmas, its definitions), the kernel checks the proof term
against the statement, and the world declares the global `boot_NNNN` with that term as its value (`World.define`: the kernel's checked path again).  So a lemma is a theorem of the world like any other; the
independent checker (checker/recheck.py) reads it with the rest of the dependencies of a proof that uses it; nothing in this file is trusted.

WHAT IS NOT ADMITTED.  (1) a lemma whose source is not in the train split (the caller passes the split: a sub-theorem of a dev or test problem never becomes a premise); (2) a statement that is one the world
already has (a theorem of the library or of the bank, up to the renaming of variables: the node of the kernel is the same), one that was admitted, or one in the `forbid` set of printed-statement digests
(the caller puts the statements of the bank's problems there); (3) one that says the same as a HELD-OUT problem -- the same relation atoms on a grid of small values, under any order and naming of the
parameters, the test of bank_dups -- so that a held-out problem is never one `apply` of a lemma away; (4) a proof of fewer than `min_len` actions (a single lemma application or decision procedure is not a
lemma) or more than `max_len`; (5) a statement longer than `max_tokens`; (6) more than `max_lemmas` lemmas in all (the tokens of the policy's vocabulary stop at `ml_features.GMAX` globals).

PREMISES.  The lemmas are appended to the world after everything else, so they form one block of global ids; a task sees its own premises plus that block (`gym_premises.Premises` can say it: the ids below
the end of the block, less the ranges hidden -- the gap between the task's premises and the block).  The block grows with `add`; a task started before keeps the premises it was started with.

DETERMINISM.  Installing a list of records in the same order on the same world gives the same globals: a worker process that installs the learner's records ends with the learner's world (`block_digest`
says so).  A record that does not replay is skipped, deterministically, by every process.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import json, hashlib, collections, itertools
import gym_env as G, gym_acts as ACTS, gym_tree as GT, gym_premises as PM, gym_build as GB
import bank as B
from author import AuthorError
import itp_env_lnl as E

FORMAT = 1
SPLITS_ADMITTED = ('train',)


class LemmaRecord:
    __slots__ = ('id', 'name', 'code', 'steps', 'n_before', 'origin', 'source', 'length', 'digest', 'gid', 'status')

    def __init__(self, id, name, code, steps, n_before, origin, source, length, digest, gid=None, status='pending'):
        self.id = id; self.name = name; self.code = code; self.steps = steps; self.n_before = n_before; self.origin = origin; self.source = source; self.length = length; self.digest = digest
        self.gid = gid; self.status = status

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'statement_code': [list(t) for t in self.code], 'steps': self.steps, 'n_before': self.n_before, 'origin': self.origin, 'source': self.source,
                'length': self.length, 'statement_digest': self.digest}

    @classmethod
    def from_dict(cls, d):
        return cls(int(d['id']), str(d['name']), tuple(tuple(t) for t in d['statement_code']), d['steps'], int(d['n_before']), d.get('origin', ''), str(d.get('source', '')), int(d['length']),
                   str(d['statement_digest']))

    def __repr__(self): return 'LemmaRecord(%s, %d actions, %s)' % (self.name, self.length, self.status)


def statement_digest(w, stmt):
    return GT.digest(w.show(stmt))


def used_in(steps, lo, hi):
    """the ids of the lemmas of the block [lo, hi) that portable actions mention (as their source, or inside a code)"""
    out = set()
    def tok(t):
        if len(t) >= 2 and t[0] == 'c' and lo <= t[1] < hi: out.add(t[1])
    for st in steps or ():
        if not isinstance(st, dict): continue
        s = st.get('source')
        if isinstance(s, (list, tuple)) and s: tok(s)
        for code in st.get('codes') or ():
            for t in code: tok(t)
    return out


class LemmaLibrary:
    def __init__(self, bk, eng, ctx, forbid=(), held_out=(), max_lemmas=300, min_len=2, max_len=40, max_tokens=120, meaning=True, name_prefix='boot_'):
        import ml_features as F
        self.bk = bk; self.w = bk.w; self.k = bk.w.k; self.eng = eng; self.ctx = ctx
        self.library_end = self.w.gid[B.LIBRARY_LIMIT]
        self.lo = self.k.num_globals()                              # the block of lemmas begins here
        self.max_lemmas = min(int(max_lemmas), max(0, F.GMAX - self.lo))
        self.min_len = min_len; self.max_len = max_len; self.max_tokens = max_tokens; self.prefix = name_prefix
        self.forbid = set(forbid); self.records = []; self.n_installed = 0
        self.rejected = collections.Counter()
        self._types = None                                          # the node of the type of every theorem of the world (statements the world has)
        self.meaning = None
        self.held_out = tuple(held_out)
        self._use_meaning = meaning
        self._world_digest = {}                                     # digest of each installed record (by id)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ premises
    @property
    def hi(self): return self.lo + self.n_installed

    def premises(self, base, n=None):
        """`base` (None, an int, a Premises: what a task sees) widened by the installed lemmas [lo, lo + n); n defaults to all of them"""
        n = self.n_installed if n is None else n
        b = PM.of(base)
        if n <= 0 or b is None: return b
        if isinstance(b, int): upto0, hidden0 = b, []
        else: upto0, hidden0 = b.upto, list(b.hidden)
        if upto0 > self.lo: raise ValueError('the premises %r reach into the block of lemmas (it begins at %d)' % (b, self.lo))
        return PM.of(PM.Premises(self.lo + n, hidden0 + [(upto0, self.lo)]))

    def lemma_premises(self, n_before):
        """what the proof of a lemma may use: the library and the first `n_before` lemmas"""
        return self.premises(self.library_end, n_before) if n_before else self.library_end

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ admission
    def _world_types(self):
        if self._types is None:
            k = self.k; types = set()
            for gid in range(self.lo):
                _, ty, val, cf, ci, ctors, rec = k.global_info(gid)
                if val is None or k.is_inductive(gid): continue
                ok, sort = k.infer_open(ty, [])
                if ok and sort == k.mk_type(0): types.add(ty)
            self._types = types
        return self._types

    def _meaning_index(self):
        if self.meaning is None:
            import bank_dups as BD, semantic as SEM
            idx = []; sem = self.bk.semantics
            kinds = BD.readable(self.bk, [n for n in self.held_out if n in self.bk.placed])
            for n, ks in kinds.items():
                t = BD._tables(sem, self.bk.statement_ast(n), ks, tuple(range(len(ks))))
                if t is not None and t[1]: idx.append((n, ks, t))
            self.meaning = idx
        return self.meaning

    def same_as_held_out(self, stmt):
        """the name of a held-out problem that `stmt` says the same as (the test of bank_dups, layer 2), or None; statements the evaluator cannot read are not covered (only their digest is)"""
        if not self._use_meaning or not self.held_out: return None
        import bank_dups as BD, semantic as SEM
        sem = self.bk.semantics
        try:
            ast = sem.decode(stmt.n); kinds, _, _ = sem.parts(ast)
        except (SEM.Unsupported, Exception): return None
        if not (1 <= len(kinds) <= 4 and all(k in ('Z', 'N') for k in kinds)): return None
        for n, ks, t in self._meaning_index():
            if sorted(ks) != sorted(kinds): continue
            for perm in itertools.permutations(range(len(ks))):
                # the candidate's parameter j takes the value of the held-out problem's parameter perm[j] (bank_dups.meaning_pairs)
                if any(kinds[j] != ks[perm[j]] for j in range(len(kinds))): continue
                tc = BD._tables(sem, ast, ks, perm)
                if BD._same(t, tc): return n
        return None

    def add(self, cand, split='train', source=''):
        """admit a hindsight example (a dict as `gym_tree.Example.to_dict` makes it: `statement_code`, `steps`, `statement_digest`, `length`, `origin`) if it passes every rule of the module
        documentation, and install it in this world: (LemmaRecord, None) or (None, reason).  The reasons are counted in `rejected`."""
        def no(reason): self.rejected[reason] += 1; return None, reason
        if split not in SPLITS_ADMITTED: return no('not train')
        code = cand.get('statement_code'); steps = cand.get('steps')
        if code is None or not steps: return no('no code or no proof')
        length = len(steps)
        if length < self.min_len: return no('too short')
        if length > self.max_len: return no('too long')
        if len(code) > self.max_tokens: return no('statement too long')
        if len([r for r in self.records if r.status == 'ok']) >= self.max_lemmas: return no('library full')
        digest = cand.get('statement_digest')
        if digest is not None and (digest in self.forbid or digest in {r.digest for r in self.records}): return no('known statement')
        n0 = self.n_installed; prem = self.lemma_premises(n0)
        try: stmt = GT.statement_from_code(self.eng, code, prem)
        except (GB.BuildError, TypeError, ValueError, AuthorError, E.ITPError, RuntimeError, KeyError, IndexError): return no('not sayable with the library')
        digest = statement_digest(self.w, stmt)
        if digest in self.forbid or digest in {r.digest for r in self.records}: return no('known statement')
        if stmt.n in self._world_types(): return no('already a theorem of the world')
        if any(self.k.global_info(r.gid)[1] == stmt.n for r in self.records if r.status == 'ok'): return no('known statement')
        other = self.same_as_held_out(stmt)
        if other is not None: return no('says the same as a held-out problem')
        rec = LemmaRecord(len(self.records), '%s%04d' % (self.prefix, len(self.records)), tuple(tuple(t) for t in code), steps, n0, cand.get('origin', ''), source, length, digest)
        self.records.append(rec)
        if not self._install(rec, stmt=stmt):
            self.records.pop(); self.rejected['does not replay from the library'] += 1
            return None, 'does not replay from the library'
        self.eng.refresh()
        return rec, None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ installation
    def _install(self, rec, stmt=None):
        """replay the record's proof under the library and the lemmas before it, let the kernel check it, declare the global; False if it does not replay (the record is then 'failed' and nothing is declared)"""
        prem = self.lemma_premises(rec.n_before)
        try:
            if stmt is None: stmt = GT.statement_from_code(self.eng, rec.code, prem)
            term = ACTS.replay_portable(self.eng, stmt, rec.steps, prem)                          # the kernel has checked `term` against `stmt`
            self.w.define(rec.name, stmt, term)                                                   # ... and checks it again, as the value of a declared constant
        except (G.ActionError, GB.BuildError, AuthorError, E.ITPError, RuntimeError, TypeError, ValueError, KeyError, IndexError):
            rec.status = 'failed'; return False
        rec.gid = self.w.gid[rec.name]; rec.status = 'ok'; self.n_installed += 1
        return True

    def install_records(self, payload):
        """install records made by another process (a list of the dicts of `records_json`), those this world does not have yet, in order; returns the number installed.  Does not refresh the engine: `sync` does"""
        n = 0
        for d in payload:
            rec = LemmaRecord.from_dict(d)
            if rec.id < len(self.records): continue
            if rec.id != len(self.records): raise ValueError('lemma %d arrives before lemma %d' % (rec.id, len(self.records)))
            self.records.append(rec)
            if self._install(rec): n += 1
            else: self.rejected['does not replay from the library'] += 1
        return n

    def records_json(self, since=0):
        return [r.to_dict() for r in self.records[since:]]

    def sync(self, nets=(), runners=()):
        """after lemmas were installed: the engine's indexes, the global features of the world, the networks' tables, the policy runners' caches"""
        self.eng.refresh()
        bags = self.ctx.refresh_globals()
        for net in nets: net.set_globals(bags)
        for r in runners: r.refresh()
        return bags

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------ bookkeeping
    def block_digest(self):
        """a hash of the installed lemmas (names, statements, in the order of their ids): two worlds with the same digest have the same block"""
        h = hashlib.sha256()
        for r in self.records:
            if r.status == 'ok': h.update(('%s|%s|%d;' % (r.name, r.digest, r.gid)).encode())
        return h.hexdigest()[:16]

    def ok_records(self): return [r for r in self.records if r.status == 'ok']

    def uses(self, steps):
        """the ids (globals) of the installed lemmas that portable actions use"""
        return used_in(steps, self.lo, self.hi)

    def save(self, path, extra=None):
        d = {'format': FORMAT, 'library_end': self.library_end, 'base': self.lo, 'block_digest': self.block_digest(), 'records': self.records_json(), 'rejected': dict(self.rejected)}
        if extra: d.update(extra)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f: json.dump(d, f)
        _os.replace(tmp, path)

    def load(self, path):
        """install the lemmas of a file made by `save` (every one is replayed and checked by the kernel: a file cannot make the world believe a statement); returns the number installed"""
        d = json.load(open(path))
        if d.get('format') != FORMAT: raise ValueError('%s: not a lemma file of format %d' % (path, FORMAT))
        if d.get('library_end') != self.library_end: raise ValueError('%s was made over a library of %s globals, this world has %d' % (path, d.get('library_end'), self.library_end))
        n = self.install_records(d['records'])
        self.eng.refresh()
        return n

    def summary(self):
        ok = self.ok_records()
        return {'installed': len(ok), 'failed': sum(1 for r in self.records if r.status != 'ok'), 'rejected': dict(self.rejected), 'mean_length': (sum(r.length for r in ok) / len(ok)) if ok else 0.0,
                'origins': dict(collections.Counter((r.origin.get('kind') if isinstance(r.origin, dict) else str(r.origin)) for r in ok)), 'block_digest': self.block_digest()}
