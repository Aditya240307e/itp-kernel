"""synth_terms.py -- random, small, tidy terms in the context of a goal, said in the typed term builder's language.

    tg = TermGen(eng, vocab)                    # eng: a gym_env.Engine;  vocab: the operators and numerals the terms are made of (`Vocab`)
    code, node = tg.term(g, ty, rng, ops=2)     # a random term of type `ty` in the context of goal g, with at most `ops` operators: (its code, its kernel handle)

A generator that draws terms from the whole library would produce statements nobody would write. The vocabulary is the one an olympiad statement uses: the variables of the context, a few small numerals and
the ring operations. Terms are built one token at a time by the same builder an agent says its terms with (gym_build.py), so everything that is generated can also be SAID by a policy, and every term
is checked by the kernel when it is built.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import itp_paths  # noqa: E402,F401
import gym_build as GB
import synth_poly as SPY


class Vocab:
    """the operators (by name, with their arities) and the numerals of a domain; `role` says what an operator is (add, mul, sub, neg) so that the generator can keep terms tidy"""
    ROLES = {'Z.add': 'add', 'Z.mul': 'mul', 'Z.sub': 'sub', 'Z.neg': 'neg', 'N.add': 'add', 'N.mul': 'mul', 'add': 'add', 'mul': 'mul'}
    def __init__(self, w, type_name, ops, numerals=(0, 1, 2, 3, 4, 5)):
        self.type_name = type_name; self.ty = w[type_name].n
        self.ops = {w.gid[n]: a for n, a in ops.items() if n in w.gid}
        self.role = {w.gid[n]: self.ROLES[n] for n in ops if n in w.gid}
        self.gid_of = {r: g for g, r in self.role.items()}
        self.numerals = tuple(numerals)

    def parse(self, code):
        """the prefix code of a term as a tree  ('v', level) | ('n', value) | (role, child, ..)  (None if the code is not a well-formed term of this vocabulary)"""
        pos = [0]
        def go():
            if pos[0] >= len(code): raise ValueError
            t = code[pos[0]]; pos[0] += 1
            if t[0] == 'v': return ('v', t[1])
            if t[0] == 'n': return ('n', t[1])
            if t[0] == 'c' and t[1] in self.role: return (self.role[t[1]],) + tuple(go() for _ in range(t[2]))
            raise ValueError
        try:
            tree = go()
        except ValueError:
            return None
        return tree if pos[0] == len(code) else None

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ terms as Python trees
    def random_tree(self, rng, nvars, ops=2, p_op=0.55, p_num=0.2, weights=None, no_unit=False):
        """a random tidy term as a tree, with at most `ops` operators over the variables 0..nvars-1 (a variable's chance is its entry in `weights`, default 1); None if no tidy term was drawn. The same shape of
        draw as `TermGen.term`, but without the kernel: hundreds of draws cost less than one kernel term"""
        operators = [(r, a) for g, a in self.ops.items() for r in [self.role[g]]]
        for _ in range(12):
            left = [ops]
            def go():
                if operators and left[0] > 0 and rng.random() < p_op:
                    r, a = rng.choice(operators); left[0] -= 1
                    return (r,) + tuple(go() for _ in range(a))
                if self.numerals and (nvars == 0 or rng.random() < p_num): return ('n', rng.choice(self.numerals))
                ws = [(weights or {}).get(i, 1.0) for i in range(nvars)]
                return ('v', rng.choices(range(nvars), weights=ws)[0])
            t = go()
            if self.tidy(t) and not (no_unit and t[0] == 'n' and t[1] in (0, 1)): return t
        return None

    def code(self, tree):
        """the prefix code (typed term builder tokens) of a tree"""
        if tree[0] == 'v': return [('v', tree[1], 0)]
        if tree[0] == 'n': return [('n', tree[1])]
        out = [('c', self.gid_of[tree[0]], len(tree) - 1)]
        for c in tree[1:]: out += self.code(c)
        return out

    def value(self, tree, sigma):
        """the value of a tree when variable i is sigma[i] (integers)"""
        t = tree[0]
        if t == 'v': return sigma[tree[1]]
        if t == 'n': return tree[1]
        a = [self.value(c, sigma) for c in tree[1:]]
        if t == 'add': return a[0] + a[1]
        if t == 'mul': return a[0] * a[1]
        if t == 'sub': return a[0] - a[1]
        return -a[0]

    def tidy(self, tree):
        """is the term written the way a person would write it?  No operation on a unit or a zero (x + 0, x * 1, 0 * x, x - 0), no operation on two numerals (2 + 3), no x - x, no double negation, no negated
        numeral, and nothing that cancels (`(a + b) - a`: see synth_poly.reducible)"""
        if tree is None or not self._local(tree): return False
        p = SPY.tree_poly(tree)
        if p is None or tree[0] in ('v', 'n'): return p is not None
        vs, occ = SPY.tree_vars(tree)
        return not SPY.reducible(p, vs, occ)

    def _local(self, tree):
        if tree[0] in ('v', 'n'): return True
        role, kids = tree[0], tree[1:]
        if not all(self._local(c) for c in kids): return False
        num = [c[1] if c[0] == 'n' else None for c in kids]
        if role == 'neg': return kids[0][0] not in ('n', 'neg')
        if all(n is not None for n in num): return False
        if role == 'add': return not any(n == 0 for n in num)
        if role == 'mul': return not any(n in (0, 1) for n in num)
        if role == 'sub': return num[1] != 0 and num[0] != 0 and kids[0] != kids[1]
        return True


VOCABS = {
    'Z': dict(type_name='Z', ops={'Z.add': 2, 'Z.mul': 2, 'Z.sub': 2, 'Z.neg': 1}),
    'N': dict(type_name='N', ops={'N.add': 2, 'N.mul': 2}),
    'Nat': dict(type_name='Nat', ops={'add': 2, 'mul': 2}),
}


class TermGen:
    def __init__(self, eng, vocab):
        self.eng = eng; self.vocab = vocab

    def term(self, g, ty, rng, ops=2, p_op=0.55, p_num=0.2, avoid=(), weights=None, no_unit=False):
        """a random term of type `ty` in the context of g. `ops`: the most operators it may have; `p_op`: the chance of an operator at a hole while some are left; `p_num`: of a numeral rather than a variable at
        a leaf. `weights`: {variable level: weight} (the variables of the context that are not listed have weight 1); `avoid`: kernel handles the term must not be (tried a few times)."""
        for _ in range(12):
            code, node = self._once(g, ty, rng, ops, p_op, p_num, weights)
            tree = self.vocab.parse(code)
            if node not in avoid and self.vocab.tidy(tree) and not (no_unit and tree is not None and tree[0] == 'n' and tree[1] in (0, 1)): return code, node
        return code, node

    def build(self, g, ty, code):
        """the term of a code, built (and so kernel-checked) token by token in the context of g: (code, node)"""
        b = self.eng.builder(g, expected=ty)
        for tok in code: b.push(tok)
        if not b.done: raise GB.BuildError('the code is incomplete')
        return tuple(b.code), b.term()

    def _once(self, g, ty, rng, ops, p_op, p_num, weights):
        b = self.eng.builder(g, expected=ty); left = [ops]
        vocab = self.vocab
        while not b.done:
            toks = b.tokens()
            H = b.hole_type
            variables = [t for t in toks if t[0] == 'v' and t[2] == 0]
            numerals = [t for t in toks if t[0] == 'n' and t[1] in vocab.numerals]
            operators = [t for t in toks if t[0] == 'c' and t[1] in vocab.ops and vocab.ops[t[1]] == t[2]] if left[0] > 0 else []
            if operators and (not (variables or numerals) or rng.random() < p_op):
                t = rng.choice(operators); left[0] -= 1
            elif numerals and (not variables or rng.random() < p_num):
                t = rng.choice(numerals)
            elif variables:
                ws = [(weights or {}).get(v[1], 1.0) for v in variables]
                t = rng.choices(variables, weights=ws)[0]
            else:
                raise GB.BuildError('no way to fill a hole of type %s' % b._show(H))
            b.push(t)
        return tuple(b.code), b.term()
