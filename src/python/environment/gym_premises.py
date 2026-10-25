"""gym_premises.py -- WHICH GLOBALS A PROOF ATTEMPT MAY USE.

A proof attempt is made in a world: a library of definitions and lemmas, then the blocks of the problems of a bank. What an attempt may USE is decided by the engine's `premises`:

    None            every global of the world
    an int n        the globals with ids below n           (the library that precedes a theorem: the generator's regime, the library's own lemmas)
    a Premises      the ids below `upto` except the half-open ranges in `hidden`

The ranges exist for the problem bank.  The blocks of the problems follow one another in one world; with a plain prefix a problem would see the theorems and the helper lemmas of every problem before it,
so that the library of a late problem would be larger than that of an early one, would contain solutions of other problems (leakage between a training and a test split), and would not be the library a
model was pre-trained on.  A problem is therefore attacked with `Premises(upto, [(library_end, start_of_its_block)])`: the library, plus its own definitions (strict regime) or its own definitions and its
own helper lemmas (prefix regime), and nothing of the other problems.

A `Premises` is a value: immutable, hashable, with a JSON form (`to_json`: an int when nothing is hidden) that `of` reads back.
"""


class Premises:
    __slots__ = ('upto', 'hidden')

    def __init__(self, upto, hidden=()):
        if not isinstance(upto, int) or isinstance(upto, bool) or upto < 0: raise ValueError('upto must be a non-negative int, not %r' % (upto,))
        rs = sorted((int(a), int(b)) for a, b in hidden)
        merged = []
        for a, b in rs:
            if not 0 <= a <= b: raise ValueError('hidden range (%d, %d) is not a half-open range of ids' % (a, b))
            if a == b: continue
            if merged and a <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else: merged.append((a, b))
        object.__setattr__(self, 'upto', upto); object.__setattr__(self, 'hidden', tuple((a, min(b, upto)) for a, b in merged if a < upto))

    def __setattr__(self, name, value): raise AttributeError('Premises is immutable (it is a value: it is hashed)')
    def __delattr__(self, name): raise AttributeError('Premises is immutable (it is a value: it is hashed)')
    def __reduce__(self): return (Premises, (self.upto, self.hidden))                    # (pickle, copy: through the constructor, which normalises)

    def sees(self, gid):
        if not 0 <= gid < self.upto: return False
        for a, b in self.hidden:
            if a <= gid < b: return False
            if gid < a: break
        return True

    def count(self):
        """how many globals it offers"""
        return self.upto - sum(b - a for a, b in self.hidden)

    def __eq__(self, other): return isinstance(other, Premises) and (self.upto, self.hidden) == (other.upto, other.hidden)
    def __ne__(self, other): return not self == other
    def __hash__(self): return hash((self.upto, self.hidden))
    def __repr__(self): return 'Premises(%d%s)' % (self.upto, ', hidden=%s' % (list(self.hidden),) if self.hidden else '')

    def to_json(self):
        return self.upto if not self.hidden else {'upto': self.upto, 'hidden': [[a, b] for a, b in self.hidden]}


def of(x):
    """the premises that `x` says: None, an int, a Premises (returned as it is) or the JSON form of one; a Premises with nothing hidden is returned as its int"""
    if x is None or (isinstance(x, int) and not isinstance(x, bool)): return x
    if isinstance(x, Premises): return x.upto if not x.hidden else x
    if isinstance(x, dict):
        extra = set(x) - {'upto', 'hidden'}
        if extra or 'upto' not in x: raise ValueError('premises: unknown keys %s' % sorted(extra))
        return of(Premises(x['upto'], [tuple(r) for r in x.get('hidden', ())]))
    raise ValueError('premises: cannot read %r' % (x,))


def to_json(x):
    """the JSON form of premises (None, an int or a Premises)"""
    return x.to_json() if isinstance(x, Premises) else x


def upto(x, n):
    """the exclusive upper bound of the ids that premises `x` may offer in a world of `n` globals"""
    if x is None: return n
    if isinstance(x, Premises): return min(x.upto, n)
    return min(x, n)


def sees(x, gid, n=None):
    """may premises `x` use the global `gid` (in a world of `n` globals, when given)"""
    if n is not None and not 0 <= gid < n: return False
    if x is None: return gid >= 0
    if isinstance(x, Premises): return x.sees(gid)
    return 0 <= gid < x


def for_block(library_end, start, upto_):
    """the premises of a problem whose block begins at `start`: the library (ids below `library_end`) and the block up to `upto_` (an int)"""
    if upto_ < start: raise ValueError('the premises end (%d) before the block starts (%d)' % (upto_, start))
    return of(Premises(upto_, [(library_end, start)])) if start > library_end else upto_
