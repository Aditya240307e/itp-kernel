"""audit.py -- what does a theorem rest on?

`dependencies(k, tree)` walks a term the way Lean's `#print axioms` does: every constant it mentions is classified, definitions are unfolded (their
type AND value are walked, transitively), and the result is the set of
    axioms  -- constants with no value that are neither inductive families nor constructors (em, propext, funext, choice, ...)
    quot    -- the five quotient primitives (Quot, Quot.mk, Quot.lift, Quot.ind, Quot.sound); their one computation rule lives in the kernel
that the term can reach. Inductive families and constructors are part of the kernel's checked declaration path and are not reported.

The point of the function is the trusted-base accounting of a proof: a result that reports no axioms and no quotient constants is a theorem of the
bare kernel; one that reports only Quot.sound is a theorem of the kernel plus the quotient rule; and so on. Nothing here is trusted by the kernel."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'src', 'python')); import itp_paths  # noqa: E402,F401
import harness as H
from harness import parse_raw

KINDS = ('inductive', 'constructor', 'definition', 'quot', 'axiom')


def kind(k, gid):
    name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
    if k.quot_role(gid): return 'quot'
    if cf >= 0: return 'constructor'
    if k.is_inductive(gid): return 'inductive'
    if val is not None: return 'definition'
    return 'axiom'


def consts_of(tree):
    """every global id mentioned in a hybrid term tree (iterative: proof terms can be deep)"""
    out, stack = set(), [tree]
    while stack:
        t = stack.pop()
        tag = t[0]
        if tag == 'const': out.add(t[1])
        elif tag == 'app': stack.append(t[1]); stack.append(t[2])
        elif tag in ('lam', 'pi'): stack.append(t[1]); stack.append(t[2])
        elif tag == 'let': stack.extend(t[1:4])
        elif tag == 'match': stack.append(t[1]); stack.append(t[2]); stack.extend(t[3])
        elif tag == 'fix': stack.append(t[2]); stack.append(t[3])
    return out


def dependencies(k, tree, memo=None):
    """(axiom names, quotient-constant names) reachable from `tree` (a hybrid term tree)"""
    memo = {} if memo is None else memo
    seen, work, axioms, quots = set(), list(consts_of(tree)), set(), set()
    while work:
        gid = work.pop()
        if gid in seen: continue
        seen.add(gid)
        name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
        kd = kind(k, gid)
        if kd == 'axiom': axioms.add(name)
        elif kd == 'quot': quots.add(name)
        if gid not in memo:
            memo[gid] = consts_of(parse_raw(k.dump_raw(ty), True)) | (consts_of(parse_raw(k.dump_raw(val), True)) if val is not None else set())
        work.extend(memo[gid])
    return axioms, quots


def dependencies_of_global(k, gid, memo=None):
    return dependencies(k, ('const', gid), memo)
