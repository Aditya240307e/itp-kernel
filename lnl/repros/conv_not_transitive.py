"""Finding F28: kernel conversion (`conv`, NbE) is incomplete and NOT transitive for stuck `match`es.

    python3 repros/conv_not_transitive.py

Both  HAVE2 = fst q + snd (pair a b)  and  EXP (the same sum with the projection of the pair written out as stuck matches) convert to their own normal forms, but the two normal forms
differ in how far they reduced the matches that sit inside a stuck match's branches, and `conv` of the two original terms answers False (in both directions). It is an INCOMPLETENESS only:
`conv` never says True for terms that are not convertible. The tactic layer works around it by normalising both sides (`simp_normalize`) before comparing, and `rw`/`change` by pattern use the
normal form the kernel itself produced.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100000)
import bin_lib, quot_lib
w = bin_lib.build_bin_defs()
quot_lib.declare_prod(w)
N = w['N']; Prod = w['Prod']; pair, fst, snd = w['pair'], w['fst'], w['snd']; nadd = w['N.add']
NN = Prod(N, N)
q = w._bind(NN, 'q')[1]; a = w._bind(N, 'a')[1]; b = w._bind(N, 'b')[1]; p0 = w._bind(NN, 'p0')[1]
k = w.k
H0 = nadd(fst(N, N, q), snd(N, N, p0))
Hn = w.normalize(H0)
pr = pair(N, N, a, b)
EXP = w.T(k.replace_fvars(w._n(Hn), [(3, w._n(pr))]))          # the normal form of  fst q + snd p0  with p0 := pair a b substituted afterwards
HAVE2 = nadd(fst(N, N, q), snd(N, N, pr))
r1, r2 = w.conv(HAVE2, EXP), w.conv(EXP, HAVE2)
print('conv(HAVE2, EXP) =', r1, '  conv(EXP, HAVE2) =', r2)
print('conv(HAVE2, norm HAVE2) =', w.conv(HAVE2, w.normalize(HAVE2)), '  conv(EXP, norm EXP) =', w.conv(EXP, w.normalize(EXP)))
print('the two normal forms are the same term:', w._n(w.normalize(HAVE2)) == w._n(w.normalize(EXP)))
print('=> NOT TRANSITIVE (HAVE2 ~ norm HAVE2, EXP ~ norm EXP, but not HAVE2 ~ EXP)' if not r1 and not w._n(w.normalize(HAVE2)) == w._n(w.normalize(EXP)) else 'transitive here')
