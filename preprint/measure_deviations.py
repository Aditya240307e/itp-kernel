#!/usr/bin/env python3
"""measure_deviations.py -- does any shipped inductive family fall under deviation D2?  (no PyTorch needed)

    python3 -I preprint/measure_deviations.py --out preprint/measurements/deviations.json           # about two minutes, 2 GB
    python3 -I preprint/measure_deviations.py --quick --out /tmp/deviations_quick.json              # the library chain only (seconds)

The kernel's strict-positivity check (condition A5 of the report) does not look inside the arguments of the head I of a recursive field, nor inside the result indices of a constructor (deviation D2).
The usual condition forbids the family under definition in both places.  This script reads back every inductive family of the world of the whole bank (the libraries and the statements of the bank) through the
kernel's term view and reports, for each constructor of each family, whether I occurs

  in an argument of a recursive occurrence (a field whose type, after its own products, is headed by I),
  in an argument of the constructor's result type,
  in a field type anywhere else (not as the head of the field's final codomain, or to the left of an arrow): the case the kernel does reject (A5), counted as a control,

so that the number of families D2 could have touched is measured and not assumed.  Nothing here changes the kernel.
"""
import os, sys, time, json, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src', 'python')); sys.path.insert(0, HERE)
import itp_paths  # noqa: E402,F401
sys.setrecursionlimit(100000)
import measure as M  # noqa: E402                                  (the machine description)
clock = time.perf_counter


def children(k, n):
    out = []
    for x in k.view(n)[1:]:
        if hasattr(x, 'tag'): out.append(x)
        elif isinstance(x, (list, tuple)): out.extend(y for y in x if hasattr(y, 'tag'))
    return out


def occurs(k, n, gid, memo):
    """does the constant `gid` occur in the term `n` (shared subterms are visited once)"""
    key = (n.tag, n.index)
    if key in memo: return memo[key]
    v = k.view(n)
    if v[0] == 'const' and v[1] == gid: r = True
    else:
        memo[key] = False                                           # (terms are acyclic: this only stops a revisit while the node is open)
        r = any(occurs(k, c, gid, memo) for c in children(k, n))
    memo[key] = r
    return r


def spine(k, n):
    args = []
    while True:
        v = k.view(n)
        if v[0] != 'app': return n, args[::-1]
        args.append(v[2]); n = v[1]


def peel_pis(k, n):
    doms = []
    while True:
        v = k.view(n)
        if v[0] != 'pi': return doms, n
        doms.append(v[1]); n = v[2]


def scan_family(k, gid):
    """for each constructor of the family: which of the three places the family occurs in"""
    name, ty, val, cf, ci, ctors, rec = k.global_info(gid)
    memo = {}
    rows = []
    for cg in ctors:
        cname, cty = k.global_info(cg)[:2]
        fields, res = peel_pis(k, cty)
        in_rec_index = in_other_field = in_result_index = False; recursive_fields = 0
        for f in fields:
            zs, cod = peel_pis(k, f)
            if any(occurs(k, z, gid, memo) for z in zs): in_other_field = True            # to the left of an arrow: rejected by A5
            head, args = spine(k, cod)
            hv = k.view(head)
            if hv[0] == 'const' and hv[1] == gid:
                recursive_fields += 1
                if any(occurs(k, a, gid, memo) for a in args): in_rec_index = True
            elif occurs(k, cod, gid, memo): in_other_field = True                          # nested in another family's argument: rejected by A5
        head, args = spine(k, res)
        if any(occurs(k, a, gid, memo) for a in args): in_result_index = True
        rows.append({'constructor': cname, 'recursive_fields': recursive_fields, 'family_in_index_of_recursive_field': in_rec_index, 'family_in_result_index': in_result_index, 'family_elsewhere_in_a_field': in_other_field})
    return {'family': name, 'constructors': rows}


def control():
    """the scan must flag the two families of listings/dev_positivity.py and must not flag the families of the Nat library: a positive and a negative control, run before the real scan"""
    import nat_lib
    w = nat_lib.build_nat(); k = w.k; TYPE, PROP, False_ = w.TYPE, w.PROP, w['False']
    w.inductive('Q', w.arrow(PROP, TYPE), [('mkq', [PROP, lambda A: w['Q'](w.arrow(w['Q'](A), False_))], lambda A, h: [A])])
    w.inductive('R', w.arrow(PROP, TYPE), [('mkr', [PROP], lambda A: [w.arrow(w['R'](A), False_)])])
    flagged = lambda r: any(c['family_in_index_of_recursive_field'] or c['family_in_result_index'] or c['family_elsewhere_in_a_field'] for c in r['constructors'])
    got = {name: flagged(scan_family(k, w.gid[name])) for name in ('Q', 'R', 'Nat', 'Le', 'Eq', 'And', 'Or', 'Ex')}
    want = {'Q': True, 'R': True, 'Nat': False, 'Le': False, 'Eq': False, 'And': False, 'Or': False, 'Ex': False}
    assert got == want, 'the scan does not tell the two families of the listing from the families of the library: %s' % got
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True); ap.add_argument('--quick', action='store_true', help='the library chain only (not the bank)')
    a = ap.parse_args(argv)
    out = {'machine': M.machine(), 'args': {k: v for k, v in vars(a).items()}}
    out['control'] = control(); print('control: the two families of the listing are flagged, the families of the Nat library are not')
    t0 = clock()
    if a.quick:
        import nat_lib
        w = nat_lib.build_nat()
        k = w.k
    else:
        import bank_all
        bk = bank_all.load(); k = bk.w.k
    out['world_seconds'] = round(clock() - t0, 2); out['globals'] = k.num_globals()
    fams = [g for g in range(k.num_globals()) if k.is_inductive(g)]
    res = [scan_family(k, g) for g in fams]
    ctors = sum(len(r['constructors']) for r in res)
    rec_fams = sum(1 for r in res if any(c['recursive_fields'] for c in r['constructors']))
    bad_idx = [r['family'] for r in res if any(c['family_in_index_of_recursive_field'] for c in r['constructors'])]
    bad_res = [r['family'] for r in res if any(c['family_in_result_index'] for c in r['constructors'])]
    bad_else = [r['family'] for r in res if any(c['family_elsewhere_in_a_field'] for c in r['constructors'])]
    out['families'] = len(fams); out['constructors'] = ctors; out['families_with_a_recursive_field'] = rec_fams
    out['family_in_index_of_recursive_field'] = bad_idx; out['family_in_result_index'] = bad_res; out['family_elsewhere_in_a_field'] = bad_else
    out['scan'] = res
    print('%d globals, %d inductive families, %d constructors, %d families with a recursive field' % (k.num_globals(), len(fams), ctors, rec_fams))
    print('  family in an index of a recursive field: %s | in a result index: %s | elsewhere in a field (must be none: A5 rejects it): %s' % (bad_idx, bad_res, bad_else))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f: json.dump(out, f, indent=1, sort_keys=True, default=str)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
