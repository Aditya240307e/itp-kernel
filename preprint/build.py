#!/usr/bin/env python3
"""build.py -- flatten preprint/src/*.tex into the single file preprint/itp_preprint.tex, and compile it.

    python3 preprint/build.py                 numbers -> flatten -> pdflatex, bibtex, pdflatex until the references settle -> preprint/itp_preprint.pdf
    python3 preprint/build.py --no-pdf        only write preprint/itp_preprint.tex
    python3 preprint/build.py --fast          one pdflatex run (while writing)
    python3 preprint/build.py --draft         the numbers of measurements that do not exist yet are printed as ?? (a layout check; the final build refuses)

The flattened .tex is self-contained apart from references.bib: the generated numbers are pasted into it between the markers `% BEGIN GENERATED` and `% END GENERATED`.
Edit the files of src/, not the flattened file (it is overwritten).
"""
import os, sys, re, glob, subprocess, shutil, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src'); BUILD = os.path.join(HERE, 'build')
TEX = os.path.join(HERE, 'itp_preprint.tex'); PDF = os.path.join(HERE, 'itp_preprint.pdf')


def flatten(draft=False):
    subprocess.run([sys.executable, '-I', os.path.join(HERE, 'make_numbers.py')] + (['--allow-missing'] if draft else []), check=True)
    gen = open(os.path.join(HERE, 'generated_numbers.tex')).read().rstrip('\n')
    parts = [open(f).read().rstrip('\n') + '\n' for f in sorted(glob.glob(os.path.join(SRC, '*.tex')))]
    text = '\n'.join(parts)
    text = re.sub(r'% BEGIN GENERATED\n.*?% END GENERATED', lambda m: '% BEGIN GENERATED\n' + gen + '\n% END GENERATED', text, flags=re.S)
    with open(TEX, 'w') as f: f.write(text)
    return text


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def settled():
    """what another run of pdflatex reads back: the auxiliary files (labels, citations, the contents, the outline)"""
    return [open(os.path.join(BUILD, 'itp_preprint.' + e), 'rb').read() if os.path.exists(os.path.join(BUILD, 'itp_preprint.' + e)) else b'' for e in ('aux', 'toc', 'out')]


def compile_pdf(fast):
    os.makedirs(BUILD, exist_ok=True)
    shutil.copy(TEX, os.path.join(BUILD, 'itp_preprint.tex')); shutil.copy(os.path.join(HERE, 'references.bib'), os.path.join(BUILD, 'references.bib'))
    shutil.rmtree(os.path.join(BUILD, 'listings'), ignore_errors=True); shutil.copytree(os.path.join(HERE, 'listings'), os.path.join(BUILD, 'listings'), ignore=shutil.ignore_patterns('__pycache__'))
    tex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', 'itp_preprint.tex']
    code, out = run(tex, BUILD)
    if code != 0:
        print(out[-3500:]); sys.exit('pdflatex failed')
    if not fast:
        code, out = run(['bibtex', 'itp_preprint'], BUILD)
        bl = open(os.path.join(BUILD, 'itp_preprint.blg')).read()
        warn = [l for l in bl.splitlines() if l.startswith('Warning') or 'error' in l.lower()]
        if warn: print('\n'.join(warn[:30]))
        for _ in range(6):                                       # until the cross-references and the contents have settled, whatever the files of an earlier build were
            before = settled()
            code, out = run(tex, BUILD)
            if code != 0:
                print(out[-3500:]); sys.exit('pdflatex failed')
            if settled() == before: break
        else: sys.exit('the cross-references did not settle in six runs of pdflatex')
    log = open(os.path.join(BUILD, 'itp_preprint.log')).read()
    for pat, label in ((r'LaTeX Warning: (Citation|Reference)[^\n]*undefined[^\n]*', 'undefined'), (r'Overfull \\[hv]box[^\n]*', 'overfull'), (r'LaTeX Warning: Label[^\n]*multiply[^\n]*', 'multiply defined'), (r'Package pgfplots Warning[^\n]*', 'pgfplots')):
        hits = re.findall(pat, log)
        full = [m.group(0) for m in re.finditer(pat, log)]
        if full: print('%s: %d\n   %s' % (label, len(full), '\n   '.join(full[:12])))
    shutil.copy(os.path.join(BUILD, 'itp_preprint.pdf'), PDF)
    m = re.search(r'Output written on [^\n]*\((\d+) pages?', log)
    print('built %s (%s pages)' % (PDF, m.group(1) if m else '?'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--no-pdf', action='store_true'); ap.add_argument('--fast', action='store_true'); ap.add_argument('--draft', action='store_true'); a = ap.parse_args()
    flatten(draft=a.draft)
    if not a.no_pdf: compile_pdf(a.fast)
