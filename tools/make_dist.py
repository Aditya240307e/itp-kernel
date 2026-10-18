#!/usr/bin/env python3
"""make_dist.py -- pack the repository into a .tar.gz and a .zip that hold exactly what is needed to use it.

    python3 tools/make_dist.py                         dist/itp-YYYY-MM-DD-<commit>.tar.gz and .zip
    python3 tools/make_dist.py --name NAME             another archive name (also the top-level directory inside it)
    python3 tools/make_dist.py --artifacts DIR         also include the files of DIR under artifacts/ (trained models, which are not in git), with a listing and digests
    python3 tools/make_dist.py --out DIR               write into DIR instead of dist/

What goes in: the files git knows (tracked and untracked-but-not-ignored) when the repository has a .git directory, otherwise every file except build products and caches.  What stays out: .git, build/,
dist/, __pycache__, *.pyc, *.so.  The archive also holds MANIFEST.sha256 (the SHA-256 of every file in it, in `sha256sum -c` format), and a file RELEASE.txt saying which commit it was made from.
"""
import argparse, datetime, hashlib, io, os, subprocess, sys, tarfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', 'build', 'dist', '__pycache__', '.venv', 'artifacts'}
SKIP_SUFFIX = ('.pyc', '.so', '.pyd', '.o')


def git(*args):
    r = subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def file_list():
    """sorted relative paths of the files to pack"""
    out = git('ls-files', '-co', '--exclude-standard') if os.path.isdir(os.path.join(ROOT, '.git')) else None
    if out is not None:
        names = [n for n in out.split('\n') if n and os.path.isfile(os.path.join(ROOT, n))]
    else:
        names = []
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            names += [os.path.relpath(os.path.join(d, f), ROOT) for f in files]
    generated = ('MANIFEST.sha256', 'RELEASE.txt')                  # (an extracted archive holds the previous ones; they are made afresh below)
    return sorted(n for n in names if not n.endswith(SKIP_SUFFIX) and not set(n.split('/')) & SKIP_DIRS and n not in generated)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--out', default=os.path.join(ROOT, 'dist')); ap.add_argument('--artifacts')
    a = ap.parse_args(argv)
    commit = git('rev-parse', '--short', 'HEAD') or 'nogit'
    dirty = bool(git('status', '--porcelain'))
    name = a.name or 'itp-%s-%s%s' % (datetime.date.today().isoformat(), commit, '-dirty' if dirty else '')
    members = [(n, os.path.join(ROOT, n)) for n in file_list()]
    if a.artifacts:
        for f in sorted(os.listdir(a.artifacts)):
            p = os.path.join(a.artifacts, f)
            if os.path.isfile(p): members.append(('artifacts/' + f, p))
    manifest = ''.join('%s  %s\n' % (sha256(p), n) for n, p in members)
    release = 'commit %s%s\nbranch %s\nmade %s\nfiles %d\n' % (git('rev-parse', 'HEAD') or 'unknown', ' (with uncommitted changes)' if dirty else '', git('branch', '--show-current') or 'unknown',
                                                              datetime.datetime.now().astimezone().isoformat(timespec='seconds'), len(members))
    extra = [('MANIFEST.sha256', manifest.encode()), ('RELEASE.txt', release.encode())]
    os.makedirs(a.out, exist_ok=True)
    tar_path, zip_path = os.path.join(a.out, name + '.tar.gz'), os.path.join(a.out, name + '.zip')
    with tarfile.open(tar_path, 'w:gz') as tf:
        for n, p in members:
            ti = tf.gettarinfo(p, arcname='%s/%s' % (name, n)); ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
            with open(p, 'rb') as f: tf.addfile(ti, f)
        for n, data in extra:
            ti = tarfile.TarInfo('%s/%s' % (name, n)); ti.size = len(data); ti.mtime = int(datetime.datetime.now().timestamp()); ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for n, p in members: zf.write(p, '%s/%s' % (name, n))
        for n, data in extra: zf.writestr('%s/%s' % (name, n), data)
    print('%s  (%.1f MB, %d files)' % (tar_path, os.path.getsize(tar_path) / 1e6, len(members) + len(extra)))
    print('%s  (%.1f MB)' % (zip_path, os.path.getsize(zip_path) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
