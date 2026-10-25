#!/usr/bin/env python3
"""make_dist.py -- pack the repository into a .tar.gz and a .zip that hold exactly what is needed to use it.

    python3 tools/make_dist.py                         dist/itp-YYYY-MM-DD-<commit>.tar.gz and .zip
    python3 tools/make_dist.py --name NAME             another archive name (also the top-level directory inside it)
    python3 tools/make_dist.py --artifacts DIR         also include the files of DIR under artifacts/ (trained models, which are not in git), with a listing and digests
    python3 tools/make_dist.py --out DIR               write into DIR instead of dist/
    python3 tools/make_dist.py --git                   also put the repository's history into the archive as a .git directory: the extracted directory is a working git repository at the commit the archive was made from

What goes in: the files git knows (tracked and untracked-but-not-ignored) when the repository has a .git directory, otherwise every file except build products and caches.  What stays out: .git (unless
--git), build/, dist/, __pycache__, *.pyc, *.so.  The archive also holds MANIFEST.sha256 (the SHA-256 of every file in it, in `sha256sum -c` format), and a file RELEASE.txt saying which commit it was made from.

--git adds the history: a .git directory made for the purpose from the repository's -- its objects and its branches and tags, packed (`git gc`), with a NEW minimal configuration: no remotes (their URLs are
paths on the machine that made the archive), no user.name / user.email (commits made from the extracted directory carry the identity of whoever makes them), no hooks, no reflogs, no stashes, no remote-tracking
branches; the index is rebuilt from HEAD, and `info/exclude` hides MANIFEST.sha256 and RELEASE.txt, which git does not know.  The extracted directory is a working repository at the commit the archive was made
from: `git log`, `git fsck`, `git status` (clean, unless the tree had uncommitted changes when the archive was made -- RELEASE.txt says so).  The files of .git are not in MANIFEST.sha256 (git checks them
itself: `git fsck`); the empty directories of .git (refs/heads, ...) are in the archive, because git does not accept a repository without them.
"""
import argparse, datetime, hashlib, io, os, shutil, subprocess, sys, tarfile, tempfile, zipfile

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


def git_dir_for_archive(tmp):
    """a .git directory in `tmp` for an archive (see the module documentation); returns [(name inside the archive, path, is a directory)], the directories included"""
    src, dst = os.path.join(ROOT, '.git'), os.path.join(tmp, '.git')
    if not os.path.isdir(src) or shutil.which('git') is None: raise SystemExit('make_dist --git: this directory is not a git repository (or git is not installed)')
    os.makedirs(dst)
    for d in ('objects', 'refs'): shutil.copytree(os.path.join(src, d), os.path.join(dst, d), symlinks=True)
    for f in ('HEAD', 'packed-refs'):
        if os.path.exists(os.path.join(src, f)): shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    def g(*args):
        r = subprocess.run(['git', '--git-dir', dst] + list(args), capture_output=True, text=True)
        if r.returncode != 0: raise SystemExit('make_dist --git: `git %s` failed: %s' % (' '.join(args), (r.stderr or r.stdout).strip()))
        return r.stdout.strip()
    def source_config(*args): return subprocess.run(['git', 'config', '--file', os.path.join(src, 'config')] + list(args), capture_output=True, text=True).stdout
    g('config', 'core.repositoryformatversion', source_config('--get', 'core.repositoryformatversion').strip() or '0')         # (the format, and the extensions it needs: sha256 objects, say)
    for line in source_config('--get-regexp', r'^extensions\.').splitlines():
        key, _, val = line.partition(' ')
        g('config', key, val)
    g('config', 'core.filemode', 'true'); g('config', 'core.bare', 'false'); g('config', 'core.logallrefupdates', 'true')          # (nothing else of the configuration: no remote, no user, no tracking branches)
    for ref in g('for-each-ref', '--format=%(refname)').split():
        if not ref.startswith(('refs/heads/', 'refs/tags/')): g('update-ref', '-d', ref)                                          # (remote-tracking branches, stashes, backups of a rewrite)
    os.makedirs(os.path.join(dst, 'info'), exist_ok=True)
    with open(os.path.join(dst, 'info', 'exclude'), 'w') as f: f.write('# written by tools/make_dist.py: the two files it adds to the archive, which git does not know\n/MANIFEST.sha256\n/RELEASE.txt\n')
    g('read-tree', 'HEAD')
    g('gc', '--quiet', '--prune=now')
    shutil.rmtree(os.path.join(dst, 'logs'), ignore_errors=True)
    g('fsck', '--no-dangling')
    out = []
    for d, dirs, files in os.walk(dst):
        dirs.sort()
        out.append((os.path.relpath(d, tmp), d, True))
        for f in sorted(files): out.append((os.path.relpath(os.path.join(d, f), tmp), os.path.join(d, f), False))
    return sorted(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--out', default=os.path.join(ROOT, 'dist')); ap.add_argument('--artifacts'); ap.add_argument('--git', action='store_true', help='also put the history into the archive (a .git directory)')
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
    gtmp = tempfile.mkdtemp(prefix='itp_dist_git_') if a.git else None
    try:
        return pack(a, name, members, manifest, dirty, git_dir_for_archive(gtmp) if a.git else [])
    finally:
        if gtmp: shutil.rmtree(gtmp, ignore_errors=True)


def pack(a, name, members, manifest, dirty, git_members):
    release = 'commit %s%s\nbranch %s\nmade %s\nfiles %d\n' % (git('rev-parse', 'HEAD') or 'unknown', ' (with uncommitted changes)' if dirty else '', git('branch', '--show-current') or 'unknown',
                                                              datetime.datetime.now().astimezone().isoformat(timespec='seconds'), len(members))
    if git_members: release += 'history %s commits, in .git (branches and tags; no remotes, hooks or user identity; `git log`, `git fsck`, `git status`)\n' % (git('rev-list', '--count', 'HEAD') or '?')
    extra = [('MANIFEST.sha256', manifest.encode()), ('RELEASE.txt', release.encode())]
    os.makedirs(a.out, exist_ok=True)
    tar_path, zip_path = os.path.join(a.out, name + '.tar.gz'), os.path.join(a.out, name + '.zip')
    with tarfile.open(tar_path, 'w:gz') as tf:
        for n, p in members:
            ti = tf.gettarinfo(p, arcname='%s/%s' % (name, n)); ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
            with open(p, 'rb') as f: tf.addfile(ti, f)
        for n, p, is_dir in git_members:
            ti = tf.gettarinfo(p, arcname='%s/%s' % (name, n)); ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
            if is_dir: tf.addfile(ti)
            else:
                with open(p, 'rb') as f: tf.addfile(ti, f)
        for n, data in extra:
            ti = tarfile.TarInfo('%s/%s' % (name, n)); ti.size = len(data); ti.mtime = int(datetime.datetime.now().timestamp()); ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for n, p in members: zf.write(p, '%s/%s' % (name, n))
        for n, p, is_dir in git_members: zf.write(p, '%s/%s' % (name, n))                       # (a directory is written as an entry `name/`)
        for n, data in extra: zf.writestr('%s/%s' % (name, n), data)
    print('%s  (%.1f MB, %d files%s)' % (tar_path, os.path.getsize(tar_path) / 1e6, len(members) + len(extra), ' + the history in .git' if git_members else ''))
    print('%s  (%.1f MB)' % (zip_path, os.path.getsize(zip_path) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
