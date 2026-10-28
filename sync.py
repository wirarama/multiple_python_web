#!/usr/bin/env python3
"""
Sinkronisasi script dari repo sumber ke folder apps/.

Hanya file yang cocok dengan pola 'include' di sync_config.json yang disalin,
sehingga dokumen paper/HKI, cache, venv, dan berkas eksperimen tidak ikut.
Hanya memakai pustaka standar Python, jadi bisa dijalankan sebelum venv dibuat
(di Windows maupun di server Linux).

Contoh:
    python sync.py                         # dari folder saudara (../software_defined_network, dst.)
    python sync.py --only sdn,smarthome    # sebagian aplikasi saja
    python sync.py --dry-run               # lihat perubahan tanpa menyalin
    python sync.py --source git            # clone dangkal dari GitHub (cocok di server)
    python sync.py --source git --git-base git@github.com:wirarama/   # repo private via SSH
    python sync.py --source-root D:/repo   # folder induk sumber lain
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "sync_config.json"
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".ipynb_checkpoints",
             ".pytest_cache", ".mypy_cache", "instance"}


def pattern_to_regex(pat: str) -> re.Pattern:
    """Glob sederhana: '**/' = nol/lebih folder, '**' = apa saja, '*' = tanpa '/'."""
    i, out = 0, []
    while i < len(pat):
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def matches(rel: str, regs) -> bool:
    return any(r.match(rel) for r in regs)


def list_files(base: Path):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith((".pyc", ".pyo")):
                continue
            p = Path(dirpath) / fn
            yield p.relative_to(base).as_posix()


def select(src: Path, spec: dict):
    inc = [pattern_to_regex(p) for p in spec.get("include", ["**"])]
    exc = [pattern_to_regex(p) for p in spec.get("exclude", [])]
    return sorted(r for r in list_files(src) if matches(r, inc) and not matches(r, exc))


def git_clone(url: str, workdir: Path, name: str):
    dest = workdir / name
    print(f"  git clone --depth 1 {url}")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    r = subprocess.run(["git", "clone", "--quiet", "--depth", "1", url, str(dest)], env=env)
    if r.returncode != 0:
        print("  ! clone gagal. Bila repo private, pakai kunci SSH: "
              "python sync.py --source git --git-base git@github.com:wirarama/")
        return None
    return dest


def req_names(path: Path, _seen=None):
    names = set()
    _seen = _seen if _seen is not None else set()
    if not path.is_file() or path in _seen:
        return names
    _seen.add(path)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith(("-r ", "--requirement ")):
            names |= req_names(path.parent / line.split(None, 1)[1], _seen)
            continue
        if not line or line.startswith("-"):
            continue
        m = re.match(r"[A-Za-z0-9_.\-]+", line)
        if m:
            names.add(m.group(0).lower().replace("_", "-"))
    return names


def remap(rel: str, spec: dict) -> str:
    """'rename': {"data/": "data_seed/"} memindahkan awalan path di tujuan."""
    for a, b in spec.get("rename", {}).items():
        if rel.startswith(a):
            return b + rel[len(a):]
    return rel


def sync_app(key: str, spec: dict, src: Path, dry: bool, prune: bool) -> dict:
    dest = ROOT / spec["dest"]
    files = select(src, spec)
    if not files:
        raise SystemExit(f"[{key}] tidak ada file cocok di {src} - periksa pola include.")
    stat = {"baru": 0, "ubah": 0, "sama": 0, "hapus": 0}
    for rel_src in files:
        rel = remap(rel_src, spec)
        s, d = src / rel_src, dest / rel
        if d.exists() and filecmp.cmp(s, d, shallow=False):
            stat["sama"] += 1
            continue
        stat["ubah" if d.exists() else "baru"] += 1
        print(f"  {'~' if d.exists() else '+'} {rel}")
        if not dry:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    if prune and dest.exists():
        keep = {remap(r, spec) for r in files}
        pres = [pattern_to_regex(p) for p in spec.get("preserve", [])]
        for rel in list(list_files(dest)):
            if rel in keep or matches(rel, pres):
                continue
            stat["hapus"] += 1
            print(f"  - {rel}")
            if not dry:
                (dest / rel).unlink()
        if not dry:  # buang folder kosong
            for dirpath, _, _ in sorted(os.walk(dest), key=lambda t: -len(t[0])):
                p = Path(dirpath)
                if p != dest and p.name not in SKIP_DIRS and not any(p.iterdir()):
                    p.rmdir()
    return stat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["local", "git"], default="local",
                    help="local = folder repo di komputer ini; git = clone dari URL GitHub")
    ap.add_argument("--source-root", help="folder induk berisi repo sumber (menimpa path 'local')")
    ap.add_argument("--git-base", help="ganti awalan URL git, mis. git@github.com:wirarama/ (repo private via SSH)")
    ap.add_argument("--only", help="daftar kunci aplikasi dipisah koma, mis. sdn,pyspark")
    ap.add_argument("--dry-run", action="store_true", help="tampilkan perubahan saja")
    ap.add_argument("--no-prune", action="store_true", help="jangan hapus file lama di apps/")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["apps"]
    keys = list(cfg)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        bad = [k for k in keys if k not in cfg]
        if bad:
            raise SystemExit(f"Kunci tidak dikenal: {bad}. Pilihan: {list(cfg)}")

    tmp = Path(tempfile.mkdtemp(prefix="mpw_sync_")) if args.source == "git" else None
    try:
        for k in keys:
            spec = cfg[k]
            print(f"[{k}] {spec['title']}")
            if args.source == "git":
                url = spec["git"]
                if args.git_base:
                    url = args.git_base + url.rstrip("/").rsplit("/", 1)[-1]
                src = git_clone(url, tmp, k)
                if src is None:
                    continue
            elif args.source_root:
                src = Path(args.source_root).expanduser().resolve() / Path(spec["local"]).name
            else:
                src = (ROOT / spec["local"]).resolve()
            if not src.is_dir():
                print(f"  ! sumber tidak ditemukan: {src} (dilewati)")
                continue
            st = sync_app(k, spec, src, args.dry_run, not args.no_prune)
            print(f"  baru {st['baru']}, ubah {st['ubah']}, sama {st['sama']}, hapus {st['hapus']}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    # Peringatan bila ada paket baru di requirements aplikasi yang belum ada di requirements gabungan
    combined = req_names(ROOT / "requirements.txt")
    missing = {}
    for k, spec in cfg.items():
        for n in req_names(ROOT / spec["dest"] / "requirements.txt") - combined:
            missing.setdefault(n, []).append(k)
    if missing:
        print("\nPERHATIAN: paket berikut belum tercantum di requirements.txt gabungan:")
        for n, ks in sorted(missing.items()):
            print(f"  {n}  (dipakai: {', '.join(ks)})")
        print("Tambahkan ke requirements.txt lalu jalankan ulang: bash setup.sh")
    print("\nSelesai." + (" (dry-run, tidak ada file diubah)" if args.dry_run else ""))


if __name__ == "__main__":
    sys.exit(main())
