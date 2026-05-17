"""
Selectively pull ONLY the needed frames out of the 18 GB RTSD archive.

datasets/rstd/extract_list.txt holds the ~16.8k `rtsd-frames/<name>.jpg`
paths that contain our target classes (5_19_1, 5_20, 6_4).  This reads them
straight from the archive without unpacking the whole thing.

Usage:
    python tools/rtsd_extract.py --archive PATH_TO_RTSD_ARCHIVE
        [--list datasets/rstd/extract_list.txt]
        [--out  datasets/rstd]

Supports .zip and .tar/.tar.gz/.tgz.  Members are matched by suffix so it
works whether the archive stores 'rtsd-frames/x.jpg' or
'RTSD/rtsd-frames/x.jpg'.
"""
import argparse
import tarfile
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--list", type=Path,
                    default=Path("../datasets/rstd/extract_list.txt"))
    ap.add_argument("--out", type=Path, default=Path("../datasets/rstd"))
    a = ap.parse_args()

    if not a.archive.exists():
        raise SystemExit(f"archive not found: {a.archive}")
    wanted = [l.strip() for l in a.list.read_text().splitlines() if l.strip()]
    want_set = set(wanted)
    # frame file names are unique in RTSD -> match by basename too, which is
    # robust to any nesting depth (e.g. rtsd-frames/rtsd-frames/x.jpg).
    base2rel = {Path(w).name: w for w in wanted}
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"need {len(want_set)} frames -> {a.out}")

    got = 0

    def take(name: str) -> str | None:
        """Return the canonical wanted rel path if `name` matches, else None.
        Output always normalises to a single 'rtsd-frames/<file>' level."""
        n = name.replace("\\", "/")
        if n in want_set:
            return n
        i = n.rfind("rtsd-frames/")
        if i != -1 and n[i:] in want_set:
            return n[i:]
        return base2rel.get(Path(n).name)   # nesting-proof fallback

    suf = a.archive.suffix.lower()
    if suf == ".zip":
        with zipfile.ZipFile(a.archive) as z:
            for info in z.infolist():
                rel = take(info.filename)
                if rel is None:
                    continue
                dst = a.out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(dst, "wb") as f:
                    f.write(src.read())
                got += 1
                if got % 1000 == 0:
                    print(f"  {got}/{len(want_set)}")
    elif suf in (".tar", ".gz", ".tgz") or a.archive.name.endswith(".tar.gz"):
        with tarfile.open(a.archive) as t:
            for m in t:
                if not m.isfile():
                    continue
                rel = take(m.name)
                if rel is None:
                    continue
                dst = a.out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(dst, "wb") as f:
                    f.write(src.read())
                got += 1
                if got % 1000 == 0:
                    print(f"  {got}/{len(want_set)}")
    else:
        raise SystemExit(f"unsupported archive type: {suf}")

    print(f"done: extracted {got}/{len(want_set)} frames into {a.out/'rtsd-frames'}")
    miss = len(want_set) - got
    if miss:
        print(f"WARNING: {miss} listed frames not found in archive "
              f"(name-prefix mismatch?) — send me one archive path sample")


if __name__ == "__main__":
    main()
