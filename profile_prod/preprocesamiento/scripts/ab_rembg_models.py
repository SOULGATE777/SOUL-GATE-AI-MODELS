#!/usr/bin/env python3
"""Offline A/B helper for historical rembg mattes (NOT used by the live service).

Production white-BG is Photoroom (`PHOTOROOM_API_KEY`). This script still imports
`rembg`, which is no longer in service requirements — pip-install rembg separately
if you need to re-run A/B offline (SERVER ONLY — needs >=12GB RAM).

Defaults:
  --protect none   (never invent a half-frame protect rect — that fabricates BG shards)
  --models u2net isnet-general-use
  refuses hosts with <12GB RAM unless --force
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _host_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _iter_images(roots: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            out.append(root)
            continue
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return out


def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _white_ring(bgr: np.ndarray, pct: float = 0.08, min_px: int = 12) -> np.ndarray:
    """Mimic prod edge ring (white border) before rembg — optional mode."""
    h, w = bgr.shape[:2]
    border = max(min_px, int(round(min(h, w) * pct)))
    out = np.full((h + 2 * border, w + 2 * border, 3), 255, dtype=np.uint8)
    out[border : border + h, border : border + w] = bgr
    return out


def _metrics(orig_bgr: np.ndarray, comp_bgr: np.ndarray, alpha: np.ndarray) -> Dict[str, float]:
    """white_purity = fraction of near-white BG pixels; face_core_mae on central crop."""
    h, w = orig_bgr.shape[:2]
    white = (comp_bgr[..., 0] > 248) & (comp_bgr[..., 1] > 248) & (comp_bgr[..., 2] > 248)
    bg = alpha < 32
    bg_n = int(bg.sum()) or 1
    white_purity = float((white & bg).sum()) / float(bg_n)

    fg = alpha >= 128
    fg_frac = float(fg.sum()) / float(h * w)

    x0, x1 = int(w * 0.35), int(w * 0.65)
    y0, y1 = int(h * 0.30), int(h * 0.70)
    core_o = orig_bgr[y0:y1, x0:x1].astype(np.float32)
    core_c = comp_bgr[y0:y1, x0:x1].astype(np.float32)
    face_core_mae = float(np.mean(np.abs(core_o - core_c))) if core_o.size else 0.0

    bg_mean = float(comp_bgr[bg].mean()) if bg_n else 255.0

    return {
        "white_purity": white_purity,
        "fg_frac": fg_frac,
        "face_core_mae": face_core_mae,
        "bg_mean": bg_mean,
    }


def _run_one(
    path: Path,
    session,
    model: str,
    out_dir: Path,
    max_side: int,
    ring: bool,
) -> Dict[str, object]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"path": str(path), "model": model, "error": "imread_failed"}

    work = _resize_max_side(bgr, max_side)
    if ring:
        work = _white_ring(work)

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    t0 = time.perf_counter()
    mask = remove(pil, session=session, only_mask=True)
    cut = remove(pil, session=session)
    elapsed = time.perf_counter() - t0

    alpha = np.asarray(mask)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    cut_rgba = np.asarray(cut)
    if cut_rgba.shape[-1] == 4:
        rgb_cut = cut_rgba[..., :3]
        a = cut_rgba[..., 3:4].astype(np.float32) / 255.0
        white = np.full_like(rgb_cut, 255)
        comp = (rgb_cut.astype(np.float32) * a + white.astype(np.float32) * (1.0 - a)).astype(
            np.uint8
        )
    else:
        comp = cut_rgba

    comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
    mets = _metrics(work, comp_bgr, alpha.astype(np.uint8))

    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    cv2.imwrite(str(model_dir / f"{stem}_comp.jpg"), comp_bgr)
    cv2.imwrite(str(model_dir / f"{stem}_alpha.png"), alpha)

    return {
        "path": str(path),
        "name": path.name,
        "model": model,
        "seconds": round(elapsed, 3),
        **mets,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True, help="Image files or directories")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["u2net", "isnet-general-use"],
        help="rembg model names",
    )
    ap.add_argument(
        "--protect",
        default="none",
        choices=["none"],
        help="Only 'none' is supported (face protect needs detector; use API confirm)",
    )
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument(
        "--ring",
        action="store_true",
        help="Apply 8%%/12px white ring before rembg (closer to prod path)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max images (0=all)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Skip >=12GB RAM guard",
    )
    args = ap.parse_args(argv)

    ram = _host_ram_gb()
    if ram and ram < 12.0 and not args.force:
        print(
            f"ERROR: host RAM {ram:.1f}GB < 12GB. Refuse rembg A/B (OOM risk). "
            "Use EC2/Docker >=12GB or pass --force.",
            file=sys.stderr,
        )
        return 2

    roots = [Path(r) for r in args.roots]
    images = _iter_images(roots)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print("ERROR: no images found", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"host_ram_gb={ram:.1f} images={len(images)} models={args.models} ring={args.ring}")
    rows: List[Dict[str, object]] = []

    for model in args.models:
        print(f"\n=== loading session: {model} ===", flush=True)
        t0 = time.perf_counter()
        session = new_session(model)
        print(f"session ready in {time.perf_counter() - t0:.1f}s", flush=True)
        for i, path in enumerate(images, 1):
            print(f"[{model}] {i}/{len(images)} {path.name}", flush=True)
            row = _run_one(path, session, model, out_dir, args.max_side, args.ring)
            rows.append(row)
            if "error" not in row:
                print(
                    f"  white_purity={row['white_purity']:.3f} "
                    f"face_core_mae={row['face_core_mae']:.2f} "
                    f"fg_frac={row['fg_frac']:.3f} "
                    f"t={row['seconds']}s",
                    flush=True,
                )

    summary: Dict[str, Dict[str, float]] = {}
    for model in args.models:
        mrows = [r for r in rows if r.get("model") == model and "error" not in r]
        if not mrows:
            continue
        summary[model] = {
            "n": float(len(mrows)),
            "mean_white_purity": float(np.mean([r["white_purity"] for r in mrows])),
            "mean_face_core_mae": float(np.mean([r["face_core_mae"] for r in mrows])),
            "mean_fg_frac": float(np.mean([r["fg_frac"] for r in mrows])),
            "mean_seconds": float(np.mean([r["seconds"] for r in mrows])),
        }

    csv_path = out_dir / "metrics.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    report = {
        "summary": summary,
        "gate": {
            "note": "isnet wins if mean_white_purity >= u2net+0.02 AND "
            "mean_face_core_mae not worse than u2net by >15%; then confirm via API",
        },
        "rows": rows,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if "u2net" in summary and "isnet-general-use" in summary:
        u, i = summary["u2net"], summary["isnet-general-use"]
        d_p = i["mean_white_purity"] - u["mean_white_purity"]
        mae_ratio = (
            i["mean_face_core_mae"] / u["mean_face_core_mae"]
            if u["mean_face_core_mae"] > 1e-6
            else 1.0
        )
        win = d_p >= 0.02 and mae_ratio <= 1.15
        print(
            f"\nDELTA white_purity={d_p:+.4f} face_core_mae_ratio={mae_ratio:.3f} "
            f"-> {'ISNET_WINS' if win else 'KEEP_U2NET_OR_SPOTCHECK'}"
        )

    print(f"\nwrote {csv_path} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
