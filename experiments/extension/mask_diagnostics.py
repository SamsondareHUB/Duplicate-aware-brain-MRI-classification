"""Tumor-region diagnostic inference on already-trained extension checkpoints (data/ckpt/*.pt). ANALYSIS-TIME INPUT VARIANTS ONLY:
ground-truth masks are never used in training or in standard inference. Variants are built on the frozen 224x224 [0,1] input
before the usual 3-channel ImageNet normalisation:
  full               unchanged
  tumor_only         X * M                      (everything outside the official mask set to 0; sharp mask border artifact)
  tumor_dil_only     X * dilate(M, 7 px)        (tumor + thin context margin)
  bg_only            X * (1 - dilate(M, 7 px))  (tumor + margin zeroed: no tumor pixels remain)
  zero_tight         X * (1 - M)                (tumor zeroed, tight)
  zero_shifted       X * (1 - shift(M))         (ARTIFACT CONTROL: same-area/shape hole at a displaced non-tumor location)
Usage: python mask_diagnostics.py RUN_ID [RUN_ID ...]  ->  results/extension/mask/{run_id}_variants.csv.gz"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments/extension"))
from leakage.preprocess import load_all
from run_sanity import to_input
from ext_runner import build_model
from ext_specs import get_spec

OUT = ROOT / "results/extension/mask"
VARIANTS = ("full", "tumor_only", "tumor_dil_only", "bg_only", "zero_tight", "zero_shifted")
SHIFTS = [(0, 80), (0, -80), (80, 0), (-80, 0), (80, 80), (-80, -80), (80, -80), (-80, 80), (0, 56), (0, -56), (56, 0), (-56, 0)]


def dilate(m, r=7): return (F.max_pool2d(m[:, None], 2 * r + 1, 1, r)[:, 0] > 0).float()


def shifted(m, dil):
    out = torch.zeros_like(m); ov = np.zeros(len(m))
    for i in range(len(m)):
        best = None
        for dy, dx in SHIFTS:
            s = torch.roll(m[i], (dy, dx), (0, 1))
            o = float((s * dil[i]).sum() / s.sum().clamp(min=1))
            if best is None or o < best[0]: best = (o, s)
            if o == 0: break
        out[i] = best[1]; ov[i] = best[0]
    return out, ov


def variants(X, M):
    m = M.float(); dil = dilate(m); sh, ov = shifted(m, dil)
    return {"full": X, "tumor_only": X * m, "tumor_dil_only": X * dil, "bg_only": X * (1 - dil),
            "zero_tight": X * (1 - m), "zero_shifted": X * (1 - sh)}, ov


def run(run_id, ids, Xt, Mt, pos):
    out = OUT / f"{run_id}_variants.csv.gz"
    if out.exists(): return
    spec = get_spec(run_id)
    model = build_model(spec["model"]); model.load_state_dict(torch.load(ROOT / f"data/ckpt/{run_id}.pt", map_location="cpu"))
    model.to("mps").eval()
    idx = np.array([pos[i] for i in spec["test_ids"]])
    rows = []
    with torch.no_grad():
        for b in range(0, len(idx), 64):
            bi = idx[b:b + 64]
            V, ov = variants(Xt[bi], Mt[bi])
            for name in VARIANTS:
                pr = torch.softmax(model(to_input(V[name]).to("mps")).float().cpu().double(), 1).numpy()
                rows.append(pd.DataFrame({"run_id": run_id, "variant": name, "image_id": np.array(spec["test_ids"])[b:b + 64],
                                          "shift_overlap": ov, **{f"prob_{k}": pr[:, k] for k in range(3)}}))
    pd.concat(rows).to_csv(out.with_suffix(".tmp.gz"), index=False, compression="gzip"); out.with_suffix(".tmp.gz").replace(out)
    del model; torch.mps.empty_cache()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv")
    pos = {int(i): k for k, i in enumerate(ids)}
    Xt = torch.from_numpy(X); Mt = torch.from_numpy(np.load(ROOT / "data/interim/masks224.npy"))
    for rid in sys.argv[1:]:
        run(rid, ids, Xt, Mt, pos); print("mask-diag done", rid, flush=True)
