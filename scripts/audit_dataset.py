"""Reproducible audit of the Cheng et al. brain-tumor .mat files. Does NOT create splits or train anything.

python scripts/audit_dataset.py [--data data/raw/extracted] [--cvind data/raw/figshare/cvind.mat]
Outputs reports/manifests/{image_manifest,exact_duplicate_groups,near_duplicate_candidates,
similarity_distribution,nn_similarity}.csv and audit_summary.json
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_audit.matio import read_mat, read_cvind, LABEL_NAMES
from data_audit.hashing import raw_array_hash, canonical_image_hash, display_hash
from data_audit.folds import patient_fold_violations, verify_cvind_mapping
from data_audit.duplicates import group_by_key, pair_category, union_find_clusters
from data_audit import near

# Stage-A RETRIEVAL parameters: deliberately permissive, only decide which pairs get verified.
# They are NOT duplicate definitions.
RETR_THUMB_NCC = 0.90
RETR_PHASH_HAMMING = 10
PROVISIONAL_CLUSTER_NCC_FULL = 0.99  # provisional, for cluster labelling only
THR_GRID = [0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def _file_key(p):
    m = re.match(r"(\d+)$", Path(p).stem)
    return (0, int(m.group(1))) if m else (1, Path(p).stem)


def _load(i_path):
    i, p = i_path
    r = read_mat(p)
    if not r.readable:
        return i, None
    t = near.thumbnail(r.image)
    return i, (raw_array_hash(r.image), canonical_image_hash(r.image), display_hash(r.image),
               near.phash64(t), near.zscore_vec(t).astype(np.float32), r)


def _verify_t(args):
    k, pa, pb, tname = args
    ra, rb = read_mat(pa), read_mat(pb)
    return k, near.verify_pair(ra.image, np.ascontiguousarray(near.DIHEDRAL[tname](rb.image)))


def _verify(args):
    k, pa, pb = args
    ra, rb = read_mat(pa), read_mat(pb)
    return k, near.verify_pair(ra.image, rb.image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw/extracted"))
    ap.add_argument("--cvind", default=str(ROOT / "data/raw/figshare/cvind.mat"))
    ap.add_argument("--out", default=str(ROOT / "reports/manifests"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = {}  # summary

    # ---------- inventory ----------
    files = sorted(Path(a.data).rglob("*.mat"), key=lambda p: _file_key(p))
    S["FILES_FOUND"] = len(files)
    with Pool(a.workers) as pool:
        loaded = pool.map(_load, list(enumerate(map(str, files))), chunksize=16)
    cvind = read_cvind(a.cvind)
    S["cvind_length"] = int(len(cvind))

    rows, thumbs, phs, ok_idx = [], {}, {}, []
    recs = {}
    for i, res in loaded:
        p = files[i]
        stem = p.stem
        base = {"image_id": stem, "file_path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)}
        if res is None:
            r = read_mat(str(p))
            rows.append({**base, "readable": False, "error": r.error})
            continue
        h_raw, h_can, h_disp, ph, z, r = res
        thumbs[i], phs[i], recs[i] = z, ph, r
        ok_idx.append(i)
        rows.append({**base, "readable": True, "error": "", "shape": "x".join(map(str, r.image.shape)),
                     "dtype": str(r.image.dtype), "min": int(r.image.min()), "max": int(r.image.max()),
                     "label": r.label, "label_name": LABEL_NAMES.get(r.label, "UNKNOWN"),
                     "patient_id": r.patient_id, "mask_shape": "x".join(map(str, r.extra["mask_shape"] or ())),
                     "mat_fields": ";".join(r.extra["fields"]),
                     "raw_array_hash": h_raw, "canonical_image_hash": h_can, "display_hash": h_disp,
                     "_idx": i})
    df = pd.DataFrame(rows)
    S["FILES_READABLE"] = int(df.readable.sum())
    S["FILES_UNREADABLE"] = int((~df.readable).sum())
    S["unreadable_files"] = df.loc[~df.readable, ["image_id", "error"]].to_dict("records")

    # ---------- fold mapping (hypothesis: cvind[k-1] <-> file k.mat; tested, not assumed) ----------
    numeric = df.image_id.str.fullmatch(r"\d+")
    stems_num = df.image_id[numeric].astype(int)
    contiguous = len(stems_num) == len(df) and sorted(stems_num) == list(range(1, len(df) + 1))
    S["file_numbers_contiguous_1_to_N"] = bool(contiguous)
    if contiguous and len(df) == len(cvind) and df.readable.all():
        df["fold"] = [int(cvind[int(s) - 1]) for s in df.image_id]
        S["fold_mapping"] = verify_cvind_mapping(df.patient_id.values, df.fold.values)
    else:
        df["fold"] = np.nan
        S["fold_mapping"] = {"status": "UNRESOLVED", "reason": "file numbering not contiguous/readable or length mismatch"}
    d = df[df.readable].copy()

    # ---------- composition ----------
    S["UNIQUE_PATIENTS"] = int(d.patient_id.nunique())
    S["CLASS_COUNTS"] = {k: int(v) for k, v in d.label_name.value_counts().sort_index().items()}
    S["IMAGE_SHAPES"] = {k: int(v) for k, v in d["shape"].value_counts().items()}
    S["dtypes"] = {k: int(v) for k, v in d.dtype.value_counts().items()}
    pc = d.groupby("patient_id").size()
    S["slices_per_patient"] = {"min": int(pc.min()), "median": float(pc.median()), "max": int(pc.max()),
                               "mean": round(float(pc.mean()), 2)}
    S["patients_per_class"] = {k: int(v) for k, v in d.groupby("label_name").patient_id.nunique().items()}
    pl = d.groupby("patient_id").label.nunique()
    S["patients_with_multiple_labels"] = {k: int(v) for k, v in pl[pl > 1].items()}
    S["fold_sizes_ORIGINAL_FOLD_COUNTS"] = {int(k): int(v) for k, v in d.fold.value_counts().sort_index().items()}
    S["class_by_fold"] = {int(f): {k: int(v) for k, v in g.label_name.value_counts().sort_index().items()}
                          for f, g in d.groupby("fold")}
    S["patients_by_fold"] = {int(f): int(g.patient_id.nunique()) for f, g in d.groupby("fold")}
    viol = patient_fold_violations(d.patient_id.values, d.fold.values)
    S["PATIENT_CROSS_FOLD_VIOLATIONS"] = len(viol)
    S["patient_cross_fold_violation_ids"] = viol

    # ---------- exact duplicates ----------
    meta = d.set_index("image_id")
    def exact_groups(col):
        return group_by_key(d.image_id.tolist(), d[col].tolist())
    G = {c: exact_groups(c) for c in ("raw_array_hash", "canonical_image_hash", "display_hash")}
    S["exact_groups_by_hash"] = {c: len(g) for c, g in G.items()}
    S["exact_groups_extra_images_by_hash"] = {c: int(sum(len(x) - 1 for x in g)) for c, g in G.items()}
    S["raw_vs_canonical_groups_identical"] = G["raw_array_hash"] == G["canonical_image_hash"]
    S["display_only_groups_not_raw"] = len([g for g in G["display_hash"] if g not in G["raw_array_hash"]])
    grows = []
    for gid, g in enumerate(G["raw_array_hash"], 1):
        m = meta.loc[g]
        grows.append({"group_id": f"EX{gid:04d}", "group_size": len(g), "image_ids": ";".join(g),
                      "patient_ids": ";".join(m.patient_id), "labels": ";".join(m.label_name),
                      "folds": ";".join(map(str, m.fold.astype(int))),
                      "n_patients": m.patient_id.nunique(), "n_classes": m.label.nunique(),
                      "n_folds": m.fold.nunique(),
                      "cross_patient": m.patient_id.nunique() > 1, "cross_class": m.label.nunique() > 1,
                      "cross_fold": m.fold.nunique() > 1,
                      "also_canonical_identical": g in G["canonical_image_hash"],
                      "raw_array_hash": m.raw_array_hash.iloc[0]})
    eg = pd.DataFrame(grows, columns=["group_id", "group_size", "image_ids", "patient_ids", "labels", "folds",
                                      "n_patients", "n_classes", "n_folds", "cross_patient", "cross_class",
                                      "cross_fold", "also_canonical_identical", "raw_array_hash"])
    eg.to_csv(out / "exact_duplicate_groups.csv", index=False)
    S["EXACT_DUPLICATE_GROUPS"] = len(eg)
    S["EXACT_DUPLICATE_CROSS_FOLD_GROUPS"] = int(eg.cross_fold.sum()) if len(eg) else 0
    S["CROSS_PATIENT_EXACT_DUPLICATE_GROUPS"] = int(eg.cross_patient.sum()) if len(eg) else 0
    S["CROSS_CLASS_EXACT_DUPLICATE_GROUPS"] = int(eg.cross_class.sum()) if len(eg) else 0
    pc4 = Counter()
    for g in G["raw_array_hash"]:
        for x, y in combinations(g, 2):
            pc4[pair_category(meta.patient_id[x], meta.patient_id[y], meta.fold[x], meta.fold[y])] += 1
    S["exact_pair_categories"] = dict(pc4)
    exact_pairs = {(x, y) for g in G["raw_array_hash"] for x, y in combinations(g, 2)}

    # ---------- near duplicates: stage A retrieval ----------
    idx = [i for i in ok_idx]; pos = {i: k for k, i in enumerate(idx)}
    Z = np.stack([thumbs[i] for i in idx]); H = np.stack([phs[i] for i in idx]).astype(np.int8)
    C = Z @ Z.T
    ham = (H[:, None, :] != H[None, :, :]).sum(-1)
    iu = np.triu_indices(len(idx), 1)
    c_vals, h_vals = C[iu], ham[iu]
    sel = (c_vals >= RETR_THUMB_NCC) | (h_vals <= RETR_PHASH_HAMMING)
    A, B = iu[0][sel], iu[1][sel]
    S["near_stageA"] = {"retrieval_thumb_ncc_ge": RETR_THUMB_NCC, "retrieval_phash_hamming_le": RETR_PHASH_HAMMING,
                        "all_pairs": int(len(c_vals)), "candidate_pairs": int(sel.sum()),
                        "by_thumb_ncc_only": int((c_vals >= RETR_THUMB_NCC).sum()),
                        "by_phash_only": int((h_vals <= RETR_PHASH_HAMMING).sum())}
    ids_arr = df.loc[idx, "image_id"].tolist()
    pid_arr = df.loc[idx, "patient_id"].tolist()
    # nearest-neighbour distribution (all images, not just candidates)
    Cn = C.copy(); np.fill_diagonal(Cn, -2)
    same = np.array([[pid_arr[i] == pid_arr[j] for j in range(len(idx))] for i in range(len(idx))])
    nn = pd.DataFrame({"image_id": ids_arr, "patient_id": pid_arr,
                       "best_same_patient_thumb_ncc": np.where(same, Cn, -2).max(1),
                       "best_other_patient_thumb_ncc": np.where(~same, Cn, -2).max(1)})
    nn.loc[nn.best_same_patient_thumb_ncc < -1, "best_same_patient_thumb_ncc"] = np.nan
    nn.to_csv(out / "nn_similarity.csv", index=False)

    # geometric-robustness diagnostic: pairs that look alike only after a flip/rotation/transpose of one image
    Bt, arg = near.dihedral_best(Z)
    Bs = np.maximum(Bt, Bt.T)  # symmetric
    GEO_THR = 0.95
    gi = np.triu_indices(len(idx), 1)
    gsel = (Bs[gi] >= GEO_THR) & (C[gi] < GEO_THR)
    names = list(near.DIHEDRAL)
    geo = pd.DataFrame({"image_a": [ids_arr[i] for i in gi[0][gsel]], "image_b": [ids_arr[j] for j in gi[1][gsel]],
                        "patient_a": [pid_arr[i] for i in gi[0][gsel]], "patient_b": [pid_arr[j] for j in gi[1][gsel]],
                        "thumb_ncc_identity": C[gi][gsel], "thumb_ncc_best_transform": Bs[gi][gsel],
                        "transform_of_b": [names[arg[i, j] if Bt[i, j] >= Bt[j, i] else arg[j, i]]
                                           for i, j in zip(gi[0][gsel], gi[1][gsel])]})
    ptab = df.set_index("image_id").file_path
    with Pool(a.workers) as pool:
        gv = dict(pool.imap_unordered(_verify_t, [(k, str(ROOT / ptab[r.image_a]), str(ROOT / ptab[r.image_b]), r.transform_of_b)
                                                  for k, r in enumerate(geo.itertuples())], chunksize=8))
    for col in ("ncc_full", "ssim_full", "mad_norm"):
        geo[col + "_after_transform"] = [gv[k][col] for k in range(len(geo))]
    geo.to_csv(out / "geometric_transform_diagnostic.csv", index=False)
    S["geometric_diagnostic"] = {"thumb_ncc_ge": GEO_THR, "pairs_found_only_after_dihedral_transform": int(gsel.sum()),
                                 "pairs_identity_thumb_ncc_ge_same_thr": int((C[gi] >= GEO_THR).sum()),
                                 "max_ncc_full_after_transform": float(geo.ncc_full_after_transform.max()) if len(geo) else None,
                                 "max_ssim_full_after_transform": float(geo.ssim_full_after_transform.max()) if len(geo) else None}

    # ---------- stage B verification ----------
    paths = df.set_index("image_id").file_path
    tasks = [(k, str(ROOT / paths[ids_arr[A[k]]]), str(ROOT / paths[ids_arr[B[k]]])) for k in range(len(A))]
    with Pool(a.workers) as pool:
        ver = dict(pool.imap_unordered(_verify, tasks, chunksize=32))
    prow = []
    for k in range(len(A)):
        x, y = ids_arr[A[k]], ids_arr[B[k]]
        mx, my = meta.loc[x], meta.loc[y]
        pr = {"image_a": x, "image_b": y, "patient_a": mx.patient_id, "patient_b": my.patient_id,
              "label_a": mx.label_name, "label_b": my.label_name, "fold_a": int(mx.fold), "fold_b": int(my.fold),
              "same_patient": mx.patient_id == my.patient_id, "same_class": mx.label == my.label,
              "same_fold": mx.fold == my.fold,
              "category": pair_category(mx.patient_id, my.patient_id, mx.fold, my.fold),
              "thumb_ncc": float(C[A[k], B[k]]), "phash_hamming": int(ham[A[k], B[k]]),
              "is_exact_raw_duplicate": (x, y) in exact_pairs or (y, x) in exact_pairs, **ver[k]}
        prow.append(pr)
    P = pd.DataFrame(prow)
    if len(P):
        edges = [(pos_i, pos_j) for pos_i, pos_j in zip(
            [ids_arr.index(x) for x in P.image_a[P.ncc_full >= PROVISIONAL_CLUSTER_NCC_FULL]],
            [ids_arr.index(y) for y in P.image_b[P.ncc_full >= PROVISIONAL_CLUSTER_NCC_FULL]])]
        roots = union_find_clusters(len(ids_arr), edges)
        in_edge = {n for e in edges for n in e}
        cid = {ids_arr[n]: f"NC{roots[n]:04d}" for n in in_edge}
        P["provisional_cluster_id_ncc_full_ge_0.99"] = [cid.get(x, "") if x in cid and y in cid and cid[x] == cid[y] else ""
                                                       for x, y in zip(P.image_a, P.image_b)]
        P = P.sort_values(["ncc_full", "ssim_full"], ascending=False)
    P.to_csv(out / "near_duplicate_candidates.csv", index=False)
    S["NEAR_DUPLICATE_CANDIDATE_PAIRS"] = int(len(P))
    S["NEAR_DUPLICATE_CROSS_FOLD_CANDIDATE_PAIRS"] = int((~P.same_fold).sum()) if len(P) else 0
    S["near_candidate_pairs_non_exact"] = int((~P.is_exact_raw_duplicate).sum()) if len(P) else 0
    S["near_candidate_cross_class_pairs"] = int((~P.same_class).sum()) if len(P) else 0

    # similarity distribution: counts of NON-exact candidate pairs at labelled thresholds, by relation category
    dist = []
    Q = P[~P.is_exact_raw_duplicate] if len(P) else P
    for metric in ("ncc_full", "ssim_full"):
        for t in THR_GRID:
            sub = Q[Q[metric] >= t]
            row = {"metric": metric, "threshold": t, "pairs_total": len(sub),
                   "cross_class": int((~sub.same_class).sum())}
            for cat in ["within-patient/within-fold", "within-patient/cross-fold",
                        "cross-patient/within-fold", "cross-patient/cross-fold"]:
                row[cat] = int((sub.category == cat).sum())
            dist.append(row)
    pd.DataFrame(dist).to_csv(out / "similarity_distribution.csv", index=False)

    # ---------- compact, Git-friendly outputs (the full candidate table stays a local, git-ignored intermediate) ----------
    qs = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    srows = []
    groups = {"all_candidates": P, "same_patient": P[P.same_patient], "cross_patient": P[~P.same_patient],
              "cross_patient_cross_fold": P[(~P.same_patient) & (~P.same_fold)]}
    for gname, sub in groups.items():
        for met in ("thumb_ncc", "ncc_full", "ssim_full", "mad_norm"):
            v = sub[met].dropna()
            srows.append({"population": gname, "metric": met, "n": len(v), "min": v.min() if len(v) else np.nan,
                          **{f"q{int(q * 100):02d}": v.quantile(q) if len(v) else np.nan for q in qs},
                          "max": v.max() if len(v) else np.nan})
    for met in ("best_same_patient_thumb_ncc", "best_other_patient_thumb_ncc"):
        v = nn[met].dropna()
        srows.append({"population": "nearest_neighbour_all_images", "metric": met, "n": len(v), "min": v.min(),
                      **{f"q{int(q * 100):02d}": v.quantile(q) for q in qs}, "max": v.max()})
    pd.DataFrame(srows).round(4).to_csv(out / "similarity_summary.csv", index=False)
    keep = [c for c in P.columns if c != "provisional_cluster_id_ncc_full_ge_0.99"]
    P[keep].head(200).to_csv(out / "top_similarity_pairs.csv", index=False)  # P is sorted by ncc_full desc
    xp = P[(~P.same_patient) & (P.ncc_full >= 0.90)]
    xp[["image_a", "image_b", "patient_a", "patient_b", "label_a", "label_b", "fold_a", "fold_b", "same_class",
        "same_fold", "category", "ncc_full", "ssim_full", "mad_norm", "thumb_ncc", "phash_hamming"]].to_csv(
        out / "cross_patient_high_similarity_pairs.csv", index=False)
    S["compact_outputs"] = {"top_similarity_pairs_rows": int(min(200, len(P))),
                            "cross_patient_ncc_ge_0.90_pairs": int(len(xp))}

    df.drop(columns=["_idx"], errors="ignore").to_csv(out / "image_manifest.csv", index=False)
    (out / "audit_summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps({k: v for k, v in S.items() if k.isupper() or k.startswith("EXACT") or k.startswith("NEAR")}, indent=1, default=str))


if __name__ == "__main__":
    main()
