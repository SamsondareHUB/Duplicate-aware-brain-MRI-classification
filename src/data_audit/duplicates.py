"""Duplicate grouping and cross-patient / cross-fold / cross-class classification."""
from collections import defaultdict


def group_by_key(ids, keys):
    """Groups of size>=2 sharing an identical key. Deterministic ordering."""
    d = defaultdict(list)
    for i, k in zip(ids, keys):
        d[k].append(i)
    groups = [sorted(v) for v in d.values() if len(v) > 1]
    return sorted(groups, key=lambda g: g[0])


def pair_category(pa, pb, fa, fb):
    """Patient/fold relation of a pair -> one of four mutually exclusive categories."""
    return f"{'within' if pa == pb else 'cross'}-patient/{'within' if fa == fb else 'cross'}-fold"


def union_find_clusters(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]
