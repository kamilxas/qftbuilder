"""Neural rungs: GATv2 scorer over vertices + probability-guided decoders.

Optional dependency block (``pip install qft-builder[neural]``): a GATv2
encoder with global-context injection feeds two heads — per-vertex
P(v belongs to a shortest dominating path/walk) and a graph-level existence
classifier for the simple-path (WHP) variant. Production checkpoints from the
source project are bundled (`checkpoints/whp.pt`, `checkpoints/wnshp.pt`;
trained on synthetic families up to n=22 with exact DP labels, validated on
IBM-style topologies).

Everything torch-related is guarded: the module imports fine without torch
(``HAVE_TORCH = False``) and the solver silently skips neural rungs.

The decoders and ``refine_path`` are torch-free and operate on a
``networkx.Graph`` plus a numpy vector of vertex probabilities, so they are
importable (and testable) without the neural extra.

Feature caveat: ``node_features`` / ``graph_features`` replicate the training
pipeline bit-for-bit — do not "improve" them without retraining the bundled
checkpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .walk import coverage, covers, walk_valid

__all__ = [
    "HAVE_TORCH",
    "checkpoint_path",
    "load_model",
    "graph_to_data",
    "decode_whp",
    "decode_whp_best",
    "decode_wnshp",
    "decode_wnshp_samples",
    "refine_path",
    "is_simple_walk",
]

try:  # pragma: no cover - import guard
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAVE_TORCH = False

GRAPH_FEAT_DIM = 9
_CKPT_DIR = Path(__file__).parent / "checkpoints"


def checkpoint_path(name: str) -> Path:
    """Path to a bundled checkpoint: ``whp`` or ``wnshp``."""
    p = _CKPT_DIR / f"{name}.pt"
    if not p.exists():
        raise FileNotFoundError(f"bundled checkpoint missing: {p}")
    return p


# -- feature extraction (must match training exactly) ------------------------


def _max_cut_components(G: nx.Graph, nodes: List) -> int:
    """Largest number of components left by removing one vertex, via the
    block-cut tree (O(n+m)). >= 3 proves the graph has no Hamiltonian path."""
    n = len(nodes)
    if n <= 2:
        return 1
    blocks_per_node: dict = {}
    for block in nx.biconnected_components(G):
        for v in block:
            blocks_per_node[v] = blocks_per_node.get(v, 0) + 1
    extra = nx.number_connected_components(G) - 1
    best = 1
    for v in nodes:
        comps = extra + max(blocks_per_node.get(v, 1), 1)
        if comps > best:
            best = comps
    return best


def node_features(G: nx.Graph, nodes: List) -> np.ndarray:
    """[N, 2]: degree normalized by the graph's max degree, local clustering."""
    max_deg = max((G.degree(v) for v in nodes), default=0) or 1
    return np.array(
        [[G.degree(v) / max_deg, nx.clustering(G, v)] for v in nodes],
        dtype=np.float32,
    )


def graph_features(G: nx.Graph, nodes: List) -> np.ndarray:
    """[1, 9] graph-level features for the existence head (see source project
    ``dataset_generator._graph_features`` for the rationale of each)."""
    n = len(nodes)
    if n == 0:
        return np.zeros((1, GRAPH_FEAT_DIM), dtype=np.float32)
    degs = [G.degree(v) for v in nodes]
    m = G.number_of_edges()
    deg1 = sum(1 for d in degs if d == 1)
    deg2 = sum(1 for d in degs if d == 2)
    min_deg = min(degs) if degs else 0
    mean_deg = 2.0 * m / n
    denom = n * (n - 1)
    density = (2.0 * m / denom) if denom else 0.0
    try:
        n_artic = len(list(nx.articulation_points(G))) if n > 2 else 0
    except Exception:
        n_artic = 0
    cut_excess = min(max(_max_cut_components(G, nodes) - 2, 0), 4) / 4.0
    bip_excess = 0.0
    if n >= 3:
        try:
            if nx.is_bipartite(G):
                part_a, part_b = nx.bipartite.sets(G)
                imbalance = abs(len(part_a) - len(part_b))
                bip_excess = min(max(imbalance - 1, 0), n) / n
        except Exception:
            bip_excess = 0.0
    feats = [
        n / 50.0,
        deg1 / n,
        deg2 / n,
        min_deg / max(n - 1, 1),
        mean_deg / max(n - 1, 1),
        density,
        n_artic / n,
        cut_excess,
        bip_excess,
    ]
    return np.array([feats], dtype=np.float32)


# -- model (only defined when torch is present) ------------------------------

if HAVE_TORCH:

    class _ResGATv2Block(nn.Module):
        def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
            super().__init__()
            self.gat = GATv2Conv(
                dim, dim, heads=heads, concat=False,
                dropout=dropout, add_self_loops=True,
            )
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, edge_index):
            return self.norm(F.gelu(self.drop(self.gat(x, edge_index))) + x)

    class GATv2Encoder(nn.Module):
        """Multi-layer GATv2 with per-layer global-context injection
        ([mean_pool || max_pool] broadcast back to nodes)."""

        def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=4,
                     heads=4, dropout=0.1) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            )
            self.gat_layers = nn.ModuleList(
                [_ResGATv2Block(hidden_dim, heads, dropout) for _ in range(num_layers)]
            )
            self.ctx_projs = nn.ModuleList(
                [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim),
            )

        def forward(self, x, edge_index, batch):
            h = self.input_proj(x)
            for gat, ctx_proj in zip(self.gat_layers, self.ctx_projs):
                h = gat(h, edge_index)
                g = torch.cat(
                    [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1,
                )
                h = h + ctx_proj(g)[batch]
            return self.output_proj(h)

    class NodeScorer(nn.Module):
        """Per-vertex logit of membership in the shortest dominating path."""

        def __init__(self, node_dim: int, hidden: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 * node_dim, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, h, batch):
            g_mean = global_mean_pool(h, batch)[batch]
            g_max = global_max_pool(h, batch)[batch]
            return self.net(torch.cat([h, g_mean, g_max], dim=-1)).squeeze(-1)

    class GraphClassifier(nn.Module):
        """Graph-level existence logit; hand features bypass the pooling
        bottleneck (a no-path graph may differ by two low-degree vertices)."""

        def __init__(self, node_dim: int, graph_feat_dim: int = GRAPH_FEAT_DIM,
                     hidden: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * node_dim + graph_feat_dim, hidden), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, h, batch, graph_attr):
            g = torch.cat(
                [global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1,
            )
            return self.net(torch.cat([g, graph_attr], dim=-1)).squeeze(-1)

    class WHPNet(nn.Module):
        """Inference port of the source project's WHPNet. Constructor keeps
        the full training-time kwarg set so saved ``model_config`` dicts load
        unchanged; loss-related kwargs are ignored at inference."""

        def __init__(self, node_features=2, hidden_dim=128, embedding_dim=64,
                     num_layers=4, heads=4, dropout=0.1, node_loss_weight=1.0,
                     graph_loss_weight=1.0, graph_feat_dim=GRAPH_FEAT_DIM,
                     node_focal_gamma=2.0, graph_focal_gamma=2.0,
                     target_prefix="whp") -> None:
            super().__init__()
            self.graph_feat_dim = graph_feat_dim
            self.target_prefix = target_prefix
            self.encoder = GATv2Encoder(
                in_dim=node_features, hidden_dim=hidden_dim,
                out_dim=embedding_dim, num_layers=num_layers,
                heads=heads, dropout=dropout,
            )
            self.node_scorer = NodeScorer(embedding_dim, hidden=hidden_dim)
            self.graph_clf = GraphClassifier(
                embedding_dim, graph_feat_dim=graph_feat_dim, hidden=hidden_dim
            )

        @staticmethod
        def _batch(data):
            if getattr(data, "batch", None) is not None:
                return data.batch
            return torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        def forward(self, data):
            batch = self._batch(data)
            h = self.encoder(data.x, data.edge_index, batch)
            ga = getattr(data, "graph_attr", None)
            if ga is None:
                ga = torch.zeros((int(batch.max()) + 1, self.graph_feat_dim),
                                 device=data.x.device)
            return self.node_scorer(h, batch), self.graph_clf(h, batch, ga)

        @torch.no_grad()
        def predict(self, data):
            self.eval()
            nl, gl = self.forward(data)
            return torch.sigmoid(nl), torch.sigmoid(gl)


def load_model(which: str = "wnshp", map_location: str = "cpu"):
    """Load a bundled checkpoint (``"whp"`` or ``"wnshp"``) or a path to a
    checkpoint saved by the source project's trainer."""
    if not HAVE_TORCH:
        raise ImportError(
            "torch/torch-geometric not installed; run `pip install qft-builder[neural]`"
        )
    path = checkpoint_path(which) if which in ("whp", "wnshp") else Path(which)
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "model_config" not in ckpt:
        raise RuntimeError(f"{path}: not a WHPNet checkpoint")
    model = WHPNet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def graph_to_data(G: nx.Graph):
    """Inference-only PyG ``Data``: features + directed edge_index + graph_attr.
    Nodes must be 0..n-1 (the solver relabels first)."""
    if not HAVE_TORCH:
        raise ImportError("torch not installed")
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    x = torch.from_numpy(node_features(G, nodes))
    edges = list(G.edges())
    if edges:
        fwd_src = [idx[u] for u, v in edges]
        fwd_dst = [idx[v] for u, v in edges]
        edge_index = torch.tensor([fwd_src + fwd_dst, fwd_dst + fwd_src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=len(nodes))
    data.graph_attr = torch.from_numpy(graph_features(G, nodes))
    return data


# -- torch-free decoders -----------------------------------------------------


def _tables(G: nx.Graph) -> Tuple[List[List[int]], List[int]]:
    """(adjacency lists, closed-neighbourhood bitmasks) over nodes 0..n-1."""
    n = G.number_of_nodes()
    adj: List[List[int]] = [[] for _ in range(n)]
    cover_m = [0] * n
    for v in range(n):
        cover_m[v] = 1 << v
        for u in G.neighbors(v):
            adj[v].append(u)
            cover_m[v] |= 1 << u
    return [sorted(a) for a in adj], cover_m


def _resolve_width(n: int, beam_width: Optional[int], multiplier: int) -> int:
    if beam_width is not None:
        return beam_width
    return max(64, multiplier * n)


def _greedy_whp(n, adj, cover_m, probs):
    if n == 0:
        return []
    full = (1 << n) - 1
    if n == 1:
        return [0] if cover_m[0] == full else None
    start = int(np.argmax(probs))
    path, visited, covered = [start], {start}, cover_m[start]
    if covered == full:
        return path
    while True:
        cur = path[-1]
        best_nb, best_p = -1, -1.0
        for nb in adj[cur]:
            if nb in visited:
                continue
            if probs[nb] > best_p:
                best_p, best_nb = float(probs[nb]), nb
        if best_nb == -1:
            return None
        path.append(best_nb)
        visited.add(best_nb)
        covered |= cover_m[best_nb]
        if covered == full:
            return path
        if len(path) == n:
            return None


def _beam_whp(n, adj, cover_m, probs, width, length_weight):
    if n == 0:
        return []
    full = (1 << n) - 1
    if n == 1:
        return [0] if cover_m[0] == full else None
    EPS = 1e-9
    starts = np.argsort(-probs)[:width].tolist()
    beam = []
    best_full = None
    for s in starts:
        sc = -float(np.log(probs[s] + EPS)) + length_weight
        if cover_m[s] == full:
            if best_full is None or sc < best_full[0]:
                best_full = (sc, [s])
            continue
        beam.append((sc, s, 1 << s, cover_m[s], (s,)))
    for _ in range(n):
        if not beam:
            break
        nxt = []
        for sc, cur, vm, cm, path in beam:
            for nb in adj[cur]:
                if vm >> nb & 1:
                    continue
                new_sc = sc - float(np.log(probs[nb] + EPS)) + length_weight
                if best_full is not None and new_sc >= best_full[0]:
                    continue
                new_cm = cm | cover_m[nb]
                new_path = path + (nb,)
                if new_cm == full:
                    if best_full is None or new_sc < best_full[0]:
                        best_full = (new_sc, list(new_path))
                else:
                    nxt.append((new_sc, nb, vm | (1 << nb), new_cm, new_path))
        nxt.sort(key=lambda x: x[0])
        beam = nxt[:width]
    return None if best_full is None else best_full[1]


def _exact_whp(n, adj, cover_m, probs, top_k=18):
    """Guided bitmask DP: exact shortest simple dominating path within the
    top-K most probable vertices (all vertices when n <= top_k)."""
    if n == 0:
        return []
    full = (1 << n) - 1
    if n == 1:
        return [0] if cover_m[0] == full else None
    if n <= top_k:
        allowed = list(range(n))
    else:
        order = np.argsort(-probs, kind="stable")
        allowed = sorted(order[:top_k].tolist())
    k = len(allowed)
    g2l = {g: i for i, g in enumerate(allowed)}
    adj_local: List[List[int]] = [[] for _ in range(k)]
    for li, gi in enumerate(allowed):
        for u in adj[gi]:
            lu = g2l.get(u)
            if lu is not None:
                adj_local[li].append(lu)
    size = 1 << k
    dp = [[False] * k for _ in range(size)]
    par = [[-1] * k for _ in range(size)]
    for s in range(k):
        dp[1 << s][s] = True
    for mask in range(1, size):
        for v in range(k):
            if not dp[mask][v] or not ((mask >> v) & 1):
                continue
            for u in adj_local[v]:
                if (mask >> u) & 1:
                    continue
                nm = mask | (1 << u)
                if not dp[nm][u]:
                    dp[nm][u] = True
                    par[nm][u] = v
    by_pc: List[List[int]] = [[] for _ in range(k + 1)]
    for mask in range(1, size):
        by_pc[bin(mask).count("1")].append(mask)
    for length in range(1, k + 1):
        for mask in by_pc[length]:
            covered = 0
            mm = mask
            while mm:
                lo = mm & -mm
                covered |= cover_m[allowed[lo.bit_length() - 1]]
                if covered == full:
                    break
                mm ^= lo
            if covered != full:
                continue
            for v in range(k):
                if dp[mask][v] and ((mask >> v) & 1):
                    out = []
                    cm_, cv = mask, v
                    while cv != -1:
                        out.append(cv)
                        p = par[cm_][cv]
                        cm_ ^= 1 << cv
                        cv = p
                    out.reverse()
                    return [allowed[i] for i in out]
    return None


def decode_whp(
    G: nx.Graph,
    probs: np.ndarray,
    method: str = "beam",
    beam_width: Optional[int] = None,
    multiplier: int = 8,
    top_k: int = 18,
    length_weight: float = 1.0,
) -> Optional[List[int]]:
    """Decode a *simple* dominating path from per-vertex probabilities.
    Methods: ``greedy`` / ``beam`` (score = Σ -log p + β·len) / ``exact``
    (guided bitmask DP). Falls back to greedy when beam/exact fail."""
    n = G.number_of_nodes()
    if n == 0:
        return None
    if n == 1:
        return [0]
    adj, cover_m = _tables(G)
    probs = np.asarray(probs, dtype=np.float64)
    width = _resolve_width(n, beam_width, multiplier)
    if method == "greedy":
        return _greedy_whp(n, adj, cover_m, probs)
    if method == "beam":
        p = _beam_whp(n, adj, cover_m, probs, width, length_weight)
        return p if p is not None else _greedy_whp(n, adj, cover_m, probs)
    if method == "exact":
        p = _exact_whp(n, adj, cover_m, probs, top_k=top_k)
        return p if p is not None else _greedy_whp(n, adj, cover_m, probs)
    raise ValueError(f"unknown WHP decode method: {method!r}")


def is_simple_walk(G: nx.Graph, path: List[int]) -> bool:
    if not path:
        return False
    if len(set(path)) != len(path):
        return False
    if len(path) == 1:
        return True
    return all(G.has_edge(path[i], path[i + 1]) for i in range(len(path) - 1))


def _trim_ends(G, path):
    changed = True
    while changed and len(path) > 1:
        changed = False
        if covers(G, path[1:]):
            path = path[1:]
            changed = True
        if len(path) > 1 and covers(G, path[:-1]):
            path = path[:-1]
            changed = True
    return path


def _shortcut(G, path):
    i = 0
    while i < len(path) - 2:
        best_j = -1
        for j in range(len(path) - 1, i + 1, -1):
            if G.has_edge(path[i], path[j]):
                cand = path[: i + 1] + path[j:]
                if covers(G, cand):
                    best_j = j
                    break
        if best_j != -1:
            path = path[: i + 1] + path[best_j:]
        i += 1
    return path


def refine_path(G: nx.Graph, path: Optional[List[int]], max_iters: int = 8):
    """Greedy verified shortening of a simple dominating path (trim ends +
    shortcut detours), preserving both invariants. Never longer than input."""
    if not path or not is_simple_walk(G, path) or not covers(G, path):
        return None
    for _ in range(max_iters):
        before = len(path)
        path = _trim_ends(G, path)
        path = _shortcut(G, path)
        path = _trim_ends(G, path)
        if len(path) == before:
            break
    return path


_WHP_CONFIGS = [
    dict(method="exact", top_k=18),
    dict(method="beam", multiplier=64, length_weight=3.0),
    dict(method="beam", multiplier=32, length_weight=2.0),
    dict(method="beam", multiplier=8, length_weight=1.0),
]


def decode_whp_best(G: nx.Graph, probs: np.ndarray) -> Optional[List[int]]:
    """Verified best-of decode: shortest covering simple path over several
    decoder configs, each polished by :func:`refine_path`."""
    best = None
    for c in _WHP_CONFIGS:
        try:
            p = decode_whp(G, probs, **c)
        except Exception:
            p = None
        if p and is_simple_walk(G, p) and covers(G, p):
            p = refine_path(G, p)
            if best is None or len(p) < len(best):
                best = p
    return best


# -- WNSHP (walk) decoders ---------------------------------------------------


def _bfs_coverage_route(src, covered, adj, cover_m):
    notcov = ~covered
    if cover_m[src] & notcov:
        return []
    prev = {src: -1}
    from collections import deque

    q = deque([src])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w in prev:
                continue
            prev[w] = u
            if cover_m[w] & notcov:
                out = [w]
                p = u
                while p != src:
                    out.append(p)
                    p = prev[p]
                out.reverse()
                return out
            q.append(w)
    return None


def _greedy_wnshp(n, adj, cover_m, probs):
    if n == 0:
        return []
    full = (1 << n) - 1
    if n == 1:
        return [0] if cover_m[0] == full else None
    start = int(np.argmax(probs))
    walk = [start]
    covered = cover_m[start]
    ops = 1
    while covered != full:
        cur = walk[-1]
        best_nb, best_key = -1, (0, -1.0)
        for nb in adj[cur]:
            gain = (cover_m[nb] & ~covered).bit_count()
            key = (gain, float(probs[nb]))
            if key > best_key:
                best_key, best_nb = key, nb
        if best_nb != -1 and best_key[0] > 0:
            walk.append(best_nb)
            covered |= cover_m[best_nb]
            ops += 1
        else:
            route = _bfs_coverage_route(cur, covered, adj, cover_m)
            if not route:
                return None
            for w in route:
                walk.append(w)
                covered |= cover_m[w]
                ops += 1
                if covered == full:
                    break
        if ops > 4 * n * n:
            return None
    return walk


def _beam_wnshp(n, adj, cover_m, probs, width, length_weight):
    if n == 0:
        return []
    full = (1 << n) - 1
    if n == 1:
        return [0] if cover_m[0] == full else None
    EPS = 1e-9
    starts = np.argsort(-probs)[:width].tolist()
    beam = []
    best_full = None
    for s in starts:
        sc = -float(np.log(probs[s] + EPS)) + length_weight
        if cover_m[s] == full:
            if best_full is None or sc < best_full[0]:
                best_full = (sc, [s])
            continue
        beam.append((sc, s, cover_m[s], 1 << s, (s,)))
    for _ in range(2 * n):
        if not beam:
            break
        cand: Dict = {}
        for sc, cur, cm, vm, path in beam:
            for nb in adj[cur]:
                first = not ((vm >> nb) & 1)
                add = -float(np.log(probs[nb] + EPS)) if first else 0.0
                new_sc = sc + add + length_weight
                if best_full is not None and new_sc >= best_full[0]:
                    continue
                new_cm = cm | cover_m[nb]
                new_path = path + (nb,)
                if new_cm == full:
                    if best_full is None or new_sc < best_full[0]:
                        best_full = (new_sc, list(new_path))
                    continue
                new_vm = vm | (1 << nb)
                key = (nb, new_cm, new_vm)
                old = cand.get(key)
                if old is None or new_sc < old[0]:
                    cand[key] = (new_sc, nb, new_cm, new_vm, new_path)
        beam = sorted(cand.values(), key=lambda x: x[0])[:width]
    return None if best_full is None else best_full[1]


def decode_wnshp(
    G: nx.Graph,
    probs: np.ndarray,
    method: str = "beam",
    beam_width: Optional[int] = None,
    multiplier: int = 8,
    length_weight: float = 1.0,
) -> Optional[List[int]]:
    """Decode a dominating *walk* (repeats allowed) from vertex probabilities.
    The beam scores Σ -log p over first visits + β per step, so real walk
    length (the CNOT driver) is optimized, not just the anchor set. Greedy
    fallback always covers on a connected graph."""
    n = G.number_of_nodes()
    if n == 0:
        return None
    if n == 1:
        return [0]
    adj, cover_m = _tables(G)
    probs = np.asarray(probs, dtype=np.float64)
    width = _resolve_width(n, beam_width, multiplier)
    if method == "greedy":
        return _greedy_wnshp(n, adj, cover_m, probs)
    if method == "beam":
        p = _beam_wnshp(n, adj, cover_m, probs, width, length_weight)
        return p if p is not None else _greedy_wnshp(n, adj, cover_m, probs)
    raise ValueError(f"unknown WNSHP decode method: {method!r}")


def decode_wnshp_samples(
    G: nx.Graph,
    probs: np.ndarray,
    k: int = 4,
    temp: float = 0.7,
    seed: Optional[int] = None,
    multiplier: int = 8,
    length_weight: float = 1.0,
) -> List[List[int]]:
    """K stochastic decodes via Gaussian logit noise — beam width saturates,
    so diversity comes from different probability basins. Caller takes the
    shortest covering candidate after local search."""
    n = G.number_of_nodes()
    if n <= 1 or k <= 0:
        return []
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(k):
        noisy = logit + temp * rng.standard_normal(size=logit.shape)
        pp = 1.0 / (1.0 + np.exp(-noisy))
        w = decode_wnshp(G, pp, method="beam", multiplier=multiplier,
                         length_weight=length_weight)
        if w:
            walks.append(w)
    return walks
