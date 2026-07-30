"""Graph input adapters and standard quantum-device topologies.

Every public solver entry point accepts *any* of the formats below via
:func:`as_graph`; internally everything runs on a connected ``networkx.Graph``
with integer nodes ``0..n-1``, and results are mapped back to the caller's
original node labels.

Accepted input formats
----------------------
- ``networkx.Graph`` (any hashable node labels),
- iterable of edge pairs ``[(u, v), ...]``,
- square adjacency matrix (``numpy`` array or nested lists),
- adjacency dict ``{u: [v, ...]}``,
- qiskit ``CouplingMap`` (duck-typed via ``get_edges()``),
- path to an edge-list file (``.txt``/``.edgelist``: ``u v`` per line) or a
  JSON file (``{"edges": [[u, v], ...]}`` or a plain list of pairs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Hashable, List, Tuple

import networkx as nx

__all__ = [
    "as_graph",
    "lnn",
    "cycle",
    "sun",
    "sun_16q",
    "double_sun_27q",
    "heavy_hex",
    "square_lattice",
    "standard_benchmark",
]


# -- input adapters ----------------------------------------------------------


def _from_edges(edges) -> nx.Graph:
    G = nx.Graph()
    for e in edges:
        u, v = e[0], e[1]
        if u != v:
            G.add_edge(u, v)
    return G


def _from_matrix(mat) -> nx.Graph:
    import numpy as np

    A = np.asarray(mat)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got shape {A.shape}")
    G = nx.Graph()
    n = A.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] or A[j, i]:
                G.add_edge(i, j)
    return G


def _from_file(path: Path) -> nx.Graph:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        edges = obj["edges"] if isinstance(obj, dict) else obj
        return _from_edges(edges)
    # edge list: one "u v" pair per line, '#' comments allowed
    edges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError(f"bad edge-list line: {line!r}")
        u, v = parts[0], parts[1]
        u = int(u) if u.lstrip("-").isdigit() else u
        v = int(v) if v.lstrip("-").isdigit() else v
        edges.append((u, v))
    return _from_edges(edges)


def as_graph(obj) -> Tuple[nx.Graph, Dict[int, Hashable]]:
    """Normalize any supported input into ``(G, labels)``.

    ``G`` is a simple connected undirected graph with nodes ``0..n-1``;
    ``labels[i]`` is the caller's original label for internal node ``i``.

    Raises ``ValueError`` for empty or disconnected inputs (a QFT walk needs a
    connected coupling graph; solve components separately if needed).
    """
    if isinstance(obj, nx.Graph):
        H = nx.Graph()
        H.add_nodes_from(obj.nodes())
        H.add_edges_from((u, v) for u, v in obj.edges() if u != v)
    elif isinstance(obj, (str, Path)):
        H = _from_file(Path(obj))
    elif isinstance(obj, dict):
        H = _from_edges((u, v) for u, vs in obj.items() for v in vs)
        H.add_nodes_from(obj.keys())
    elif hasattr(obj, "get_edges"):  # qiskit CouplingMap
        H = _from_edges(obj.get_edges())
    else:
        try:
            import numpy as np

            arr = np.asarray(obj)
            is_matrix = arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 2
        except Exception:
            is_matrix = False
        # A square list-of-lists is ambiguous with an edge list only for 2x2;
        # treat anything square and numeric as a matrix, else as edges.
        if is_matrix:
            H = _from_matrix(obj)
        else:
            H = _from_edges(obj)

    if H.number_of_nodes() == 0:
        raise ValueError("empty graph")
    if not nx.is_connected(H):
        raise ValueError(
            f"graph must be connected ({nx.number_connected_components(H)} "
            f"components found); solve components separately"
        )

    order = sorted(H.nodes(), key=lambda v: (str(type(v)), v if isinstance(v, (int, float, str)) else str(v)))
    labels = {i: v for i, v in enumerate(order)}
    inv = {v: i for i, v in labels.items()}
    G = nx.Graph()
    G.add_nodes_from(range(len(order)))
    G.add_edges_from((inv[u], inv[v]) for u, v in H.edges())
    return G, labels


# -- standard device topologies ---------------------------------------------


def _ints(G: nx.Graph) -> nx.Graph:
    """Relabel nodes to 0..n-1 in iteration order (heavy-hex nodes are
    heterogeneous tuples, so sorting them would fail)."""
    return nx.convert_node_labels_to_integers(G, first_label=0)


def lnn(n: int) -> nx.Graph:
    """Linear-nearest-neighbour chain of ``n`` qubits."""
    return _ints(nx.path_graph(n))


def cycle(n: int) -> nx.Graph:
    """Ring of ``n`` qubits."""
    return _ints(nx.cycle_graph(n))


def sun(cycle_len: int, n_tails: int) -> nx.Graph:
    """Ring of ``cycle_len`` qubits with ``n_tails`` pendant qubits attached
    to its first ``n_tails`` ring vertices ("sun" topology)."""
    assert 0 <= n_tails <= cycle_len, "n_tails in [0, cycle_len]"
    G = nx.cycle_graph(cycle_len)
    nxt = cycle_len
    for i in range(n_tails):
        G.add_edge(i, nxt)
        nxt += 1
    return _ints(G)


def sun_16q() -> nx.Graph:
    """16-qubit IBM 'sun': ring of 12 with 4 tails (structural representative
    of the device from Khadiev et al.; exact maps come from qiskit backends)."""
    return sun(12, 4)


def double_sun_27q() -> nx.Graph:
    """27-qubit 'double sun': two (ring-10 + 3 tails) suns joined through a
    bridge qubit. Structural representative of a 27-qubit IBM device."""
    s1 = sun(10, 3)
    s2 = sun(10, 3)
    G = nx.disjoint_union(s1, s2)
    bridge = G.number_of_nodes()
    G.add_node(bridge)
    G.add_edge(0, bridge)
    G.add_edge(13, bridge)
    return _ints(G)


def heavy_hex(m: int, n: int) -> nx.Graph:
    """Heavy-hexagon lattice (IBM Eagle/Heron family): hexagonal lattice with
    every edge subdivided by a 'link' qubit. Data qubits have degree <= 3,
    link qubits degree 2. ``m, n`` are hexagon counts per axis."""
    base = nx.hexagonal_lattice_graph(m, n)
    H = nx.Graph()
    H.add_nodes_from(base.nodes())
    link_id = 0
    for u, v in base.edges():
        link = ("link", link_id)
        link_id += 1
        H.add_edge(u, link)
        H.add_edge(link, v)
    return _ints(H)


def square_lattice(rows: int, cols: int) -> nx.Graph:
    """rows x cols square grid (Falcon-era topology)."""
    return _ints(nx.grid_2d_graph(rows, cols))


def standard_benchmark(max_n: int = 60) -> Dict[str, nx.Graph]:
    """The 10-topology benchmark used throughout the project, filtered to
    ``n <= max_n``. Returns ``{name: graph}``."""
    cand: Dict[str, nx.Graph] = {
        "lnn_8": lnn(8),
        "lnn_16": lnn(16),
        "lnn_27": lnn(27),
        "cycle_16": cycle(16),
        "sun_16q": sun_16q(),
        "double_sun_27q": double_sun_27q(),
        "grid_4x4": square_lattice(4, 4),
        "grid_5x5": square_lattice(5, 5),
        "heavy_hex_1x2": heavy_hex(1, 2),
        "heavy_hex_2x2": heavy_hex(2, 2),
    }
    return {
        name: G
        for name, G in cand.items()
        if G.number_of_nodes() <= max_n and nx.is_connected(G)
    }
