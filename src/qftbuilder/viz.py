"""Solution rendering: anchors, covered qubits, region, walk order.

Improvements over the source project's figure script:

- the walk is drawn as *directed arrows* with visit-order numbers (repeat
  visits shown as "3/7"), so non-simple walks are readable;
- works straight from a :class:`~qftbuilder.solver.SolveResult`, a
  ``sub_qft`` result dict, or a plain walk list;
- captions show the exact CNOT account (whites/moves), not only the bound;
- vertex classes follow the thesis convention: black = on the walk,
  white = covered off-walk, grey = outside the region.

Everything returns the matplotlib ``Axes`` so callers can compose figures.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from .graphs import as_graph
from .solver import SolveResult

__all__ = ["draw_solution", "draw_benchmark", "grid_layout"]

C_BLACK = "#000000"    # black vertices: on the walk (thesis convention)
C_WHITE = "#ffffff"    # white vertices: covered, off the walk
C_OUT = "#d9d9d9"      # outside the region (sub-QFT only)
C_WALK = "#1a1a1a"     # walk arrows
C_EDGE = "#bfbfbf"


def grid_layout(rows: int, cols: int) -> Dict[int, tuple]:
    """Positions for :func:`qftbuilder.graphs.square_lattice` node numbering."""
    return {i * cols + j: (j, -i) for i in range(rows) for j in range(cols)}


def _normalize(graph, solution, k):
    """-> (G internal, labels, walk internal, region internal, caption bits)."""
    G, labels = as_graph(graph)
    inv = {v: i for i, v in labels.items()}
    extra = {}
    if isinstance(solution, SolveResult):
        walk = [inv[u] for u in solution.walk]
        region = set(G.nodes())
        extra = {"cnot": solution.cost, "source": solution.source}
    elif isinstance(solution, dict):  # sub_qft / kwalk result
        walk = [inv[u] for u in (solution.get("walk") or [])]
        subset = solution.get("subset")
        region = set(inv[u] for u in subset) if subset else set(G.nodes())
        if solution.get("cnot") is not None:
            extra = {"cnot": solution["cnot"]}
        k = solution.get("k", k)
    else:  # plain walk in caller labels
        walk = [inv[u] for u in solution]
        region = set(G.nodes())
    return G, labels, walk, region, extra, k


def draw_solution(
    graph,
    solution,
    k: Optional[int] = None,
    ax=None,
    layout: Optional[Dict] = None,
    title: Optional[str] = None,
    node_size: int = 420,
):
    """Draw a dominating-walk solution over its coupling graph.

    ``solution`` may be a :class:`SolveResult`, a ``sub_qft``/``kwalk`` result
    dict, or a walk (list of the caller's node labels). ``layout`` maps the
    *caller's* labels to 2-D positions (default: Kamada-Kawai)."""
    G, labels, walk, region, extra, k = _normalize(graph, solution, k)
    n = G.number_of_nodes()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))

    if layout is not None:
        inv = {v: i for i, v in labels.items()}
        pos = {inv[u]: xy for u, xy in layout.items() if u in inv}
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=7)

    anchors = set(walk)
    cover, out = set(), set()
    for v in G.nodes():
        if v in anchors:
            continue
        if v not in region:
            out.add(v)
        elif any(u in anchors for u in G.neighbors(v)):
            cover.add(v)
        else:
            out.add(v)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE, width=1.0)
    # walk arrows with slight curvature so back-and-forth steps stay visible
    for i in range(len(walk) - 1):
        a, b = walk[i], walk[i + 1]
        ax.annotate(
            "",
            xy=pos[b], xytext=pos[a],
            arrowprops=dict(
                arrowstyle="-|>", color=C_WALK, lw=2.4, shrinkA=12, shrinkB=12,
                connectionstyle=f"arc3,rad={0.12 + 0.06 * (i % 3)}",
            ),
        )
    for nodes, color, size, edge in (
        (out, C_OUT, node_size - 160, "#999999"),
        (cover, C_WHITE, node_size - 100, "#000000"),
        (anchors, C_BLACK, node_size, "#000000"),
    ):
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, ax=ax, nodelist=list(nodes), node_color=color,
                node_size=size, edgecolors=edge, linewidths=1.2,
            )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=7,
        labels={v: labels[v] for v in anchors}, font_color="white",
    )
    rest = cover | out
    if rest:
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=7,
            labels={v: labels[v] for v in rest}, font_color="black",
        )
    # visit-order tags ("3/7" for repeat visits)
    orders: Dict[int, List[int]] = {}
    for step, v in enumerate(walk, start=1):
        orders.setdefault(v, []).append(step)
    for v, steps in orders.items():
        x, y = pos[v]
        ax.annotate(
            "/".join(map(str, steps)), (x, y),
            textcoords="offset points", xytext=(9, 9),
            fontsize=6.5, color=C_WALK, fontweight="bold",
        )

    if title is None:
        bits = [f"n={n}"]
        if k is not None:
            bits.append(f"k={k}")
        if walk:
            bits.append(f"walk={len(walk)}")
        if "cnot" in extra:
            bits.append(f"CNOT={extra['cnot']}")
        if "source" in extra:
            bits.append(extra["source"])
        title = ", ".join(bits)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return ax


def _legend(fig):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BLACK,
               markersize=11, label="black (on walk)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_WHITE,
               markeredgecolor="black", markersize=10, label="white (covered)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OUT,
               markersize=10, label="outside region"),
        Line2D([0], [0], color=C_WALK, lw=2.4, label="walk (numbers = order)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))


def draw_benchmark(
    topologies: Dict[str, "nx.Graph"],
    solver=None,
    cols: int = 3,
    out_path: Optional[str] = None,
):
    """Solve and draw a dict of topologies in one grid figure. Returns the
    matplotlib figure; saves a PNG when ``out_path`` is given."""
    from .solver import Solver

    solver = solver or Solver()
    names = list(topologies.keys())
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.6))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, name in zip(axes, names):
        G = topologies[name]
        res = solver.solve(G)
        draw_solution(G, res, ax=ax,
                      title=f"{name}: n={res.n}, walk={res.length}, "
                            f"CNOT={res.cost} ({res.source})")
    for ax in axes[len(names):]:
        ax.axis("off")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
