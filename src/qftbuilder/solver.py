"""Cascade solver: every available method competes, the cheapest circuit wins.

Rungs (each optional piece degrades gracefully when its dependency is absent):

1. ``whp``   — neural simple-path rung: GATv2 scorer + verified best-of decode
   (existence-gated). A simple path is the structurally cleanest embedding.
2. ``wnshp`` — neural walk rung: beam + stochastic decodes, local-searched.
3. ``mcds``  — model-free cost-aware dominating-set walk (multi-restart).
4. ``cds``   — Guha-Khuller CDS + DFS floor: always succeeds on a connected
   graph, so the cascade never returns nothing.
5. ``cpsat`` — (profile ``max``) CP-SAT exact polish of the winner,
   warm-started, accepted only on strict improvement.

Every candidate passes the monotone-safe local search and validity checks;
the selector ranks by **exact single-sweep CNOT cost** ``2*whites + 3*moves``
(white = covered off-walk qubit, black = walk vertex; or by walk length with
``objective="length"``), with a structure preference (simple path > walk >
heuristic) breaking ties. By construction the cascade
is never worse than any single rung.

Profiles: ``fast`` (heuristics only), ``balanced`` (+ neural when torch is
installed; the default), ``max`` (+ CP-SAT polish, time-budgeted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional

import networkx as nx

from .cds import cds_dfs_path
from .cost import cnot_upper_bound, walk_cost
from .graphs import as_graph
from .walk import (
    _walk_key,
    covers,
    dominating_walk,
    improve_walk_strong,
    walk_valid,
)

__all__ = ["Solver", "SolveResult", "Certificate"]

_SOURCE_RANK = {"whp": 0, "wnshp": 1, "cpsat": 2, "mcds": 3, "cds": 4}


@dataclass
class SolveResult:
    """Outcome of :meth:`Solver.solve`, in the caller's node labels."""

    walk: List[Hashable]
    source: str
    length: int                      # walk vertices, repeats counted
    distinct: int                    # distinct walk vertices
    cost: int                        # exact one-sweep CNOTs (2w + 3m, w = whites)
    cost_bound: int                  # Theorem-4 bound 3*len*n - 2*n
    candidates: Dict[str, Dict]      # per-rung {length, distinct, cost}
    n: int
    neural_used: bool
    lower_bound: Optional[int] = None    # filled by certify()
    proven_optimal: Optional[bool] = None
    labels: Dict[int, Hashable] = field(default_factory=dict, repr=False)

    @property
    def gap(self) -> Optional[float]:
        if self.lower_bound is None or self.lower_bound <= 0:
            return None
        return self.length / self.lower_bound


@dataclass
class Certificate:
    """Length lower bound for the shortest dominating walk (sub-QFT: for the
    budgeted walk), valid even when the underlying solver timed out."""

    lower_bound: int
    proven_optimal: bool
    method: str                       # 'milp' | 'milp-dual' | 'cpsat' | 'lp'
    upper_length: Optional[int] = None

    @property
    def gap(self) -> Optional[float]:
        if self.upper_length is None or self.lower_bound <= 0:
            return None
        return self.upper_length / self.lower_bound


class Solver:
    """Load once, call :meth:`solve` many times.

    Parameters
    ----------
    profile : ``"fast" | "balanced" | "max"``
        Rung set (see module docstring). ``balanced`` is the default and uses
        the bundled neural checkpoints when torch is available.
    objective : ``"cnot" | "length" | "fidelity"``
        Candidate ranking: exact one-sweep CNOT cost (default), raw length,
        or the per-edge-error-weighted account (docs/proofs.md section 5,
        weights from :func:`qftbuilder.graphs.with_edge_errors`).
    cpsat_time : float
        Time budget for the ``max`` profile's exact polish, seconds.
    """

    def __init__(
        self,
        profile: str = "balanced",
        objective: str = "cnot",
        local_search: bool = True,
        wnshp_samples: int = 2,
        sample_temp: float = 0.7,
        multiplier: int = 8,
        length_weight: float = 1.0,
        existence_threshold: float = 0.5,
        cpsat_time: float = 60.0,
        cpsat_workers: int = 8,
        verbose: bool = False,
    ) -> None:
        if profile not in ("fast", "balanced", "max"):
            raise ValueError(f"unknown profile: {profile!r}")
        if objective not in ("cnot", "length", "fidelity"):
            raise ValueError(f"unknown objective: {objective!r}")
        self.profile = profile
        self.objective = objective
        self.local_search = local_search
        self.wnshp_samples = wnshp_samples
        self.sample_temp = sample_temp
        self.multiplier = multiplier
        self.length_weight = length_weight
        self.existence_threshold = existence_threshold
        self.cpsat_time = cpsat_time
        self.cpsat_workers = cpsat_workers
        self.verbose = verbose

        self._whp_model = None
        self._wnshp_model = None
        self._neural_ready: Optional[bool] = None

    # -- neural lazy-load ---------------------------------------------------

    def _ensure_neural(self) -> bool:
        if self.profile == "fast":
            return False
        if self._neural_ready is not None:
            return self._neural_ready
        from . import neural

        if not neural.HAVE_TORCH:
            if self.verbose:
                print("[solver] torch not installed - neural rungs off")
            self._neural_ready = False
            return False
        try:
            self._whp_model = neural.load_model("whp")
            self._wnshp_model = neural.load_model("wnshp")
            self._neural_ready = True
        except Exception as e:  # missing checkpoints etc.
            if self.verbose:
                print(f"[solver] neural rungs off ({e})")
            self._neural_ready = False
        return self._neural_ready

    # -- rungs --------------------------------------------------------------

    def _ls(self, walk, G):
        # The local search runs under the *same* objective the selector ranks
        # by, so a candidate is not shortened into a costlier walk on the way
        # to being scored (docs/proofs.md, Corollary 3.3).
        if walk and self.local_search:
            return improve_walk_strong(walk, G, objective=self.objective)
        return walk

    def _rung_whp(self, G: nx.Graph) -> Optional[List[int]]:
        from . import neural

        data = neural.graph_to_data(G)
        probs, exists = self._whp_model.predict(data)
        if float(exists.item()) < self.existence_threshold:
            return None
        path = neural.decode_whp_best(G, probs.cpu().numpy())
        if path and walk_valid(G, path) and covers(G, path):
            return path
        return None

    def _rung_wnshp(self, G: nx.Graph) -> Optional[List[int]]:
        from . import neural

        n = G.number_of_nodes()
        data = neural.graph_to_data(G)
        probs, _ = self._wnshp_model.predict(data)
        probs = probs.cpu().numpy()
        cands = [
            neural.decode_wnshp(
                G, probs, method="beam",
                multiplier=self.multiplier, length_weight=self.length_weight,
            )
        ]
        if self.wnshp_samples > 0:
            cands += neural.decode_wnshp_samples(
                G, probs, k=self.wnshp_samples, temp=self.sample_temp,
                seed=n, multiplier=self.multiplier,
                length_weight=self.length_weight,
            )
        best = None
        best_key = None
        for raw in cands:
            w = self._ls(raw, G)
            if w and walk_valid(G, w) and covers(G, w):
                key = self._cost_key(w, n, G)
                if best is None or key < best_key:
                    best, best_key = w, key
        return best

    def _rung_mcds(self, G: nx.Graph) -> Optional[List[int]]:
        w = self._ls(dominating_walk(G, objective=self.objective), G)
        if w and walk_valid(G, w) and covers(G, w):
            return w
        return None

    def _rung_cds(self, G: nx.Graph) -> Optional[List[int]]:
        w = self._ls(cds_dfs_path(G), G)
        if w and walk_valid(G, w) and covers(G, w):
            return w
        return None

    # -- selection ----------------------------------------------------------

    def _cost_key(self, walk: List[int], n: int, G: Optional[nx.Graph] = None) -> tuple:
        if self.objective == "fidelity":
            if G is None:
                raise ValueError("fidelity ranking needs the graph")
            return _walk_key(walk, G, "fidelity", region_size=n)
        cost = walk_cost(walk, n)
        if self.objective == "cnot":
            return (cost, len(walk))
        return (len(walk), cost)

    def solve(self, graph) -> SolveResult:
        """Find the best dominating walk of the given coupling graph (any
        format accepted by :func:`qftbuilder.graphs.as_graph`)."""
        G, labels = as_graph(graph)
        n = G.number_of_nodes()
        neural_used = self._ensure_neural()

        cand: Dict[str, List[int]] = {}
        if neural_used:
            p = self._rung_whp(G)
            if p is not None:
                cand["whp"] = p
            p = self._rung_wnshp(G)
            if p is not None:
                cand["wnshp"] = p
        p = self._rung_mcds(G)
        if p is not None:
            cand["mcds"] = p
        p = self._rung_cds(G)
        if p is not None:
            cand["cds"] = p
        # cds_dfs_path always succeeds on a connected graph -> cand non-empty

        best_src = min(
            cand, key=lambda s: (self._cost_key(cand[s], n, G), _SOURCE_RANK[s])
        )

        if self.profile == "max":
            try:
                from .cpsat import solve_walk_cpsat

                r = solve_walk_cpsat(
                    G, time_limit=self.cpsat_time,
                    hint_walk=cand[best_src], workers=self.cpsat_workers,
                )
                if (
                    r["walk"] is not None
                    and self._cost_key(r["walk"], n, G)
                    < self._cost_key(cand[best_src], n, G)
                ):
                    cand["cpsat"] = r["walk"]
                    best_src = "cpsat"
            except ImportError:
                if self.verbose:
                    print("[solver] ortools not installed - cpsat polish off")

        summary = {
            src: {
                "length": len(w),
                "distinct": len(set(w)),
                "cost": walk_cost(w, n),
            }
            for src, w in cand.items()
        }
        best = cand[best_src]
        return SolveResult(
            walk=[labels[i] for i in best],
            source=best_src,
            length=len(best),
            distinct=len(set(best)),
            cost=walk_cost(best, n),
            cost_bound=cnot_upper_bound(len(best), n),
            candidates=summary,
            n=n,
            neural_used=neural_used,
            labels=labels,
        )

    # -- certification ------------------------------------------------------

    def certify(
        self,
        graph,
        result: Optional[SolveResult] = None,
        time_limit: float = 120.0,
        use_cpsat: Optional[bool] = None,
    ) -> Certificate:
        """Lower-bound the shortest dominating walk length and, when a
        :class:`SolveResult` is passed, mark it proven-optimal if the bound
        meets its length. Uses the MILP branch-and-bound dual (valid on
        timeout); optionally CP-SAT (usually tighter) when ortools is present."""
        from .milp import shortest_dominating_walk_milp

        G, _ = as_graph(graph)
        length, proven, dual = shortest_dominating_walk_milp(
            G, time_limit=time_limit, return_dual=True
        )
        lb = dual if dual is not None else 1
        method = "milp" if proven else "milp-dual"
        if use_cpsat is None:
            use_cpsat = self.profile == "max"
        if use_cpsat and not proven:
            try:
                from .cpsat import solve_walk_cpsat

                hint = result and [
                    {v: k for k, v in result.labels.items()}[u] for u in result.walk
                ]
                r = solve_walk_cpsat(G, time_limit=time_limit, hint_walk=hint or None)
                if r["dual_lb"] is not None and r["dual_lb"] > lb:
                    lb, method = r["dual_lb"], "cpsat"
                if r["proven"]:
                    proven = True
            except ImportError:
                pass
        upper = result.length if result is not None else length
        cert = Certificate(
            lower_bound=lb,
            proven_optimal=bool(upper is not None and lb >= upper),
            method=method,
            upper_length=upper,
        )
        if result is not None:
            result.lower_bound = cert.lower_bound
            result.proven_optimal = cert.proven_optimal
        return cert

    def certify_sub(self, graph, k: int, upper_length: Optional[int] = None,
                    time_limit: float = 120.0) -> Certificate:
        """Certificate for the sub-QFT (budgeted-coverage) walk of budget k —
        possible at all thanks to the budgeted MILP twin of ``kwalk``."""
        from .milp import shortest_budgeted_walk_milp

        G, _ = as_graph(graph)
        length, proven, dual = shortest_budgeted_walk_milp(
            G, k, time_limit=time_limit, return_dual=True
        )
        lb = dual if dual is not None else 1
        upper = upper_length if upper_length is not None else length
        return Certificate(
            lower_bound=lb,
            proven_optimal=bool(upper is not None and lb >= upper),
            method="milp" if proven else "milp-dual",
            upper_length=upper,
        )
