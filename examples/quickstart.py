"""qft-builder quickstart: feed a graph in, get walks, circuits, pictures.

Run from the repo root:  python examples/quickstart.py
Outputs land in examples/out/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import qftbuilder as qb

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# -- 1. Any input format works -----------------------------------------------
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (2, 7)]
res = qb.solve(edges)                       # ring with two tails
print(f"walk={res.walk}  length={res.length}  CNOT/sweep={res.cost} "
      f"[{res.source}, neural={res.neural_used}]")

# -- 2. A real device topology + certificate ---------------------------------
G = qb.heavy_hex(1, 2)                      # 21-qubit heavy-hexagon
res = qb.solve(G)
cert = qb.certify(G, res, time_limit=60.0)
print(f"heavy_hex_1x2: walk {res.length}, lower bound {cert.lower_bound}, "
      f"proven optimal: {cert.proven_optimal}")

# -- 3. Sub-QFT: the best 12-qubit region, provably --------------------------
r = qb.sub_qft(G, k=12)
print(f"sub-QFT k=12: region={r['subset']}, walk={r['k_path']} vertices, "
      f"{r['cnot']} CNOT/sweep, proven={r['region_proven']}")

# ...or every k at once from a single search:
sweep = qb.sub_qft_sweep(G, ks=[4, 8, 12, 16, 20])
print("k-sweep:", {k: v["k_path"] for k, v in sorted(sweep.items())})

# -- 4. Full QFT construction ------------------------------------------------
full = qb.build_full_qft(G)
print(f"full QFT: {full['cnot']} CNOTs over {full['cascades']} cascades "
      f"(strategy: {full['strategy']})")

# -- 5. Pictures -------------------------------------------------------------
ax = qb.draw_solution(G, res)
ax.figure.savefig(OUT / "heavy_hex_solution.png", dpi=150, bbox_inches="tight")
ax = qb.draw_solution(G, r)                 # region + walk, outside greyed
ax.figure.savefig(OUT / "heavy_hex_sub12.png", dpi=150, bbox_inches="tight")
fig = qb.draw_benchmark(qb.standard_benchmark(max_n=30),
                        out_path=str(OUT / "benchmark_grid.png"))
print(f"pictures saved to {OUT}")
