"""Rendering smoke tests: files are produced, no exceptions."""
import matplotlib.pyplot as plt

import qftbuilder as qb


def test_draw_solve_result(tmp_path):
    G = qb.sun_16q()
    res = qb.solve(G, profile="fast")
    ax = qb.draw_solution(G, res)
    ax.figure.savefig(tmp_path / "solve.png", dpi=72)
    plt.close(ax.figure)


def test_draw_sub_qft_region(tmp_path):
    G = qb.heavy_hex(1, 2)
    r = qb.sub_qft(G, 8)
    ax = qb.draw_solution(G, r)
    ax.figure.savefig(tmp_path / "sub.png", dpi=72)
    plt.close(ax.figure)


def test_draw_plain_walk_and_grid_layout(tmp_path):
    G = qb.square_lattice(3, 3)
    res = qb.solve(G, profile="fast")
    ax = qb.draw_solution(G, res.walk, layout=qb.grid_layout(3, 3))
    ax.figure.savefig(tmp_path / "grid.png", dpi=72)
    plt.close(ax.figure)


def test_draw_benchmark_grid(tmp_path):
    topos = {k: v for k, v in list(qb.standard_benchmark(max_n=17).items())[:3]}
    fig = qb.draw_benchmark(topos, solver=qb.Solver(profile="fast"),
                            out_path=str(tmp_path / "bench.png"))
    plt.close(fig)
