# Correctness proofs for kwalk / walk optimizations

Rigorous proofs for two correctness-critical changes in `qftbuilder`. Kept
here (not yet in `paper/`) as the mathematical justification that precedes and
governs the implementation; the test suite only *confirms* these.

Notation: `G=(V,E)` connected, `n=|V|`, `Δ` the maximum degree, `N[U]` the
closed neighbourhood of a vertex set `U`, `d(u,v)` the shortest-path distance
(a non-negative **integer**, and **symmetric**, `d(u,v)=d(v,u)`, in an
unweighted connected graph).

---

## 1. The O(1)-delta 2-opt produces bit-identical output

Context: `walk._optimize_order` reorders the anchor set to minimize the
open-tour length in the shortest-path metric. The 2-opt move was evaluated by
recomputing the full length `O(p)` per candidate (`O(p^3)` per pass); the
optimization evaluates only the change `Δ(i,j)` in `O(1)` (`O(p^2)` per pass).

**Setup.** Order `a = (a_0,…,a_{L-1})`; open-tour length
```
Len(a) = Σ_{t=0}^{L-2} d(a_t, a_{t+1}).
```
A 2-opt move at `(i,j)`, `0 ≤ i < j ≤ L-1`, reverses the block `[i..j]`:
```
a' = (a_0,…,a_{i-1},  a_j,a_{j-1},…,a_{i+1},a_i,  a_{j+1},…,a_{L-1}).
```

**Lemma 1.1 (the delta is local).**
```
Len(a') − Len(a) = Δ(i,j) := [ d(a_{i-1},a_j) + d(a_i,a_{j+1}) ]
                           − [ d(a_{i-1},a_i) + d(a_j,a_{j+1}) ],
```
where the `a_{i-1}` term is dropped when `i=0` and the `a_{j+1}` term when
`j=L-1`.

*Proof.* Partition the tour edges into: (1) the left boundary `(a_{i-1},a_i)`
(present iff `i≥1`); (2) the block-internal edges
`(a_i,a_{i+1}),…,(a_{j-1},a_j)`; (3) the right boundary `(a_j,a_{j+1})`
(present iff `j≤L-2`); (4) all others. Under the reversal, group (4) is
untouched; group (2) becomes the same vertex pairs traversed in reverse, so by
**symmetry** of `d` its total is unchanged; the left boundary becomes
`(a_{i-1},a_j)` and the right becomes `(a_i,a_{j+1})`. Only the two boundary
terms change, and their difference is `Δ(i,j)`. ∎

**Lemma 1.2 (identical decisions).** For any `eps ∈ (0,1)`,
```
Len(a') < Len(a) − eps   ⟺   Δ(i,j) < 0.
```
*Proof.* `Len` is integer-valued (a sum of integers). For integers `x,y` and
`eps∈(0,1)`: `y < x − eps ⟺ y ≤ x−1 ⟺ y < x ⟺ y−x < 0`. Take `y=Len(a')`,
`x=Len(a)`, and apply Lemma 1.1. ∎

**Theorem 1 (bit-identical output).** With the same scan order over `(i,j)`,
the full-recompute implementation (accept iff `Len(a') < Len(a) − eps`,
`eps=10^{-9}`) and the delta implementation (accept iff `Δ(i,j) < 0`) produce
the identical sequence of states `a`, hence the identical output.

*Proof.* Induction on accepted moves. Scan order is identical; by Lemma 1.2
the accept/reject decision matches on every candidate; on acceptance both
apply the same reversal `a'` and update the running length to the same value
(`Len(a')`; delta: `Len(a)+Δ(i,j)=Len(a')`). So `a` evolves identically. ∎

Confirmed empirically bit-identical on 85 representative graphs
(ER/BA/WS/regular/tree/grid/triangular/heavy-hex), 0 mismatches.

---

## 2. A* returns a minimum-length budgeted walk

Context: `kwalk._astar_budgeted` finds the shortest walk `W` with
`|N[V(W)]| ≥ k` (single `k`), replacing the level BFS for that query with A*
using `h = ceil((k − coverage)/Δ)`.

**What is guaranteed.** A* returns a walk of **minimum length** `|W|` with
`|N[V(W)]| ≥ k`, i.e. the same `OPT_bud(G,k)` as the BFS. It need **not**
return the same walk: with multiple optima, A* and the BFS break ties
differently (measured: length identical on 212/212 cells, walk differs on
201/212). Length is the objective, so this suffices.

**State space.** A state `s=(v,U)` has last vertex `v` and distinct-visited
set `U`, `v∈U`. A transition takes `s=(v,U)` to `s'=(u, U∪{u})` for `u∈N(v)`,
cost 1. Start states `(v,{v})` have `g=1`. Coverage `cov(s)=|N[U]|` is a
function of `U`. Goals: `cov(s) ≥ k`. Minimize `g` (walk length in vertices).
Heuristic `h(s) = ceil( max(k−cov(s),0) / Δ )`.

**Lemma 2.1 (each step adds ≤ Δ coverage).** For any transition
`s=(v,U) → s'=(u,U∪{u})`: `cov(s') − cov(s) ≤ Δ`.

*Proof.* `u∈N(v)` and `v∈U`, so `u∈N[U]`: **u is already covered** before the
step. Hence
```
cov(s') − cov(s) = |N[U∪{u}]| − |N[U]| = |N[u] \ N[U]|.
```
Since `u∈N[U]`, `u ∉ N[u]\N[U]`, so `N[u]\N[U] ⊆ N(u)` (open neighbourhood),
giving `|N[u]\N[U]| ≤ deg(u) ≤ Δ`. ∎

**Lemma 2.2 (admissibility).** `h(s) ≤ h*(s)`, the minimum number of steps
from `s` to a goal.

*Proof.* Any path of `t` steps from `s` raises coverage by at most `tΔ`
(Lemma 2.1), so reaching `cov ≥ k` needs `cov(s)+tΔ ≥ k`, i.e.
`t ≥ (k−cov(s))/Δ`; as `t` is integer, `t ≥ ceil((k−cov(s))/Δ) = h(s)`.
Minimizing over paths, `h*(s) ≥ h(s)`. ∎

**Lemma 2.3 (consistency).** For every transition `s→s'` of cost 1,
`h(s) ≤ 1 + h(s')`.

*Proof.* Let `x=k−cov(s)`. Coverage is monotone (`cov(s')≥cov(s)`) and, by
Lemma 2.1, `cov(s')≤cov(s)+Δ`, so `x' := k−cov(s')` satisfies
`x−Δ ≤ x' ≤ x`.
- `x ≤ 0`: `h(s)=0 ≤ 1+h(s')`.
- `0 < x ≤ Δ`: `h(s)=ceil(x/Δ)=1 ≤ 1+h(s')`.
- `x > Δ`: then `x' ≥ x−Δ > 0`, and
  `h(s') = ceil(x'/Δ) ≥ ceil((x−Δ)/Δ) = ceil(x/Δ)−1 = h(s)−1`
  (the identity `ceil((x−Δ)/Δ)=ceil(x/Δ)−1` holds since `Δ` is a positive
  integer), hence `h(s) ≤ 1+h(s')`. ∎

**Theorem 2 (optimality).** A* with a consistent heuristic and a closed set
(each state finalized once) pops every state with its optimal `g`, and the
first popped goal has minimum `g` among goals — i.e. `OPT_bud(G,k)`.

*Proof.* Consistency (Lemma 2.3) makes `f=g+h` non-decreasing along any path,
so a state is popped with optimal `g` (Hart–Nilsson–Raphael, 1968). At a goal
`cov≥k ⟹ h=0 ⟹ f=g`; A* pops by increasing `f`, so the first popped goal has
the least `g` among goals. ∎

**Budget.** If the closed set exceeds the state budget, A* returns `None` and
the caller falls back to the level BFS; this affects only whether the proof
finishes in budget, never correctness. When A* returns a walk (in budget) it
is optimal by Theorem 2.

**On multiplicity of optima.** Theorem 2 guarantees the optimal *length*, not
a unique solution. Different traversal orders (A*: heap by `f`; BFS: by level,
dict insertion) select different optimal-length walks — the observed 201/212
walk differences, with length and (here) `cost_sweep` identical. This is ties,
not error: a bug would show as a *length* difference, which never occurs.

---

## 3. True-cost local search (`3L − 2d`)

Context: the local search minimizes the walk **length** `L`, while the real
objective — and what the solver already ranks candidates by — is the exact
single-sweep CNOT account `2·whites + 3·moves`. This section shows the two are
*not* the same objective, and that one family of moves improves the true cost
at **zero** length cost.

**Setup.** A sweep over a region of `s` vertices driven by a dominating walk
`W`: `L = |W|` (vertices, repeats counted), `d = |V(W)|` distinct (black)
vertices, `moves = L−1`, `whites = s − d` (every region vertex is black or
white, since `W` dominates the region).

**Lemma 3.1 (the cost is affine in `(L,d)`).**
```
cost(W) = 2(s − d) + 3(L − 1) = 3L − 2d + (2s − 3).
```
For a fixed region size `s`, therefore,
`cost(W') < cost(W)  ⟺  3L' − 2d' < 3L − 2d`.

*Proof.* Substitution into `2·whites + 3·moves`. The additive constant
`2s − 3` does not depend on `W`. ∎

**Corollary 3.2 (at equal length, more black is strictly cheaper).** If
`L' = L` and `d' > d` then `cost(W') − cost(W) = −2(d' − d) < 0`.

**Corollary 3.3 (length-minimization is not cost-monotone).** A move with
`ΔL = −1` and `Δd ≤ −2` shortens the walk yet **raises** the true cost:
`Δcost = 3ΔL − 2Δd ≥ −3 + 4 = +1 > 0`. So a pure length-LS can strictly
worsen the objective the solver is scored on. ∎

### 3a. The stitch is length-invariant

The pipeline fixes an *anchor order* `a = (a_0,…,a_{p−1})` and then stitches
consecutive anchors with shortest paths.

**Lemma 3.4 (length is stitch-invariant).** Every stitching of a fixed anchor
order by shortest paths yields the same length
`L = 1 + Σ_{t} d(a_t, a_{t+1})`, independently of *which* shortest paths are
picked.

*Proof.* The walk is the concatenation of the segments `P_t` with each shared
endpoint counted once, so `L = 1 + Σ_t (|P_t| − 1) = 1 + Σ_t d(a_t,a_{t+1})`,
using `|P_t| − 1 = d(a_t,a_{t+1})` for a shortest path. ∎

**Corollary 3.5 (free optimization).** Among all shortest-path stitchings of a
fixed anchor order, minimizing `cost` is *equivalent* to maximizing the number
of distinct vertices `d` (Lemma 3.1 with `L` constant). The length-LS leaves
this entire degree of freedom unexploited.

### 3b. Exact per-segment maximization

**Lemma 3.6 (segment DP is exact).** Fix `a`, `b` and a set `U` of
already-used vertices. Let `D(v) = d(v,b)`. The backward DP
```
best(b) = [b ∉ U],
best(v) = [v ∉ U] + max{ best(w) : w ∈ N(v), D(w) = D(v) − 1 },
```
evaluated over vertices in increasing `D`, gives `best(a) − [a ∉ U] =`
`max over all shortest a→b paths P of |V(P) \ (U ∪ {a})|`.

*Proof.* (i) A path from `a` to `b` is shortest **iff** every step decreases
`D` by exactly one: `D` drops by at most 1 per edge, and a path of length
`D(a)` must therefore drop by exactly 1 each step; conversely such a path has
length `D(a) = d(a,b)`. So the shortest `a→b` paths are exactly the paths of
the layered DAG on edges `{(v,w) : D(w) = D(v) − 1}`. (ii) Along such a path
`D` is strictly decreasing, so its vertices are pairwise distinct and
`|V(P) \ U| = Σ_{v ∈ P} [v ∉ U]` — an additive objective. (iii) Maximizing an
additive objective over paths in a DAG is solved exactly by backward DP in
topological order, here the order of increasing `D`. Subtracting `[a ∉ U]`
removes the endpoint `a`, which the previous segment already contributed. ∎

**Scope of the guarantee (stated honestly).** Lemma 3.6 is exact *per
segment*, with `U` the vertices used by earlier segments. Processing segments
left to right is a **greedy** over segments and is not guaranteed to maximize
`d` globally; maximizing `d` over the whole stitch is not claimed. This is why
every candidate is gated by Theorem 3.

**Theorem 3 (monotone safety).** The true-cost local search returns a walk
that is valid, dominates the region, and has `cost ≤ cost(input)`; if no
candidate achieves this, the input is returned unchanged.

*Proof.* The incumbent is initialized to the input. Every candidate is
accepted only after checking validity and full coverage, and only if its
`cost` is strictly smaller than the incumbent's. Hence the incumbent is at all
times valid, covering, and of non-increasing cost. ∎

---

## 4. Coarser states and Pareto pruning in `kwalk`

Context: the budgeted search stores states `(v, U)` — last vertex and the set
of distinct visited vertices — carrying `cov = N[U]` alongside. Two sound
reductions of the state space follow.

**Lemma 4.1 (coverage is a sufficient statistic).** Successor coverage and the
goal test depend on the state only through `cov`, not through `U`:
a transition to `u` yields `cov' = cov ∪ N[u]`, and the goal is `|cov| ≥ k`.

*Proof.* `N[U ∪ {u}] = N[U] ∪ N[u] = cov ∪ N[u]`; the goal `|N[U]| ≥ k` is by
definition a predicate of `cov`. ∎

**Corollary 4.2 (state abstraction).** Keying states by `(v, cov)` instead of
`(v, U)` is optimality-preserving. It is *coarser*: states with `N[U₁] =
N[U₂]`, `U₁ ≠ U₂` merge, and no state is split.

*Proof.* By Lemma 4.1 the successor function and goal test are well defined on
`(v, cov)`, so the abstracted search graph simulates the original one step for
step with the same costs; every walk in one corresponds to a walk of equal
length in the other. ∎

**Lemma 4.3 (Pareto domination).** Let `s₁=(v, cov₁, g₁)` and
`s₂=(v, cov₂, g₂)` share the last vertex `v`, with `cov₁ ⊆ cov₂` and
`g₂ ≤ g₁`. Then discarding `s₁` cannot increase the optimum.

*Proof.* Feasible continuations depend on the state only through the last
vertex `v` (a continuation is a walk starting at `v`), so `s₁` and `s₂` admit
exactly the same continuations. Let `C` be any continuation of `t` steps,
contributing a fixed vertex set whose closed neighbourhood is `X`. From `s₁`
it ends with coverage `cov₁ ∪ X` at length `g₁ + t`; from `s₂`, with
`cov₂ ∪ X ⊇ cov₁ ∪ X` at length `g₂ + t ≤ g₁ + t`. Hence if `C` reaches the
goal from `s₁` (`|cov₁ ∪ X| ≥ k`) it also reaches it from `s₂`
(`|cov₂ ∪ X| ≥ |cov₁ ∪ X| ≥ k`), at no greater length. Every solution through
`s₁` is thus matched by one through `s₂` of length `≤` it. ∎

**Remark 4.4 (partial checking stays sound).** Lemma 4.3 licenses *discarding*
dominated states; it never obliges the search to detect them. Any subset of
the dominance tests may be performed — e.g. against a bounded number of stored
states — and the result remains optimal. Only the pruning power varies.

Note that Lemma 4.3 is stated on `cov` and `g` alone, so it carries over
verbatim to the weighted objective of §5 (`g` is then a weighted cost).

**Lemma 4.5 (automorphism-orbit dedup).** Let `σ ∈ Aut(G)`. If states
`(v, cov)` and `(σv, σ(cov))` are both reachable, exploring either one alone
preserves the optimal length. *(Unweighted objective only — see the caveat
after the proof.)*

*Proof.* `σ` maps walks to walks of equal length (it preserves adjacency both
ways) and satisfies `N[σU] = σN[U]`, hence `|σ(cov)| = |cov|`: the goal test is
`σ`-invariant. So `σ` is an automorphism of the search graph preserving costs
and goals, and it maps any solution through one state to a solution of the
same length through the other. ∎

**Caveat (weights break the symmetry).** The proof uses that every step costs
the same, i.e. that `σ` preserves the objective. Under the per-edge weights of
§5 an automorphism of `G` need not satisfy `w(σu,σv) = w(u,v)` — a symmetric
lattice with asymmetric error rates is the normal case — and the orbit
reduction becomes unsound. The implementation therefore applies it only to the
unweighted search; making it weight-aware would require automorphisms of the
*weighted* graph.

**What is guaranteed.** As with A* (§2), Lemmas 4.3 and 4.5 preserve the
optimal **length**, not the identity of the returned walk: pruning a dominated
or symmetric state can discard one optimum while keeping another of equal
length. `region_proven` therefore remains meaningful; a regression would show
up as a *length* difference, which the equivalence tests check against the
unpruned search.

---

## 5. Fidelity-aware cost

Context: CNOT error rates differ per coupling by an order of magnitude on real
hardware, so the cheapest circuit by *gate count* need not be the most likely
to survive. This section defines the weighted objective, shows it is a strict
generalization of `2·whites + 3·moves`, and gives the admissible heuristic for
searching under it.

**Setup.** Each edge `e ∈ E` carries a weight `w(e) ≥ 0`, the cost of one CNOT
on that coupling. The natural choice is `w(e) = −ln(1 − err(e))`, which turns a
product of survival probabilities into a sum; any non-negative weight works
for what follows. Blacks `B = V(W)`; whites are the covered off-walk vertices.

**Definition (weighted sweep cost).**
```
cost_w(W) = 3·Σ_{t} w(W_t, W_{t+1})  +  2·Σ_{x white} min{ w(b,x) : b ∈ N(x) ∩ B }.
```
The first term prices the collect-and-move (3 CNOTs on the traversed edge),
the second the 2-CNOT collection of each white across the edge it is collected
over.

**Proposition 5.1 (strict generalization).** If `w ≡ 1` then
`cost_w(W) = 2·whites + 3·moves`, the unweighted account of §3.

*Proof.* The first sum has one term per move, each `3·1`; the second has one
term per white, each `2·1`. ∎

**Proposition 5.2 (greedy white assignment is optimal).** Assigning every white
to its cheapest black neighbour minimizes the white term, and no other
assignment can do better.

*Proof.* A white `x` is collected by exactly one black neighbour, and the total
is `Σ_x w(b(x), x)` over independent choices: no black has a capacity limit and
no term couples two whites. A sum of independent terms is minimized by
minimizing each, i.e. `b(x) = argmin_{b ∈ N(x) ∩ B} w(b,x)`, which is what the
inner `min` denotes. The white set is non-empty of black neighbours because `W`
dominates the region. ∎

**Lemma 5.3 (the metric to stitch in).** For a fixed anchor order, the
move-term of `cost_w` is minimized by joining consecutive anchors along
**minimum-weight** paths, i.e. by Dijkstra in the metric `w`.

*Proof.* The move-term is `3·Σ_t (weight of segment t)` and the segments are
chosen independently, so each is minimized separately; the minimum-weight
`a→b` walk is a minimum-weight path (weights are non-negative, so removing a
cycle never increases the total). ∎

Note the contrast with §3a: under unit weights every shortest-path stitch has
the same length and the freedom lies in *which* vertices get picked up
(Lemma 3.4); under general `w` the stitch itself carries the objective.

**Lemma 5.4 (admissible heuristic under weights).** In the budgeted search of
§2, let `w_min = min_e w(e) ≥ 0`. Then
```
h_w(s) = 3·w_min·ceil( max(k − cov(s), 0) / Δ )
```
is admissible and consistent for the objective `cost_w` restricted to moves.

*Proof.* Admissibility: by Lemma 2.1 each step raises coverage by at most `Δ`,
so at least `ceil((k−cov)/Δ)` further steps are needed (Lemma 2.2), and each
costs at least `3·w_min`; the product is a lower bound on the remaining cost.
Consistency: for a transition of cost `c = 3·w(e) ≥ 3·w_min`, the step-count
heuristic drops by at most one (Lemma 2.3), so
`h_w(s) − h_w(s') ≤ 3·w_min ≤ c`. ∎

**Degenerate case.** If `w_min = 0` then `h_w ≡ 0` and the search degrades to
Dijkstra — still correct, only uninformed. This is why zero-error couplings
should be given a small positive floor rather than exactly `0`.
