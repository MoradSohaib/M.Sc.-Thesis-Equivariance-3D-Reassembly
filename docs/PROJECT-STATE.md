# Where the project stands

Orientation document: everything decided, measured and built so far — the data
pipeline, what the dataset turned out to contain, each design choice and the
evidence behind it, and what the model still needs.

`docs/design-notes.md` holds the technical detail; this page is the layer above it.
`docs/figure-review.md` reviews the GAT / VN / V-GAT explanatory posters — the
V-GAT one specifies a layer that is **not** equivariant, so read that before
implementing anything from it.

| | |
|---|---|
| Pipeline | **built** — 596 tests, clean under `-W error` |
| Dataset pass | 1,096,825 fragments across 1,442 objects |
| Fracture surface | 10.6% of vertices, dataset-wide |
| Model | **built and verified** — equivariance checked numerically in float64 |
| Training | **built** — loops, checkpoints, schedule, metrics; 1, 2 or N GPUs |
| Trained result | **none yet** — nothing has been run on the real dataset |
| First preflight | found a train/val split overlap and an OOM; both fixed |
| Second preflight | found the memory ceiling: cross-attention was 7.2 GB of 15.6 GB, now 0.6 GB |
| Third preflight | `batch_size=2` fits at 76%; preflight itself is now tested end to end |
| First real run | 13 min on 400 objects — caught an embedding loss minimised by collapse |
| Two full epochs | rotation flat at chance after 3,536 steps; OOM guard, DDP skip and both time projections fixed |
| Audit against the earlier design | seven defects found and fixed, two experiment controls added — see §10 |
| Multi-GPU rework | two silent faults (replica drift, collective desync) and five more fixed; stage two, fixed-length epochs, OOM retry, failure tracking, tools — see §11 |
| Thesis v6 | the largest fragment as the anchor: metrics always, loss by default; re-scores My Thesis Work checkpoints — see §12 |
| Benchmark range | `--max_fragments 20` trains and scores on GARF's 2–20 pieces; off by default — see §13 |
| Hub mirror | optional `--hf_repo_id/--hf_local_dir/--hf_token`: files pushed every epoch, pulled into a fresh folder — see §14 |
| Thesis v7 | matched rotations placed from the pair fits the chain agrees with, the anchor held, not one solve over every match; `--evaluate --jitter/--drop` without shared break vertices — see §17 |
| Thesis v7, no head | the rotation head removed: trained on the embedding term alone, every rotation fitted from the matches, the four geometric terms scores only, `best.pt` on validation acc@5; `--split all`, `--predictions` — see §18 |
| Above the benchmark range | `--min_fragments 21 --max_fragments 0` scores the patterns 2–20 leaves out; the two runs score every pattern once — see §19 |
| Other datasets | Fractura and Fantastic Breaks converted into Breaking Bad's layout — cleaned, scaled, thinned to the training density unless the pieces share their breaks — and evaluated unchanged — see §20 |
| Breaks that share no vertex | the embedding term's positives by distance (`--contact_radius`), each training piece remeshed on its own (`--remesh`) and its breaks opened as the scans' are (`--remesh_gap`), a new run from a checkpoint's weights (`--init_from`), `--evaluate --remesh`; one fine-tune, before the jitter's labels were fixed: `fractura_real` still at chance; the opening not yet trained — see §21 |
| Sim to real (V8) | true breaks and the oracle, isotropic remeshing, worn/rippled/bent breaks, a fracture head, consistency, fragment context; the first two V8 runs stayed at chance: the embedding term drew its anchors per pass (fixed: per scene), and the stored scenes taught matching by mesh (now every training scene is remeshed, `--val_remesh`) — see §22 |

---

## 1 · The question

Given the fragments of a broken object in arbitrary poses, recover the rigid
transform that puts each one back. The state of the art, GARF, reaches 6.1°
rotation RMSE using 4×H100 for 72 hours. The thesis asks whether **equivariance
can substitute for scale**: a network that is SO(3)-equivariant by construction
never has to learn the same geometry twice under different rotations, so it
should need far less data and compute. Target hardware is 2×T4.

Two commitments follow, and everything below serves them.

**Equivariance.** A layer satisfying `L(R·x) = R·L(x)` as an algebraic property
gets rotation for free, instead of learning it from augmented data.

**Two stages.** Rotation is learned; translation is *solved*. Once orientations
are known, placement is a geometric optimisation over matched interface points —
it does not need a network at all.

---

## 2 · The data pipeline

One scene directory holds an intact fine mesh plus ~80 fracture modes, each a
labelling of mesh cells into pieces. `SceneReader` parses the mesh and the sparse
cell matrix *once* and reuses them across every mode, which is the single largest
saving on a full pass.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="pipeline_dark.svg">
  <img alt="One scene is read once into fragments in their assembled frame; coincidence labels are computed there before any perturbation; the fragments then split into a ground-truth copy and a randomly perturbed copy that share one topology; only the perturbed copy is standardised and fed to the network, whose predicted rotation is applied to the centred non-standardised copy and compared against the ground truth." src="pipeline.svg" width="560">
</picture>

The two copies share one topology and differ only in coordinates — the losses
match vertices, edges and normals **by index**, so a second independently built
copy would compare unrelated rows without raising anything.

### Two bugs already caught here

Both were silent: right shapes, finite numbers, a loss that still descends.

- **Perturbing before computing coincidence** gives every fragment an empty
  fracture surface. Coincidence is defined in the assembled frame; once
  `diffuse_fragments` has scattered the pieces, nothing coincides.
- **Building the two copies independently** lets `resolve_duplicated_faces`,
  which reorders faces lexicographically, produce two different vertex orderings.
  Assert `gt.edge_index == diffused.edge_index` in the collate function.

---

## 3 · What the dataset actually contains

A full pass over 1,442 objects and 1,096,825 fragments, using the **coincidence**
ground truth — a vertex is fracture surface exactly when another fragment has a
vertex at the same point, decidable because Breaking Bad stores fragments in
their assembled frame.

| Quantity | Median | Mean | Max |
|---|---|---|---|
| Vertices per fragment | 296 | 1,550 | 83,039 |
| Fracture vertices | 125 | 164 | 3,587 |
| Fragments per scene | 3 | 8 | 99 |

Dataset-wide, the fracture surface is **10.6% of vertices, 6.2% of faces, 7.6% of
edges**.

Two reduction numbers are reported because they answer different questions and
diverge sharply: the **per-fragment mean** (49.9% of vertices removed) weights
every fragment equally, while the **dataset aggregate** (89.4%) weights by size.
Most fragments are small and small fragments are mostly fracture surface, so the
typical fragment loses half its vertices while the dataset loses nine tenths.

### The graph is far cheaper than expected

Because the true fracture surface is small, connecting every fracture vertex to
every fracture vertex of every *other* fragment is affordable with no reduction
at all:

| Fragments | Cross-fragment pairs | vs GARF's 6-layer stack |
|---|---|---|
| 3 (median scene) | 46,875 | 0.001× |
| 8 (mean scene) | 753,088 | 0.01× |
| 20 | 2,968,750 | 0.04× |
| 99 (largest) | 75,796,875 | 1.01× |

Cost is `7812.5·n(n−1)`, crossing GARF's whole stack at n ≈ 99 — exactly the
largest scene in the dataset — and staying under 10% up to n ≈ 32.

**So reduction is not needed for cost.** It was designed against the dihedral
mask, which over-labels by 3.9×; the real numbers make the unreduced graph
viable. Sampling is still used — not to make the graph affordable, but to make
the inter-fragment connection count *fixed and predictable* rather than a
function of whatever mesh happens to arrive.

---

## 4 · Features and preprocessing

Vector Neuron features are shaped `(C, 3)`; rotation acts on the last axis, so
`x → x Rᵀ`. Anything without a direction cannot live in that tensor.

| Level | Feature | Type |
|---|---|---|
| Node | centred coordinate + vertex normal | `(2, 3)` vector |
| Edge | two adjacent face normals + relative position `p_src − p_dst` | `(3, 3)` vector |
| Fragment | scale | invariant **scalar** |

Midpoint and edge length are dropped from the earlier design. Scale conditions
the vector stream through a gate rather than joining it:

```python
gain = softplus(MLP([Gram(x), log_scale]))   # invariant
x    = gain[..., None] * x                    # still equivariant
```

Use `log(scale)`, standardised. Fragments span 4 to 83,039 vertices, so raw scale
is heavy-tailed and would dominate whatever it is concatenated with. GARF applies
a positional encoding for the same reason.

### What may cross between fragments — the constraint that shapes the design

Each fragment is perturbed by its **own** rotation, so the label for fragment
*i* does not change when fragment *j* lands at a different angle. The
prediction must not either. Writing `A` for a rotation applied to one
fragment's input, fragment *i*'s representation has to satisfy both lines:

```
H_i(P_1, …, P_i A, …, P_N) = H_i(…) · A
H_i(P_1, …, P_j A, …, P_N) = H_i(…)          (j ≠ i)
```

Equivariant to its own pose; **invariant to every other fragment's**. This is
equation 7 of *Leveraging SE(3) Equivariance for Learning 3D Geometric Shape
Assembly*, and it is why their correlation module is `C_ij = G_j · F_i` — an
*invariant* matrix from the sender multiplying the *receiver's own* equivariant
features.

Note what this does **not** say. It is not "no vectors between fragments" —
`nn/cross.py` is a Vector Neuron layer throughout: `(T, C, 3)` in, `(T, C, 3)`
out, equivariant on the spatial axis at every step. The constraint is on what a
message may *depend on*, and within it the message is as rich as it can be: a
full `(C/H, C/H)` **channel-mixing matrix** per head, read off the sender's
invariants and applied to the receiver's own vector features. The sender
re-mixes the receiver's channels arbitrarily; what it cannot do is contribute a
direction of its own, because a direction carries an orientation.

```
logits_ij = ⟨q(inv_i), k(inv_j)⟩ / √d           invariant to both poses
out_i     = x_i + ( Σ_j α_ij G_j ) x_i          G_j invariant, x_i equivariant
```

Because `G` is *linear* in those invariants, `Σ_j α_ij G(v_j) = G(Σ_j α_ij v_j)` —
the sum runs on the small invariant vectors and the matrix is built once per
query. Not cosmetic: a matrix per *pair* would be gigabytes at the two million
pairs a large scene reaches, against a couple of megabytes per token.

The scores cannot be vector inner products either. `⟨W_q x_i, W_k x_j⟩` becomes
`⟨A_i a, A_j b⟩` under independent perturbations, which depends on the relative
rotation — the noise. Same argument, same conclusion.

Nothing is lost by the restriction: the relative orientation of two arbitrarily
tumbled fragments *is* the perturbation, not the object. Shape is the signal and
shape survives.

Measured: own-pose equivariance 8.9e-16, other-pose invariance 8.9e-16 against
outputs of order 1, and the mixing matrix is full rank with off-diagonal
magnitude 0.8× the diagonal — a real channel mix, not a per-channel gain.

### Edge normal ordering — implemented and tested

With midpoint and length gone, `(n₁, n₂)` is most of an edge's signal, so a
swapped pair is a silently wrong input. The order comes from the sign of
`(n₁ × n₂) · d` — invariant under proper rotation because `det R = 1`, and
ambiguous only when the normals are parallel, which is exactly when both slots
hold the same vector. Face-array order would *not* be stable; `test_orientation.py`
asserts that.

---

## 5 · Decisions, and the evidence for each

| Question | Decision | Why |
|---|---|---|
| Which fracture labels? | **Coincidence** for measurement; **dihedral** for the first training run | Coincidence is exact but training-time only. Dihedral over-labels 3.9× (precision 0.24) yet is available at inference, so it keeps train and test consistent. Coincidence is how you measure the gap. |
| Gate or feature? | **Gate on the token pool, not on the graph** | All vertices stay nodes with full features and mesh edges; the mask only decides which vertices may be *sampled* as cross-fragment tokens. This sits between the two extremes — GARF's pure-feature route stays available via `eligible_mask`, and the ablation is one argument. |
| Graph reduction? | **FPS sampling of the masked vertices** | Not needed for cost (unreduced is 0.001× GARF at the median scene) but chosen for a *fixed, predictable* inter-fragment connection count. `mode="sample"` selects deterministically and rotation-invariantly. |
| How is the token budget capped? | **One number: `total=2048` per scene, no per-fragment cap** | The total alone bounds the graph — the pair count is `(T² − Σt_i²)/2 < T²/2` for *any* split, so 2048 can never exceed 0.03× GARF's stack however lopsided. A cap does not tighten that by a single pair, and it destroys the area proportionality: a head statue with both ears, the nose and hair broken off would hand the head — carrying four mating surfaces — the same 128 tokens as one ear. An earlier version of this row argued for both limits on the grounds that a total alone starves the tail; true, but a cap cannot cure starvation, and the fix is a larger total. |
| Standardise how? | **Per scene**, not per fragment — *implemented, default* | Mating surfaces are the same size in world units. Per-fragment normalisation destroys that and asks the network to undo it from the scale feature. `normalize_fragments(fragments)` defaults to `mode="scene"`; `mode="fragment"` is GARF's convention. |
| Which vertices are nodes? | **All of them** | The fracture mask gates only which vertices may be *sampled as cross-fragment tokens*. Every vertex keeps its features and its mesh edges. |
| Which sampler? | **Farthest-point**, not dart-throwing | GARF uses Poisson-disk. FPS gives approximately the same blue-noise coverage on a fixed point set but is deterministic and exactly rotation-invariant, so the same fragment selects the same vertices in every pose. Dart-throwing depends on an RNG stream unrelated to the geometry. |
| Which distance? | **Geodesic** — along the fracture surface, Dijkstra on edges with both endpoints on the break | A fracture surface is a thin strip wrapping a curved fragment; straight-line distance measures *through the material*, so two points on opposite arms of a folded strip read as neighbours and tokens bunch. Measured on a folded strip, geodesic roughly doubles the minimum separation at tight budgets. It is also *cheaper* on large fragments (7.9 ms vs 13.4 ms at 10k vertices) and handles disconnected fracture patches with no special case. `metric="euclidean"` stays as the ablation. |
| Sample where? | **Inside each fragment**, never over the pooled scene | At input time the fragments are in arbitrary perturbed poses, so one global pass would pick points by where the pieces landed — a different sample per perturbation, and no equivariance left. |
| Labels for the first run? | **Dihedral** | Less accurate than coincidence, but it is what exists at inference — so train and test see the same labelling, with no distribution shift to misattribute later. |
| Rotation output? | **6D → Gram-Schmidt** | Two equivariant 3-vectors orthonormalised into a proper rotation. Quaternion regression is not equivariant under a linear map from vector features. |
| Geodesic loss? | **atan2**, not arccos | `arccos`'s gradient diverges as the model converges, crowding out every other term. Clamping turns that into an error floor instead. |
| Embedding loss? | **Centroid variance** | The design document's `‖Σz‖²` is minimised by embeddings that *cancel*, not agree — two opposite vectors score better than two identical ones. |

---

## 6 · The model, as designed

Every fragment is already a graph: mesh vertices are nodes, mesh edges are edges.
**Nothing new is built inside a fragment.** The only constructed connections are
between fragments.

1. **Intra-fragment layers** (5) — equivariant graph attention along real mesh
   edges. Attention scores are invariant inner products, so weighting equivariant
   messages by them stays equivariant.
2. **Inter-fragment layers** (3) — attention among fracture-surface tokens across
   *different* fragments of the *same scene*. This replaces the virtual nodes of
   the earlier design. All three sit *after* the intra layers, with **one intra
   layer following them** so what they heard spreads back through the mesh:
   without it only the tokens are informed, and the head pools over every
   vertex. Preflight measures the reach on the real meshes.
3. **Head** — two equivariant 3-vector channels per fragment, Gram-Schmidt into
   R̂, plus vertex and edge embeddings.

### The batching rule that is not optional

Cross-fragment attention is segmented **by scene**, not by batch. A batch
concatenates unrelated objects and fragment ids are globally unique, so unmasked
attention lets fragments of different scenes exchange information — making the
prediction depend on batch composition. This exact bug is in the project's
history. Keep both a `batch` vector (for scatter ops) and a `ptr` of cumulative
offsets (GARF's `cu(ℓ)`, for varlen kernels), and sort by `(scene, fragment)`
first so segments are contiguous.

### Stage two: translation

Not learned. Interface correspondences come from mutual nearest neighbours in
embedding space, then translations are solved globally by weighted least
squares on surface coincidence, with normal cancellation as a per-match weight
and collision as an optional relief step. Centring is reversible — the batch
carries each fragment's divisor and centroid. **Built** — see §11.

---

## 7 · Training plan

### The workflow, end to end

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="workflow_dark.svg">
  <img alt="One shared path builds every batch: a scene directory is read into fragments in their assembled frame; a fracture mask is computed from geometry alone, while coincidence clusters and the ground-truth copy are taken in the assembled frame and are training signal only; build_scene draws a random rotation per fragment, standardises, and samples tokens; collate concatenates fragments into one ragged batch; the network runs identically in both modes. The path splits only at the end, into a training branch that steps the optimizer and a validation branch that scores the prediction. A final band lists what is missing on a genuinely broken object." src="workflow.svg" width="620">
</picture>

Three things are worth reading off it.

**Almost nothing differs between training and validation.** The same data
pipeline, the same batch layout, the same forward pass, and the same five loss
terms — validation reports them rather than stepping on them. The split is the
last inch of the diagram, not a separate path.

**Validation is not label-free.** The perturbation `Q` is generated by this
pipeline, so `Qᵀ` is known and every metric is computable. What validation
measures is generalisation to unseen *objects*, not to unknown labels.

**A real broken object is a third case, and it is missing pieces.** No `Q`, so
there is a prediction and nothing to check it against. No coincidence labels, so
the dihedral mask — precision 0.24, over-labelling by 3.9× — is all that selects
the cross-fragment tokens; GARF's answer is a trained fracture segmenter at
>99.5%, and this project does not have one. And a rotation does not place a
fragment: stage two is not built. Any number reported on Breaking Bad is
therefore an optimistic bound, and the thesis has to say so.

| | |
|---|---|
| Losses | geodesic rotation (atan2) · node position · node normal cosine · adjacent face normals · correspondence (InfoNCE over coincidence clusters). Weights all 1.0, untuned. |
| Precision | fp32. AMP measurably degraded results before, and attention scores overflow in fp16 once products exceed ~256. |
| Budget | 2×T4, 12-hour session cap. Checkpoints must always write a current state, not only a best-so-far. |
| Split | Official Breaking Bad: 407 train / 91 val on Everyday. Artifact held out for a cross-domain test. |

### Reference values — check every number against these

All measured by Monte Carlo in `tests/test_losses.py`, not assumed.

| Quantity | Value |
|---|---|
| Chance geodesic error | 126.48° = π/2 + 2/π |
| Chance Euler RMSE, **random** prediction (residual convention) | 83.25° |
| Chance Euler RMSE, **identity** prediction | 83.18° |
| Axis correct, azimuth random | 90.0° |
| Untrained `L_normal`, `L_face` | 1.0, 2.0 |
| `L_position`, unit sphere | 4/3 |
| Perfect prediction, any term | 0 (geodesic exactly; others to the 1e-8 norm floor) |

**The 83.20° previously recorded here as "chance Euler RMSE" is the wrong
baseline, and the mistake is instructive.** It is the score for predicting the
*identity* every time, not for guessing randomly — and predicting the identity
scores three degrees *better* than guessing. A model that collapses to
near-identity, which is the cheapest early way to reduce a rotation loss, would
therefore appear to beat chance on GARF's headline metric while having learned
nothing. Geodesic error reads 126.5° for both, so it does not pay for the
collapse. Report geodesic as primary and Euler RMSE only for comparability.

Euler RMSE and geodesic error are also not the same metric on the same
prediction — a 30° single-axis error is 30° geodesic but ≈17.3° Euler RMSE.
Always say which.

The previous VN-GAT fit small subsets (43° train error against a 126.5° chance
baseline) but validation never went below ~94° at any dataset size — close to the
89.9° floor expected if a model recovers a fragment's axis but not its azimuth.
That is what the cross-fragment redesign is meant to fix: a one-shot per-fragment
canonicaliser has no access to relative pose, so azimuth is unrecoverable.

---

## 8 · Built versus not built

**Built** — scene discovery and splits · `SceneReader` · mesh topology, repair,
unique edges, safe normals · fracture extraction, both methods · canonical edge
normal ordering · fracture patches / sampling / budgeting · cross-fragment
correspondence · SE(3) perturbation · visualiser · exhaustive extraction pass ·
three analysis scripts · **Vector Neuron primitives · segment reductions ·
intra-fragment VN-GAT · cross-fragment attention · the composite loss · feature
construction and batch collate · the backbone and rotation head**. 311 tests,
no skips.

**Not built** — nothing on this list any more: the translation solver (stage two) was built in Sept 2026, §11.

**Built since** — `reassembly.training`: config, the Breaking Bad dataset,
train/val/test loops, warmup-cosine schedule, checkpoint and resume, history,
GARF-comparable metrics, two-GPU DDP, and resume across Kaggle sessions
(`max_hours`, mid-epoch checkpoints, `resume_from` a read-only input, SIGTERM
handling, cumulative time tracking, and a refusal to resume onto a changed
architecture).

Verified end to end on **synthetic** scenes — icospheres with random dents, not
Breaking Bad — where it fits a memorisable set of six from 126.5° to 23.8°.
That result says exactly one thing: the conventions are self-consistent and the
architecture can fit. A transposed rotation label or a misaligned loss row could
not produce it.

It says **nothing** about generalisation, and an earlier version of this
paragraph claimed more than that. It described 23.8° as clearing "the 89.9°
axis-only floor the previous design could not clear" — conflating training and
validation error. The previous VN-GAT reached **43° on training data** and
stalled at ~94° on *validation*; 89.9° is a validation floor. A training number
cannot clear it, because training error was never what was stuck there. The
comparison was not evidence of anything and has been withdrawn.

A second correction, Sept 2026: "floor" is the wrong word for 89.9° in either
column. It is where a model sits when it has found the symmetry axis and not
the rotation about it — a state, not a limit — and the earlier design's own
scaling run reached 30.9° on eight Everyday objects. See §10.

**Verified, no longer assumed.** `torch` installs fine from the *default* PyPI
index (the earlier failure was `download.pytorch.org` being blocked, not torch
being unavailable), so §4–§6 has been executed rather than reasoned about. In
float64:

| Check | Result |
|---|---|
| Every VN primitive, `L(xRᵀ) = L(x)Rᵀ` | ≤ 1e-15 |
| Gram-Schmidt head orthonormal, det = +1 | exact to 1e-16 |
| Intra-fragment attention, equivariance | 1.3e-15 |
| Cross-fragment, own-pose equivariance | 4.4e-16 |
| Cross-fragment, other-pose **invariance** | 3.3e-16 |
| Cross-scene leakage | exactly 0 |
| Whole model, per-fragment equivariance | ≤ 1e-9 (float64 through 6 layers) |
| Rotation label round trip | exact |
| Every parameter receives gradient | 77 / 77 |

**Two bugs the verification caught**, both of the kind that trains without
complaining:

- **Near-identity initialisation that was actually zero.** Zeroing a gate's
  final weight makes its output constant, which zeroes the gradient into
  everything upstream — the whole cross-fragment pathway plus the scale gate,
  31 of 77 parameters, dead on the first steps and waking only as a bias
  drifted. A *small* final weight keeps the layer within 0.14% of the identity
  and every path alive.
- **An epsilon in the wrong place.** `sqrt(s + 1e-8)` perturbs every norm, not
  just the near-zero ones: the head came out orthonormal only to 1e-8 with
  det = 0.99999995, and the geodesic loss had a 5e-9 rad floor at both 0 and
  180°. Clamping the squared sum instead is exact away from zero, and the
  geodesic path takes a far smaller floor because `atan2`'s derivative cancels
  the norm's.

**What the second preflight found.** With the split fixed, `batch_size=1` fit —
at 11.61 GB of 15.6 GB, which is the more informative number. Measured as the
slope against pair count, the cross-fragment layers were retaining **1109 bytes
per pair** for the backward pass: at 2048 tokens over 6 fragments that is 3.6 GB
per layer, 7.2 GB across the two of them, for a *single scene*. Recomputing the
pair gathers in the backward pass instead of storing them takes it to 86 B/pair,
0.3 GB per layer — a 13× reduction for one extra forward of an indexing op,
with outputs and gradients **bitwise identical** and the equivariance and
invariance residuals unchanged at 1e-16. On by default; tests pin all three
properties including the bytes-per-pair saving.

It also produced a retraction. `tokens_per_scene = 2048` was chosen by comparing
pair counts against GARF's stack — a FLOPs argument, on 4×H100. It was never
checked against T4 *memory*, which is the binding constraint and is quadratic in
that exact number. The value survives, but it survived by luck rather than by
the check having been done.

**What the third preflight found.** `batch_size=2` fits, at 11.96 GB of 15.6 GB
— so the default `batch_size` is now 2, measured rather than assumed. But the run still died, in a
*third* instance of one bug: a preflight step allocating memory as though an
earlier step had not run. Once because step 5's graph was never freed, once
because step 7 timed at the configured batch size that step 5 had just
disproved, and once — my own fix for the first — because step 6 sized from
`fitted` and then doubled it.

That it took three sessions to find is the real lesson: `preflight` was the only
function the suite could not reach, since it needs a dataset on disk in the
Breaking Bad layout. The suite now builds a synthetic one and runs preflight
end to end against it with out-of-memory simulated, which fails on all three.
`fitted` is now a hard ceiling for every later step, and each allocating step
degrades rather than raising — a preflight that dies of its own memory use
throws away every check that had already passed.

---

## 9 · Open

- **Fragments with zero fracture surface.** Coincidence still reports a minimum of
  0, which should be impossible. Most likely single-fragment modes, where the
  labelling returns all-false by construction — but it needs counting, because
  such fragments get no inter-fragment edges at all.
- **Inference-time labelling.** Coincidence needs the assembled scene, so it is
  unavailable at test time. GARF trains a segmenter on exactly these labels and
  reaches >99.5%. Until that exists, a model trained on perfect labels reports an
  optimistic number — state it in the thesis.
- **Supervision asymmetry.** The embedding-consistency loss is defined over
  coincident vertices, so it only supervises the fracture channel. Tokens from the
  original surface are shaped by the rotation loss alone.
- **Layer counts, loss weights, token budget.** All currently guesses. The loss
  weights are deliberately all 1.0 and untuned: the terms have very different
  natural scales, and the first run should *measure* the relative sizes rather
  than assume them. `ReassemblyLoss` returns every term separately for that.
- **Nothing has been trained to convergence.** One 13-minute calibration run
  has happened, on 400 objects for 3 epochs. It moved nothing on rotation, and
  that is not a verdict: 300 optimizer steps against a planned 65,000 is half a
  percent of training, with the whole cosine schedule compressed into it. The
  outstanding question — can the machinery fit *real* geometry at all — needs an
  overfit run on a few dozen real samples, not a short run on many.
- **Nothing has been trained.** Every number above is a property of the
  architecture, not evidence that it learns. The training loop, the GARF-metric
  evaluation and the translation solver are next.

---

## 10 · Audit against the earlier design (Sept 2026)

A full read of the previous VN-GAT codebase (`Thesis 1`, 71 files) against this
one. `docs/thesis1-review.md` holds the complete findings list; this section
records only what was **changed here** as a result.

### Defects fixed

| | what was wrong | why it was invisible |
|---|---|---|
| **Non-finite loss reached `backward()`** | the `isfinite` check ran *after* the backward, so a NaN was already in `.grad` and, under DDP, already all-reduced to the peer | the counter said "skipped"; Adam's moments said otherwise, and never recover |
| **`group_weight` leaked past an OOM** | `zero_grad` discards the whole accumulation group, but the weight was only reset at the optimizer step | the next step was silently scaled down by the discarded fraction — only on batches large enough to be interesting |
| **Euler RMSE used the wrong convention** | `euler(pred) − euler(target)` componentwise is not a metric on SO(3) | it manufactured a 3° gap making "collapse to identity" look better than guessing; that artefact was documented as a property of the metric |
| **Objects were double-counted** | `volume_constrained-*` variants were separate scene directories, so one shape was two objects | 809/181 reported against an official 407/91 — and a shape could be in train under one directory and val under the other, so the object split was not one |
| **No official test split** | Breaking Bad ships train and val lists only | `--split test` fell back silently to a hashed split comparable with nothing |
| **The 89.9° "floor"** | described as a structural limit of a per-fragment canonicaliser on surfaces of revolution | it is not a floor: a fragment of a symmetric object is not itself symmetric, its fracture boundary is unique. The earlier design reached 30.9° on eight Everyday objects |
| **`E_normal` in the translation energy** | normals are translation-invariant, so `∂E_normal/∂t ≡ 0` and the joint minimisation is `E_pos` alone | DesignV5 §1.10 reproduces the joint energy and is wrong as written; the term belongs as a per-match weight |

### Two experiment controls added

**`--split-by {object,fracture}`** — what is held out. `object` is the
benchmark: val shapes are never seen. `fracture` holds out *break patterns*
instead, so train and val share every shape. The pair is diagnostic in a way
neither is alone:

- object at chance, fracture well under it → per-shape canonical orientations
  learned, no transferable rule. Changing the representation or the loss, not
  the data or the epoch count.
- both at chance → generalisation is not the problem yet; look upstream.

`fracture` draws only from the official *training* shapes, so the official
validation shapes stay clean for the benchmark. It is a diagnostic, never a
reported result.

**`--balance {none,category,object}` with `--balance-temperature`** — Everyday's
categories hold 17 bottles against 5 cups, so uniform sampling shows the model
three bottles per cup and a shape prior is cheaper to fit than an orientation
rule. `category` makes category, then object, then mode uniform in turn;
temperature interpolates geometrically to natural frequency. Training only —
validation is never reweighted, so settings stay comparable — and the startup
banner reports the effective sample size, because balancing buys
representativeness with variance.

### Three diagnostics added

- **tilt / twist** (`reassembly.evaluation.metrics.swing_twist_error`) — splits
  the residual into rotation *off* the symmetry axis and *about* it. ~90°
  geodesic has two causes that call for opposite work, and the mean cannot tell
  them apart: `tilt ≈ 90` is "the axis is not learned either"; `tilt ≈ 0,
  twist ≈ 90` is "the axis is learned, the azimuth is not".
- **`head|cos|`** — collinearity of the two channels the rotation head feeds to
  Gram–Schmidt. Near 1 means the frame's second column is numerical noise and
  the prediction is one direction plus a random roll. Invisible in the loss
  (the output is still a proper rotation), and one dot product per fragment.
- **per-category validation breakdown** — the honest counterpart to balancing:
  it shows what the model *does* per category on a val set that is never
  reweighted, so "validation improved" can be told apart from "validation is
  now dominated by different categories".

Also added: `reassembly.evaluation.metrics` carries Chamfer distance (float64,
direct differences — `cdist`'s expansion puts a scale-dependent floor under
near-coincident points) and part accuracy, ready for the translation solver.

### Still outstanding from the review

*(Sept 2026: the first three items below are done — see §11, which also
corrects the claim made about `rotate_per_fragment` here.)*

- The translation solver itself — the corrected closed-form version: mutual-NN
  matching in embedding space, normal compatibility as a per-match weight, a
  weighted graph Laplacian solved in float64 with IRLS + Huber. Until it
  exists, Chamfer and part accuracy cannot be reported end to end.
- `rotate_per_fragment`: compute features once per fragment on the clean mesh
  and derive the perturbed view by rotating the extracted vectors on GPU. Exact
  for a rigid transform, and the single largest attack on the 645 ms/scene CPU
  cost.
- Collective voting on early exits (`all_ranks_agree`) — the deadline and
  stop-signal are still local decisions, which is a latent DDP hang.
- GARF's comparable row is the **vanilla Everyday supplementary** table
  (SE(3)-Equiv 79.30°, GARF-mini 10.41°), not the headline one the docs quote.

---

## 11 · One GPU, two or N — and the six additions (Sept 2026)

### The multi-GPU path had two silent faults

Both produced finite numbers and a descending curve.

1. **Replicas drifted apart.** DDP all-reduces inside the syncing backward;
   `run_epoch` then divided the gradient by the rank's *own* fragment count. Two
   GPUs holding 5 and 4 fragments stepped by different amounts, and nothing in
   DDP re-synchronises parameters, so they drifted for the rest of the run.
2. **Rank-local skips desynchronised the collectives.** The oversized-batch
   skip, the empty-batch skip and the non-finite-loss skip were decided per GPU,
   while DDP meets inside every syncing backward. A GPU that skipped met its
   peer one time fewer, and the last all-reduce of the epoch never completed —
   a hang, not an error. (`max_vertices_per_batch` had been sold as the safe
   skip because vertex counts are "identical on every rank". They are not: each
   rank holds different scenes.)

The same reading found five more, all fixed:

- validation reported rank 0's half, and `DistributedSampler` padded it by
  *repeating* scenes, so some were counted twice;
- the time budget and the stop signal were checked per GPU (the signal handler
  was installed on rank 0 only) — another way to leave a peer waiting;
- an out-of-memory error under DDP stopped the whole run;
- a trailing accumulation group cut short by the end of the loader was never
  stepped, and its gradient leaked into the next epoch's first step;
- every rank wrote the carried-over resume checkpoint to the same file at once.

And one that was not multi-GPU at all: the scene seed was `hash((key, epoch))`.
Python salts string hashes per interpreter, so every session and every spawned
process drew a different perturbation for the same scene — validation included,
so the numbers before and after a resume were not measured on the same rotations.
It is now a `blake2b` digest.

### What replaced it

No DDP wrapper. Each GPU accumulates its own micro-batches; the GPUs meet only
at optimizer steps, which are fixed by batch **position** (every `accumulate`
batches, plus a flush at the end), so every GPU reaches every one whatever it
skipped. At each: one all-reduce of two numbers (fragments contributed, stop
votes), then one all-reduce of the summed gradient, divided by the global
fragment count *after* the reduction. Every replica applies the bit-identical
update. A parameter no GPU produced a gradient for stays `None` everywhere, so
AdamW treats it exactly as on one GPU. Replicas are made identical once, at the
start (`broadcast_parameters`), since each rank seeds itself differently. The
cost DDP would have saved — overlapping the all-reduce with the backward — is a
few milliseconds a step at 0.9M parameters.

Stops are voted at the step. Validation runs on every GPU over unpadded,
disjoint shards and is gathered, as is the training summary. `train()` forwards
a SIGTERM aimed at the launcher to the GPU processes.

**Tested with real processes** (gloo, CPU, 2 and 3 of them): every step against
a single-process reference, the replicas compared bit for bit, in four cases —
all contributing; one GPU with an unusable scene; one out of memory on every
attempt; one skipping *every* batch — plus a stop on one GPU stopping all at the
same step, and `train()` end to end with a resume. Every run has a timeout, so a
hang fails instead of stalling. Moving the rescale back to its old place fails
the test (checked). These tests hide the machine's GPUs (`no_gpu`, in
`tests/conftest.py`). Where there are GPUs, `train(devices=N)` caps N at how many
there are, so on a one-GPU Colab machine they used to run one process and fail.

### The six additions

1. **Stage two** — `reassembly.assembly`: mutual nearest neighbours in the
   invariant embedding between fracture vertices of different fragments, normal
   opposition as a per-match weight (not an energy over `t`, where it has zero
   gradient), weighted Laplacian least squares in float64 with IRLS + Huber,
   optional sphere collision relief. `evaluate` reports RMSE(T), Chamfer
   (whole shape — the benchmark's CD — and per part) and part accuracy in world
   units, per scene then over scenes, by category. A perfect prediction scores
   RMSE 0, CD 0, PA 1 (tested). The batch now carries each fragment's divisor,
   centroid and fracture mask to make world units recoverable.
2. **The perturbed copy made on the device**, by rotation (`perturb_on_device`,
   default on). **Measured, and the saving is small** — the copy was already
   made by rotating, not recomputing. The same measurement found where the
   loader's time actually went, so the pair lists moved to the device too
   (on GPU runs; on CPU they stay in the parallel loader workers). Per batch of
   two synthetic scenes, loader side (build + collate; reading and labelling are
   the same in all three and excluded), one CPU thread:

   | | median scene, ~10k vertices | large scene, ~82k vertices |
   |---|---|---|
   | before: copy + pairs in the loader | 810 ms, 112.5 MB | 1374 ms, 213.4 MB |
   | copy on the device | 784 ms (97%), 107.6 MB (96%) | 1053 ms (77%), 174.0 MB (82%) |
   | copy + pairs on the device | 147 ms (18%), 7.3 MB (6%) | 543 ms (40%), 57.9 MB (27%) |

   The pair lists are ~3.1M ordered pairs per four-fragment scene at 2,048
   tokens — two int64 lists, 89% of the batch's bytes — and took ~0.6 s of CPU to
   build. **Not measured: the GPU's side** (three batched matmuls and one index
   build per step), for want of a GPU here. `python -m scripts.benchmark_data`
   measures all three ways on the real data and times the device side,
   synchronised, on whatever GPU runs it — run it on Kaggle before relying on
   these numbers.

   **Retraction.** §10 called this change "the single largest attack on the
   645 ms/scene CPU cost". That was carried over from the earlier design, where
   the perturbed view was built by re-running feature extraction. Here it never
   was, and the measured share is 3–23% of loader time; the pair lists were the
   large cost.
3. **An out-of-memory batch is retried** with gradient checkpointing forced on
   every layer — identical gradient (bitwise, on one thread; tested), less
   memory, more time — and skipped only if that fails too. Each micro-batch's
   gradient is computed with the running total set aside, so a failure part-way
   through a backward discards that micro-batch alone. That is bitwise too, on
   one thread (tested). Its first test allowed a tolerance on two threads instead,
   and failed now and then on a busy CPU: the sums round differently from run to
   run, and AdamW enlarges the difference by the second step.
4. **Repairs counted, failures named, repeat offenders tracked.** Zero-area
   faces and zero-length vertex normals are counted per scene (repaired to zero,
   never NaN — which is why they must be counted); non-finite coordinates skip
   the scene by name; a finite loss with a non-finite gradient is dropped too. A
   failed batch is charged to every scene in it — across epochs the culprit
   accumulates while its partners change — in a tally kept in the checkpoint
   and `offenders.json`. A scene failing in more than one epoch is printed as a
   repeat offender with the command that diagnoses it.
5. **Fixed-length epochs**, 800 optimizer steps per GPU by default
   (`--steps_per_epoch`; `0` = one full pass). A restarted partial epoch now rewinds its step count to
   where the epoch began, so an interrupted run's learning-rate schedule matches
   an uninterrupted one.
6. **Tools**: `benchmark_data`, `check_scene` (with `--locate`, and a gradient
   stage), `dump_prediction` (with the solver's placement), `visualize_reassembly`,
   `render_gif`, `scaling_sweep` (with `--max-objects`, a new Config field),
   `check_version` (every file compiles, every fix present, no shadowed tests).
   All take the training `Config` flags (`scripts/config_flags.py`).

### Thesis 1's flags, and one checkpointing switch

The command line now uses Thesis 1's flag names, with Thesis 1's meaning where
the two differed: `--batch_size` is scenes per optimizer step per GPU, processed
`--micro_batch_scenes` (default 1) at a time; `--steps_per_epoch` counts
optimizer steps; `--lr_min` is absolute; `--lr_warmup_epochs`, `--val_steps`,
`--resume auto|none|PATH`, `--save_every`, `--split_source`, `--lr_schedule
cosine|constant` as there. `scripts/config_flags.py` holds the conversions; the
Config fields keep their names. Thesis 1 flags with no counterpart here stop with
what to use instead. `--fracture_pattern` now defaults to `fractured_`, as in
Thesis 1.

The two per-layer switches (`checkpoint_intra`, `checkpoint_cross`) are replaced
by one, `--grad_checkpointing`, which recomputes **whole layers** as Thesis 1
does. The old intra switch recomputed only the attention scores, and each intra
layer still kept several edge-sized `(E, C, 3)` tensors — which is why a large
scene failed "even with gradient checkpointing" here and not in Thesis 1.
Measured on CPU at 128 channels, one forward and backward, four fragments,
2,048 tokens:

| vertices | old switches, both on | `--grad_checkpointing True` |
|---|---|---|
| 2,568 | 0.87 GB held, 1.06 GB peak | 0.22 GB held, 0.55 GB peak |
| 10,248 | 3.3 GB held, backward killed above 6.5 GB | 0.43 GB held, 3.7 GB peak |
| 40,968 | — | 1.0 GB held, 4.1 GB peak |

~20 KB held per vertex instead of ~250 KB. The ~3 GB peak is mostly the cross
layers' pair tensors, rebuilt one layer at a time; it scales with the tokens, not
the vertices. Outputs and gradients are bitwise identical with it on (tested, one
thread); the pair-gather recompute in the cross layers stays on always.

The time budget (`--time_budget_hours`) is now read between epochs, as in
Thesis 1: the epoch it runs out in is finished, validated in full and saved as
complete, and the next session starts at the next epoch. Before, it stopped at
the next optimizer step and the half-done epoch was run again; and validation
had its own cap, 5% of the budget, which could cut it short without saying so.
A kill signal still stops at once, with a checkpoint.

### Still outstanding

- Everything above is verified on CPU and synthetic scenes. The first real
  multi-GPU run should confirm replicas stay identical there too
  (`reassembly.distributed.parameters_differ`), and `benchmark_data` should be run
  on the real data on the T4s.
- The perturbed-view edge arrays still ship both directions of every edge; the
  reverse copy is derivable (`(n2, n1, −δ)`), which would halve the remaining
  edge bytes.
- GARF's comparable row is the **vanilla Everyday supplementary** table
  (SE(3)-Equiv 79.30°, GARF-mini 10.41°), not the headline one.


---

## 12 · Thesis v6 — the largest fragment as the anchor (Sept 2026)

`E:\Thesis v6` is a copy of My Thesis Work with one change of question. My
Thesis Work stays as it was.

### Why

On the official split My Thesis Work's validation improved only on the
embedding term. One epoch's validation line: geodesic 123.0° (chance 126.5°),
tilt 87.0°, twist 87.8°, `head|cos|` 0.56, `match@1` 0.67 — against 72.9° on
training. Tilt near 90° means not even the axis is found on a new shape; the
head is not degenerate (0.5 is random directions); and the fracture matching
*does* transfer (0.67 of fracture points find their true partner first).

The rotation label is each fragment's rotation back into the frame its object
is stored in. For an unseen object that frame cannot be read off the input —
most Everyday objects are round, and the turn of the stored object about its
axis is arbitrary — so the loss can only fall by remembering training shapes.
The one term that improves on validation is the one that ignores orientation.
Thesis 1 behaved the same way. SE(3)-Equiv, whose cross-fragment rule this
project uses, trains on the same absolute target and reports 2.00 rad on unseen
Everyday objects, its baselines 2.21–2.24 rad (chance). GARF instead fixes
anchors "with identity rotations and zero translations" in training and the
largest fragment at inference (supplementary C.3), and its 6.1° is measured
that way.

### What changed

`reassembly/nn/anchor.py`. One rotation per scene, `C = R_a R̂_aᵀ`, on the left
of every prediction in the scene, so the anchor is exact and a prediction that
is right up to one global rotation is exact everywhere. The aligned error is the
error of the rotation *relative to the anchor* (the geodesic is bi-invariant).
The anchor is the fragment with the largest radius — the one that already sets
each scene's normalisation divisor — ties to the first; it is left out of every
average, where it would be a free perfect score (a third of the median scene).

* **Metrics**, always: the anchor-aligned error on every other fragment, the
  absolute error beside it (`absolute geo`). `best.pt` is chosen on the
  anchor-aligned geodesic. Stage two measures translations from the anchor's.
* **Loss**, by default (`--rotation_target anchor`): rotation, position,
  normal and face on the aligned prediction, over every fragment but the
  anchors. The anchor's own prediction is trained through all of them. The step
  and the epoch averages are weighted by the fragments scored — one fewer per
  scene — which also re-weights the embedding term by `n − 1` per scene instead
  of `n`. `--rotation_target absolute` is My Thesis Work's loss.
* **Old checkpoints** carry no `rotation_target` and are read as absolute:
  `--evaluate` reports their loss on that target, so the loss line reproduces
  their own log, then the two rotation errors in two rows. Resuming one under
  the anchor target is allowed and announced.
* Unchanged: the network, the cross-fragment rule, the data, every other flag.

### Measured

| check | result |
|---|---|
| per-scene global rotation removed (float64) | exact to 1.6e-14; anchor error ~1e-14 |
| chance on the scored fragments, 6,000 random predictions | 127.0° (126.5 ± 1.5) |
| a checkpoint written by My Thesis Work's own code, re-scored here | loss line identical; absolute 110.81° = My Thesis Work's 110.81°; anchor row computed on 6 of 10 fragments |
| memorisation test, six scenes, four seeds, one thread | absolute target after 45 epochs 23–36°; anchor target 37–76° after 45, 19–30° after 80 |
| tests | 461 (was 449): the learnability test runs under both targets; 56 version markers |

The anchor target fits more slowly: its targets move with the anchor's own
prediction while both are learned, and in a scene of `n` fragments the anchor
receives the gradient of `n − 1` of them. A loss aligned by the best rotation
over *all* fragments would spread that evenly; it is not built.

### How to read the first runs — decided before seeing them

* **Re-scoring a My Thesis Work checkpoint.** Anchor row far below its absolute
  row: the fragments of a new shape already agree with each other, and only the
  global frame was wrong. Both near chance: they do not agree either.
* **Training on the anchor target, official split.** Validation anchor error
  well below 126.5° while training falls: the relative question generalises, and
  the absolute target was the obstacle. Training falls and validation stays at
  chance: the fragments cannot agree on a shared frame for an unseen shape
  through invariant messages alone — the next step is then to let the anchor's
  orientation reach the other fragments, or to take rotations from the
  embedding matches as stage two takes translations. Training does not fall
  either: optimisation, not generalisation — compare its pace with the
  memorisation numbers above before concluding anything.

---

## 13 · The benchmark's 2–20 pieces (Sept 2026)

GARF trains and reports on break patterns of 2 to 20 pieces (section 4.5: "only
been trained on data with 2-20 fragments"; supplementary C.5). This dataset's
scenes run to 99 (§3), and v6 had no upper bound — only the build-time rule that
skips a single-fragment scene — so its numbers were on a harder mix than GARF's.

**What changed.**

- `Config.max_fragments` (`--max_fragments`, `0` = `None` = no limit, the
  default): `limit_fragments` in `data/catalog.py` keeps the patterns of 2 to
  `max_fragments` pieces and drops objects left with none. The count is
  `data/scene.py:piece_count` — each pattern's `compressed_fracture.npy`, read
  on 16 threads and cached per process, so the per-epoch rebuild of the
  training set costs nothing after the first. It is an upper bound on what the
  loader builds (a label that comes out empty is dropped). An unreadable label
  file is kept for the loader to report; `build` also skips, by name, a loaded
  scene above the limit.
- On the catalogue rather than at build time: a scene rejected after loading
  costs a decompression every epoch it is drawn and still counts towards the
  epoch length and the balanced sampler's weights.
- After the split (no pattern moves between train and val), before
  `--max_objects` (which then strides over objects that still have one).
- In `_DATA`: a resume onto a different limit is announced — a checkpoint
  without the key reads as no limit (`_DATA_BEFORE_IT_EXISTED`) — and resets
  the best-so-far (`_fragment_limit_changed`). With `--evaluate`, an explicit
  `--max_fragments` overrides the checkpoint's value (`override`), the rest of
  the data definition stays the checkpoint's; the range is printed and written
  to `<split>_metrics.json`. `find_scene` searches every pattern regardless.

**Verified.** 476 tests (461 before; `tests/test_fragment_limit.py`, 15).
Mutation checks — limit after `--max_objects`, lower bound ignored, a legacy
checkpoint's missing key not read as "no limit" (both in the reset and in the
resume warning), the evaluate override ignored, the build-time check removed,
an unreadable label file dropped, `--max_fragments 0` not meaning "no limit" —
each fails at least one test. 61 version markers.

---

## 14 · Optional Hugging Face Hub mirror (Oct 2026)

`--hf_repo_id`, `--hf_local_dir`, `--hf_token`: with all three, the run's files
are pushed to a Hugging Face repository after every epoch; with none, nothing
changes; with some, one line says so and training continues without it.

- `src/reassembly/hub.py` (new): `HubSync.start` checks that `--hf_local_dir` is
  the `--checkpoint_dir`, that `huggingface_hub` imports and that the repository
  can be created (private, `exist_ok`); with `--resume auto` and no local
  `last.pt`, it pulls first. `push` uploads `last.pt`, `best.pt`,
  `history.json`, `history.csv` and `offenders.json` in one commit. Every Hub
  call is guarded: a failure warns and training continues.
- `train(config, hub=...)` starts it once, before any GPU process looks for
  `last.pt`; rank 0 pushes after the epoch's files are written and before the
  stop decision, so a session's last epoch is mirrored too.
- Script-level flags in `scripts/train.py`, not `Config` fields: `Config` is
  stored in every checkpoint, and the checkpoints are what gets uploaded.

**Verified.** 487 tests (`tests/test_hub.py`, 11, against a stand-in for
`huggingface_hub`, end to end through `train`: a push per epoch after the files
exist, a fresh folder resumed from the repository, the epoch a time budget stops
on pushed). Mutation checks -- hub not passed to the worker, pull overwriting a
local `last.pt`, no pull, folder check removed, repository not private, push
after the stop, token in `repr` -- each fails a test. 64 version markers.

## 15 · Rotations from the matches, and two diagnostics fixed (Oct 2026)

### Why

Read on W10 (interleaved schedule, 128 channels, wd 0.1) with `probe_val.py`,
which rebuilds a run's own validation set from its checkpoint and reproduced
the logged `val_geodesic_deg` exactly (102.05 deg at epoch 334):

- The rotation head is the bottleneck, not the data or the embedding. Between
  fragments that touch, the head's relative rotation was 98.7 deg off at the
  median; the rotation that lines up the two fragments' embedding-matched
  points was 0.1 deg off (91% of fits under 5 deg). Chained from the anchor,
  those fits reached 90% of the scored fragments at 2.7 deg mean, and took the
  validation error from 102.0 to 14.8 deg (median 0.3) with no retraining; most
  of what remains is the unreached 10%, tiny fragments where the head is at
  chance.
- The cross layers matter: with their partners removed the error rose to 116.5
  deg, with another object's fragments swapped in to 122.3. They carry the
  large-fragment advantage and the 2-piece scenes. The touching advantage among
  small fragments survives without them, so that part is neighbours making the
  same mistake.
- Caveat for every number from this route: the two sides of a Breaking Bad
  break share the same vertices, so a correct match lines up exactly. Tables
  built on independently sampled points (GARF's) never see that.

### What changed

- `src/reassembly/assembly/rotation.py` (new): `pairwise_rotations` (mutual
  nearest neighbours as in the translation solver, Kabsch + RANSAC per pair of
  fragments, on the input coordinates), `chain_rotations` (a maximum spanning
  tree of RANSAC inlier counts, grown from the anchor; unreached fragments keep
  the head's rotation), `match_rotations` (both, for one scene).
- `score_batch(..., rotations="matched")` and `--evaluate --rotations matched`:
  scored over the same fragments as the head, printed beside it, assembled with,
  written to `<split>_metrics_matched.json`. RANSAC is seeded per scene by name.
  `dump_prediction --rotations matched` places a dump with them.
- `evaluation.metrics.swing_twist_error` reads the error in the object's frame,
  `predicted @ target^T`. It took `predicted^T @ target`, the input frame, so
  every `tilt_deg` / `twist_deg` written before this is uninformative (plates
  showed 59/54 deg that are 32/72 in the object's frame).
- `scripts/config_flags.config_flags` writes `0` for a "no limit" setting whose
  default is not `None`: `--modes_per_scene 0` used to come back as 8, so
  `scaling_sweep` passed its children fewer modes than asked.

**Verified.** 504 tests (17 new: `tests/test_assembly_rotation.py`, an
object-frame tilt/twist test, the `config_flags` round trip, `evaluate` and
`dump_prediction` with matched rotations). Exact fits and an exact chain from
exact matches; RANSAC exact with 60% wrong matches; a head made wrong on
purpose scores perfectly through the matched route and does not enter the
result. Mutation checks -- the old tilt/twist residual, the chain composed the
wrong way round -- each fail a test. 68 version markers.

## 16 · Reports and figures from saved results; the matched route checked for leaks (Oct 2026)

### Is the matched route using anything it should not?

`probe_val.py` on W10 (epoch 334, the 1,023-scene validation subset), one test
at a time, with the scenes and RANSAC draws of the normal run:

| test | what changes | all pieces, mean / median | chain reached |
|---|---|---|---|
| normal | -- | 14.8 / 0.3 deg | 90% |
| `--hide_truth` | every ground-truth field NaN before the network and the solver | identical, line for line | 90% |
| `--untrained` | random weights | 127.5 deg (chance 126.5) | 11% |
| `--shuffle_fingerprints` | each piece's embeddings dealt out at random over its break vertices | 102.7 deg (the head's, as fallback) | 1% |
| `--jitter 0.01 --drop 0.5` | noise on every input vertex, half the break vertices left out, so no two points coincide | 44.6 / 8.5 deg (head 102.8 / 105.4) | 68% |

- Nothing reads the truth; the result is learned; it is the per-point
  fingerprints that carry it. The match candidates (the fracture mask, computed
  when the data loads) are outside `--hide_truth`, and W10's are shape-only
  (`dihedral`): the mask covers 0.64 of a piece at the median against 0.31 for
  the true contact, and more than the contact in 5,098 of 5,350 fragments.
- Without coincident points the route still works but is less exact, and it
  reaches fewer pieces (placed pieces 10.7 / 3.4 deg). The 0.3 deg median is
  partly Breaking Bad's shared break vertices: report the noisy figures beside
  the clean ones, and use them for any comparison with GARF-style tables.

### What changed

- `--evaluate` saves the report it prints as `<split>_report.txt` /
  `<split>_report_matched.txt`, beside the metrics. The report is built from the
  summary alone (`training.format_evaluation`), so `scripts/report_metrics.py`
  rebuilds it from any metrics file, older ones included. The summary keeps the
  report's header under `"evaluation"` (split, samples, checkpoint, epoch,
  subsets, fragment range, rotation target).
- The metrics file writes its per-scene records one to a line
  (`training.dump_metrics`): a full split had put every per-fragment number on
  its own line -- 1.4 million lines, 32 MB. `json.load` reads it as before. The
  metrics are written before the report, so a failing report cannot lose them.
- A matched evaluation keeps the head's own angles on the same fragments
  (`_network_geodesic_deg`), so one file holds both distributions.
- `reassembly/viz/results.py` (numpy only) reads histories and evaluations back:
  per-piece arrays aligned with their scene (the anchor left out), object types
  that line up across subsets (`everyday_compressed/Bottle` is `Bottle`;
  artifact, which has no categories, is `Artifact`), distributions and tables.
  `reassembly/viz/figures.py` draws from them, and `scripts/make_figures.py`
  writes figures, CSV tables, an index and a zip in one call.

**Verified.** 512 tests (8 new): the saved report is the printed one and is
rebuilt identically from the metrics file; the head's kept angles average to
the logged error; every figure draws from a real evaluation and history; types
and per-piece alignment on synthetic records. 75 version markers.

## 17 · Thesis v7 — placing the pieces the matched rotations turned (Oct 2026)

`E:\Thesis v7` is a copy of Thesis v6 with one change, in stage two only: how
the fragments are placed once `--rotations matched` has turned them. The
network, the training and the data are untouched; no checkpoint needs
retraining. (§18 is the second change, which does retrain: the head removed.)

### Why

Read on W10's prediction dumps (seven scenes, each with the head's and the
matched rotations), in the visualizer and in numbers:

- Spoon (50 pieces) and Ring (23): about three pieces in four turned within
  5 deg, yet 3 of 50 and 1 of 23 placed -- part accuracy 0.04 and 0.00,
  RMSE(T) 0.36 and 0.32. The placed assembly spread over 0.10 and 0.21 of its
  true spread (RMS distance from the centre); vase, wine bottle, bowl and plate
  0.86-0.97.
- The rotations were not the cause. With the same matched rotations and the
  true break correspondences, the well-turned pieces land 0.0065 and 0.005
  world units from their places (median).
- The cause is the solve. `assemble` is one least-squares problem over every
  embedding match; in a many-piece scene hundreds of matches join pieces that
  do not touch, and Huber caps each one's pull without removing it. The
  assembly contracts onto its centre, and the anchor gauge then measures every
  piece from there.
- The per-piece position loss (training's, and the visualizer's) moves a
  piece's points by its rotation alone, so it can be small for a piece that
  sits in the wrong place.

### What changed

- `assembly/rotation.py`: a pair fit keeps its whole motion. `ransac_motion`
  returns the rotation, the offset and the inlier mask; `PairRotation` carries
  `shift`, `source` and `target` (the inliers' point indices).
  `ransac_rotation` wraps it with the same draws, so the matched rotations are
  unchanged bit for bit.
- `assembly/placement.py` (new), for one scene: **verify** -- a pair counts when
  the chain reached both fragments, it has at least 6 inliers, and its relative
  rotation is within 5 deg of the chained rotations (the chain's edges pass by
  construction; other passing pairs close loops); **place the reached** by least
  squares over the verified inliers alone, the root held at 0 instead of the
  zero-mean gauge, Huber IRLS as before (`solve_with_held`); **place the rest**
  -- unreached fragments, which keep the head's rotation -- from every match that
  touches them, the reached held; a fragment with no match goes to the reached
  ones' mean.
- `score_batch(..., placement=...)`: `"checked"` (the default with matched
  rotations) or `"global"` (v6's solve; the default, and the only choice, with
  the head's rotations). `--evaluate --placement`, `dump_prediction
  --placement`. The summary and report name the placement; a non-default one
  adds its name to the files (`val_metrics_matched_global.json`). Each scene
  gains `verified_matches` (the inliers the reached pieces were placed by;
  `matches` still counts every embedding match). Dumps record
  `placement_method`; the summary tables of `make_figures` a placement column
  (files from before v7 read as `global`).
- `probe_val.py` is unchanged: it measures rotations and never places.

### Measured -- simulated matches on real geometry, before the network run

Replayed through `score_batch` on the dumps' meshes and the head's rotations,
with exact break matches and then a share made wrong (pairs of break points on
two random pieces; the chain and RANSAC as in evaluation):

| scene | pieces | wrong matches | part accuracy, global -> checked | RMSE(T), global -> checked | well-turned pieces placed |
|---|---|---|---|---|---|
| Spoon | 50 | 21% / 44% | 0.041 -> 0.939 / 0.041 -> 0.898 | 0.341 -> 0.015 / 0.358 -> 0.017 | 5% -> 100% / 5% -> 100% |
| Ring | 23 | 24% / 47% | 0.091 -> 0.773 / 0.000 -> 0.727 | 0.144 -> 0.060 / 0.234 -> 0.069 | 12% -> 100% / 0% -> 100% |
| artifact | 7 | 28% / 52% | 1.000 -> 1.000 / 0.333 -> 1.000 | 0.012 -> 0.001 / 0.059 -> 0.004 | 100% / 33% -> 100% |
| wine bottle | 4 | 26% / 50% | 0.667 -> 1.000 / 0.333 -> 1.000 | 0.037 -> 0.0001 / 0.113 -> 0.001 | 67% -> 100% / 50% -> 100% |
| vase, bowl, plate | 3-5 | 24-59% | 1.000 -> 1.000 | 0.002-0.017 -> 0.0001-0.0008 | 100% |

Under `global` the Spoon contracted to 0.15 / 0.09 of its spread, as the real
dump did (0.10); under `checked`, 0.99. What `checked` leaves is pieces whose
rotation is wrong. Not measured: the network's own wrong matches (they may
cluster where random ones do not), and the placement under `probe_val.py`'s
`--jitter/--drop` noise, where no two points coincide.

### Verified

527 tests (15 new in `tests/test_assembly_placement.py`; the `evaluate` and
`dump_prediction` tests extended). The failure is reproduced in miniature: 8
stacked slabs, a quarter of the matches wrong, the rotations exact either way
-- the global solve pulls the stack to 0.6 of its spread and places no slab;
the checked placement places every slab exactly. A slab the chain cannot reach
misplaces only itself. Mutation checks -- the reached pieces placed from
every match, no agreement test, the global solve always, a sign error in the
held solve, the reached re-solved with the rest, the root left free, no
centring under the mean gauge -- each fail a test. The global path gives v6's
output byte for byte over 27 configurations (head and matched rotations, both
gauges, collisions on and off, float32 and float64, the RANSAC draws). 84
version markers.

### Measured on the network (W10, Oct 8)

`last.pt`, epoch 350; Everyday val, 2-20 pieces, 728 scenes (91 objects x 8
break patterns); `--evaluate --rotations matched`, once as is (checked) and once
with `--placement global`. The rotations were identical in both runs -- matched
15.18 deg mean, 0.31 median, acc@5 0.847, 89% of the scored pieces reached; the
head 102.87 -- so the difference is the placement alone:

| | global (v6) | checked (v7) |
|---|---|---|
| part accuracy | 0.745 | 0.940 |
| RMSE(T) | 0.0562 | 0.0126 |
| Chamfer, whole / per part | 0.00526 / 0.04097 | 0.00019 / 0.01447 |
| matches/scene | 489 | 489 (301 in the verified pair fits) |

- Part accuracy rose or held in all 20 categories; most where pieces are many
  or thin: Spoon 0.265 -> 1.000, WineGlass 0.271 -> 0.975, Statue 0.231 ->
  0.943, Ring 0.143 -> 0.819, Mirror 0.529 -> 0.969, Plate 0.540 -> 0.974,
  ToyFigure 0.439 -> 0.860. Teapot 0.875 in both.
- Lowest left: Ring 0.819, ToyFigure 0.860, Teapot 0.875, DrinkBottle 0.891,
  Bottle 0.898 -- most likely the 11% of pieces the chain did not reach, which
  keep the head's rotation; not yet checked piece by piece.
- The embedding loss and match@1 differ slightly between the two runs (1.9594
  against 1.9526, 0.659 against 0.660): `correspondence_loss` subsamples its
  anchors with the unseeded global generator. Nothing to do with the placement.

### Without shared break vertices (`--jitter`, `--drop`)

Every number above rests on Breaking Bad's shared break vertices, and
`probe_val.py --jitter/--drop` had measured only the rotations without them.
`--evaluate` now takes the same two settings (`reassembly/evaluation/noise.py`):

- `--jitter S`: Gaussian noise of `S` largest-fragment radii on every input
  vertex before the network sees the scene, each edge's relative position
  recomputed; normals and labels untouched. In largest-fragment radii under
  either normalisation (`probe_val.py` added it to the normalised coordinates,
  the same thing under the scene normalisation every run has used).
- `--drop P`: that share of the break vertices left out of the matching, for
  the rotation fits and the placement alike.
- The method reads the noisy inputs -- network, embedding, fits, solve
  (`run_epoch(inputs=...)`, `score_batch(observed=...)`); the assembly is scored
  on the clean fragments, the predicted pose applied to the true geometry. The
  loss line is the noisy inputs'.
- Draws tied to each scene's name and `--seed` (`noise.scene_generator`, one
  stream per perturbation; the RANSAC stream is the old one), so the two
  placements see the same noise.
- Both 0 by default: then `score_batch`'s output is byte-identical to before for
  both placements (checked on the slab scenes, every combination of anchor and
  collisions). The summary records `"noise"`; the report names it; the files
  carry it (`val_metrics_matched_jitter0.01_drop0.5.json`). The flags without
  `--evaluate`, a negative jitter or a drop of 1 or more stop with a message.
- **Verified.** 535 tests (8 new in `tests/test_evaluation_noise.py`, the
  `evaluate` test extended): off changes nothing; the noise moves positions
  only, at the stated size, under both normalisations; it belongs to the scene,
  not the batch; the solve reads it and the Chamfer distance does not; the drop
  thins the matches; the model is shown the noise. Mutation checks -- no
  per-fragment scale, stale edge features, scoring on the noisy geometry, the
  drop ignored, one stream per batch, the clean batch shown to the model, the
  observed batch ignored -- each fail a test. 90 version markers.

### Next

`--evaluate --rotations matched --jitter 0.01 --drop 0.5` on W10, with and
without `--placement global`: the matched route's number without the shared
break vertices, the one to put beside GARF-style tables.

## 18 · Thesis v7 — no rotation head; every rotation from the matches (Oct 2026)

The second change in `E:\Thesis v7`, and the one that needs retraining.

### Why

- On W10 (epoch 334, Everyday val) the head's rotations were 102 deg off,
  anchor-aligned, where the rotations fitted from its own embedding's matches
  were 15 deg off (§15). The head was the weak part of its own model.
- nrhl, the same network trained on the embedding term alone, beat W10 on
  Everyday's validation split: 10.2 deg against 15.2, acc@5 0.905 against
  0.847. Training the head bought nothing.
- So the network ends in its embedding; stage two fits every rotation and
  every translation; the four geometric terms only score the result.

### What changed

- `nn/model.py`: `Prediction` is `(vertex_embedding, vertex_features)`; the
  pooled frame and the Gram-Schmidt head (`pool_proj`, `head`) are gone.
  `DEFAULT_SCHEDULE` is `intra intra cross intra cross intra cross intra`, the
  schedule nrhl was trained with; `intra x5` is the ablation without a cross
  layer.
- `nn/losses.py`: the total is the contrastive embedding term alone -- a
  constant 0 with no gradient for a batch with no coincidence cluster. The
  rotation, position, normal and face terms are computed under `no_grad` from
  the rotations they are given and reported, per fragment too, for the
  breakdowns by category.
- `assembly/rotation.py`: `match_batch` fits every scene's rotations from the
  embedding matches reading nothing the method lacks at inference -- each
  scene's largest fragment held at the identity, a fragment the chain does
  not reach left there. The anchor protocol turns them for scoring, so an
  unreached fragment scores at chance.
- `training.py`: every validation batch is matched and scored (training
  batches too with `--score_train True`; off, because it matches 2,560 scenes
  an epoch at the user's settings); a batch with no cluster is left out of the
  step and named (`no-objective`); `best.pt` is the highest validation acc@5
  (`BEST_METRIC`, stored in every checkpoint as `best_metric`: a run resumed
  from one chosen by another metric -- the first head-less copy used acc@10 --
  recomputes its best-so-far from the complete epochs logged); the epoch line
  and the report give the share reached, and
  the report the scores and accuracy per category. `--w_rot`, `--w_pos`,
  `--w_normal`, `--w_face` and `--rotation_target` are refused by name
  (`scripts/config_flags.py`). A checkpoint with the head evaluates and probes,
  its head's weights left behind and said so; resuming one is refused.
- `evaluate`: no `rotations` argument -- the files are `<split>_metrics.json`
  and `<split>_report.txt` again (`_global` and the noise settings still
  added). `--split all` scores every object of the given subsets whatever its
  split, and counts the shapes the checkpoint was trained on (a
  `volume_constrained-*` copy holds its base subset's shapes). `--predictions
  DIR` writes every scored scene's dump into one zip, in the data's own layout
  (`<subset>/<category>/<object>/fractured_<k>.npz`) with a CSV index
  (`_PredictionWriter`, `assembly/dump.py` -- the writer `dump_prediction`
  uses too).
- `probe_val.py`: the matched route only (`--procrustes` accepted, always on);
  `--table` gathers several probes' summaries into one table. `--max_fragments
  N` probes break patterns of 2 to N pieces instead of the run's own range (0:
  every count) and `--modes_per_scene N` draws N patterns per object instead of
  the run's own number (0: every pattern; still `--limit` scenes of them), both
  named in the report and in the table's `pieces` and `patterns` columns; under
  `--cross none` the report says that no vertex hears another fragment.
- `viz/results.py`, `viz/figures.py`, `scripts/make_figures.py`: the best
  epoch by acc@5, a validation-accuracy figure, `tables/by_type.csv/.md`
  (every number per object type and for the whole subset); files from before
  still read.
- `evaluation/metrics.py`: `head_collinearity` removed with the head.

### Verified

524 tests, all passing (535 before: the tests of the head and of its training
went with it). The head's tests were rewritten for what replaced it: the features
turn with their fragment and the embedding does not move under any pose
(`test_model.py`); exact matches score 0 in any frame, and the metrics and the
rotation score are one number (`test_anchor.py`); the scores never enter the
total (`test_split_integration.py`); a batch without clusters has nothing to
train on and does not move the weights; a checkpoint with the head evaluates
and does not resume (`test_training_recovery.py`). The learning test now trains
the embedding on six broken slab stacks and requires the rotations fitted from
it to be right: on one thread, over seven seeds, 120 epochs took them from
chance to exact (0.0 deg, every fragment reached), where 80 epochs got one seed
in four there. Mutation checks -- the coincidence labels shuffled over the
vertices, the fitted rotations transposed, the anchor alignment on the wrong
side, the best-so-far kept across a change of metric -- each fail a test. 108
version markers.

### Next

Train the head-less model and its ablation (`--schedule intra intra intra intra
intra`); probe both; evaluate both on Everyday val and on the whole
volume-constrained Everyday, artifact and volume-constrained artifact subsets,
with figures and every prediction.

---

## 19 · Thesis v7 — the patterns above 2–20, scored on their own (Oct 2026)

**Why.** The results on the benchmark's 2–20 pieces and on the patterns above
it are to be reported, and drawn, apart. `--max_fragments` (§13) has no lower
end, so the second set could not be selected.

**What changed.**

- `Config.min_fragments` (`--min_fragments`, `0` = `None` = no lower limit, the
  default). `limit_fragments` takes `maximum=None` for no upper end, and the
  dataset applies the limit when either end is set: `--min_fragments 21
  --max_fragments 0` keeps every pattern that `--max_fragments 20` drops.
- Both ends are counted from the label file (`piece_count`), so the two runs
  score each pattern of two pieces or more exactly once; a pattern counted at
  21 that loads fewer pieces (an empty label is dropped) stays in 21+. For the
  same reason the lower end has no build-time check, unlike the upper one: the
  loaded count can be below the label count, and such a check would leave that
  pattern out of both runs.
- Validated: at least 2, and not above `max_fragments` — the message says that
  `--max_fragments 0` is no upper limit. With `--evaluate` the checkpoint's
  `max_fragments` applies unless the flag is given, so the 21+ run passes both.
- In `_DATA`, and in `_DATA_BEFORE_IT_EXISTED` (a checkpoint without the key
  reads as no lower limit); `_fragment_limit_changed` compares both ends. With
  `--evaluate`, an explicit `--min_fragments` overrides the checkpoint's value
  (`scripts/train.py`'s `override`). The banner and the report print the range
  (`21+ per scene`), `<split>_metrics.json` records `min_fragments` beside
  `max_fragments`, and `find_scene` still searches every pattern.
- The output names do not carry the range (`<split>_metrics.json`,
  `<split>_report.txt`, `<subsets>-<split>.zip`): each range is evaluated into
  its own `--checkpoint_dir` and `--predictions` folder.

**Verified.** 535 tests (524 before; `tests/test_fragment_limit.py`, 26).
Mutation checks — the lower end ignored by the dataset, a lower end alone
applying no limit, either end made exclusive, the best-so-far reset blind to
the lower end, a legacy checkpoint's missing key not read as "no lower limit",
the key left out of `_DATA`, the evaluate override ignoring `--min_fragments`,
`--min_fragments 0` not meaning "no lower limit", `find_scene` held to the
range, the report line or the metrics field without it — each fails a test.
110 version markers.

## 20 · Thesis v7 — other datasets in Breaking Bad's place (Oct 2026)

**Why.** v7 is to be scored on Fractura (GARF's HDF5 files) and Fantastic
Breaks. A first attempt — readers inside `scene.py`/`training.py`, decimation
in the loader — gave an embedding term of 0.0000 and wrong poses everywhere but
the synthetic bones. Measured on its prediction dumps, the causes were in the
data as the model received it, not in the model:

- Fractura's real bones store a vertex per face corner: one piece reached the
  model as 518,631 vertices in 168,427 patches, with no mesh edge between its
  faces. Every piece checked also holds a separate 2 × 2 × 2 cube of 12 faces
  at the origin (Blender's default cube). They are in millimetres, so part
  accuracy (0.01 in object units) was 0 throughout, and `pig_bone_15` is one
  mesh stored twice.
- The synthetic bones store the walls between cells twice, facing opposite
  ways (1,142 and 2,270 pairs in one scene's two pieces), which leaves 2,173
  and 4,268 edges with more than two faces.
- No real set has a vertex shared between pieces: no coincidence cluster, so
  no embedding term (the 0.0000), and every match comes from the embeddings
  alone. After cleaning, their edges are 1.2 to 9.9 times shorter than
  Breaking Bad's median.
- The pieces "centralising" in the viewer: in the nine failed scenes the
  placed pieces' centroids were 0.06 to 0.23 as far apart as the true ones,
  0.96 in the good one — a least-squares fit to wrong matches pulls the
  matched sets onto each other.

**What changed.**

- `scripts/convert_external.py` converts once, offline, into Breaking Bad's
  layout, so `--evaluate`, the figures, the dumps and the Visualizer run on the
  result unchanged. A CSV per subset records, per assembly, what was done, the
  density before and after, the shared vertices (found as training finds
  them), and the faces the dihedral test flags — with recall and precision
  where the dataset marks its break faces.
- `data/external.py`: readers for GARF's HDF5 (`data_split` lists, pieces in
  numeric order, `shared_faces` as the marked break) and Fantastic Breaks
  (`model_b_<k>`, `model_r_<k>`, `meta_<k>['mask']` per vertex or per face);
  `prepare` cleans, scales (largest side 1; `--scale piece` is GARF's loader's
  convention), thins, tidies what the collapse leaves, and reports.
- `mesh/clean.py`: vertices within 1e-6 of the piece's diagonal merged;
  opposite duplicates cancelled by the Breaking Bad reader's rule; separate
  components of at most 12 faces, or under 1% of the area of the largest
  component with more, dropped; a closed surface wound inward turned outward.
- `mesh/decimate.py`: quadric edge collapse (`fast-simplification`) with one
  face budget per object, split over its pieces by area; the budget from the
  target density (median edge over the largest side), corrected by secant
  steps until within 5%. An object within a factor of 2 of the target is not
  touched — which did not keep every synthetic object's shared vertices: see
  the correction below.
- `data/scene.py`: `write_scene`, one fracture mode in the layout
  `SceneReader` reads.
- Not used: a ray-cast shell (`mesh/shell.py`, unchanged). On a synthetic
  piece it kept exactly the faces the cancellation keeps, in 60.6 s against
  23 ms; on Breaking Bad's vase it deleted 14.1% of the largest piece's area,
  all of it real surface.

**Measured** — ten Fractura objects rebuilt in GARF's format from the dumps,
against seven Breaking Bad scenes:

| | real bones (3) | other real (4) | synthetic (3) | Breaking Bad (7) |
|---|---|---|---|---|
| median edge / largest side, cleaned | 0.0015–0.0038 | 0.0022–0.0052 | 0.0092–0.0118 | 0.0064–0.0230 |
| after `--edge 0.0145` | 0.0143–0.0146 | 0.0142–0.0147 | not thinned | — |
| faces flagged by the dihedral test, cleaned → written | 1–9% → 36–51% | 2–4% → 29–34% | 5–51% | 8–55% |
| objects whose pieces share vertices | 0 | 0 | 3 | 7 |

A fixed 99% reduction, split by area, would have left the seven real objects
at 1.00 to 3.66 times Breaking Bad's median edge; the density target leaves
them at 0.98 to 1.02. Thinning kept every vertex within half a target edge of
an original one and the object's size within 0.21%. On the synthetic set,
against break faces marked as GARF's script marks them (all three vertices
coincide with another piece's), the dihedral test's recall is 0.89 and its
precision 0.06 to 0.49.

**Verified.** 557 tests (535 before; `tests/test_external.py`, 22, the script
and `evaluate` end to end among them). Of 32 mutations of the new code, 31
fail a test; the one that does not keeps the last thinning pass instead of the
closest, which on the seven real objects made no difference (two passes each,
the second within 1.6%). 116 version markers. The converted samples evaluate
end to end (`--split all --predictions`, a random-weight model).

**Next.** The trained checkpoints on the converted sets: the real ones at
`--max_fragments 20`, the synthetic one with `--split val`, Fantastic Breaks,
and Breaking Bad's `--jitter/--drop` run beside the real ones.

### Correction (Oct 16): the synthetic breaks pulled apart, and two false successes

The first evaluation on the converted sets (`checkpoint-final`, epoch 117),
compared dump by dump with the run on the unconverted files — the same
checkpoint, the same matched rotations and placement:

| | scenes | part accuracy | geodesic, scored pieces |
|---|---|---|---|
| synthetic, thinned (4 objects) | 29 | 0.659 → 0.047 | 55.9° → 124.0° |
| synthetic, left alone (7 objects) | 64 | 0.607 → 0.800 | 53.5° → 29.7° |
| real bones | 17 | 0.000 → 0.059 | 137.1° → 131.5° |
| Fractura real | 26 | 0.000 → 0.038 | 123.8° → 124.0° |
| Fantastic Breaks | 150 | 0.020 → 0.020 | 128.3° → 132.1° |

- **Thinning pulled the synthetic breaks apart.** Four objects (leg 1248,
  pig 6, pig 7, rib 6889; cleaned median edge 0.0071–0.0080) were under half
  the target, 0.0162 on 200 Breaking Bad scenes, and were thinned one piece at
  a time: 13,834–23,924 faces became 2,706–4,682, the 44,138 vertices their
  pieces shared became 959, and a small piece kept as little as 0.21 of its
  area (one below 0.9 in 7 of the 29 scenes). Matches fell from 1,308 to 200 a
  scene, and the scores to chance. The four were no denser than Breaking Bad:
  against the surface (median edge over √area) all eleven synthetic objects,
  0.0087–0.0134, lie inside the seven Breaking Bad scenes, 0.0080–0.0152, and
  by the largest side Breaking Bad's own spoon is 0.0064. The 64 scenes left
  alone gained from the interior walls' cancellation alone.
- **The two real successes are not reassemblies.** `vert8`'s second piece
  (78,622 faces) is also stored inside its first, as one of its two
  components, in place and facing the same way — the 55% of its vertices that
  looked shared — so its 0.08° is a piece matched to its own copy.
  `pig_bone_27`'s scored piece is 177.9° off (179.6° before); its part
  Chamfer, 0.0052, passes 0.01, and the 220.7 mm² before is 0.0077 in object
  units: it failed only in millimetres.
- **Fantastic Breaks** is at chance both times. Ten objects stay denser than
  half the target after thinning (median edge 0.0007–0.0073, up to 291,117
  faces): their restorations are sliver meshes the collapse barely reduces.

**What changed.**

- `data/external.py`: an assembly whose pieces share at least 1% of their
  vertices (`SHARED_BREAKS`) is not thinned. Measured at full resolution:
  3.2–56% in 91 of the 93 synthetic scenes (none in the other two) and
  4.4–54% in seven Breaking Bad scenes, every coincident face facing its
  twin; none in the 36 real Fractura objects once vert8's copy is gone. The
  decision leaves debris out: the scans' cubes coincide from piece to piece.
- `mesh/clean.py`: `clean_piece(drop_debris=False)` keeps every component;
  debris goes only from pieces that do not share their breaks — on the
  synthetic pieces it was their own geometry (105 islands of 4–58 faces in 18
  scenes). `copies_of_pieces` finds a piece stored again inside another, and
  `prepare` removes the copy (vert8 alone, of the 36 real and 93 synthetic
  assemblies checked).
- The script's summary counts both; each row carries `shares_breaks`,
  `shared_full`, `copies_removed` and `copy_faces`.

**Verified.** Files rebuilt from the dumps, converted at `--edge 0.0162`: the
five synthetic scenes (the four objects and a hip) keep every face and every
shared vertex (1,142–4,787); vert8 loses the copy, shares nothing and is
thinned like the other scans (287,176 faces stored, 22,414 written); three
real bones are thinned with their cubes dropped. 564 tests, clean under
`-W error`; of 47 mutations of the conversion code, 46 fail a test (the
survivor is the one above); 118 version markers.

**Measured after the fix (Oct 16).** `bone_synthetic` and `fractura_real`
converted again and evaluated (`checkpoint-final`, epoch 117, `--split all
--modes_per_scene 0 --max_fragments 0`). Part accuracy per scene on
`bone_synthetic`:

| | scenes | unconverted | first conversion | fixed |
|---|---|---|---|---|
| hip | 14 | 0.421 | 0.820 | 0.813 |
| leg | 6 | 0.716 | 0.000 | 0.883 |
| pig | 45 | 0.725 | 0.595 | 0.876 |
| rib | 17 | 0.595 | 0.500 | 0.847 |
| vertebra | 11 | 0.460 | 0.531 | 0.531 |
| all | 93 | 0.623 | 0.565 | 0.821 |

Geodesic per scene 54.27° → 59.09° → 26.73°, matches a scene 1,225 → 825 →
1,109, share reached 0.710 → 0.573 → 0.791. Per piece the median error is
0.34° and acc@5 0.663 against 66% reached: nearly every piece the chain
reaches is right, and what is missing is reach — in 10 of the 11 vertebra
scenes (meshes and scores as in the first conversion) part accuracy equals the
share reached, none in mode_0 and mode_1, whose second piece touches nothing.
Hip is the one category to fall, 0.820 → 0.813; its only change is the 6-face
island now kept in each of hip 6979's seven scenes, as the Breaking Bad reader
keeps islands. `fractura_real`: part accuracy
0.037 → 0.000 — vert8, without its copy, is wrong like the other scans (the
bones' mean 120.72° → 129.64°; ceramics' and egg's unchanged to the second
decimal), and the embedding term is 0.0000, as on `bone_real`.

**Next.** Breaking Bad's `--jitter/--drop` run beside the scanned sets, whose
pieces share no break vertex.

## 21 · Thesis v7 — breaks whose two sides do not share their vertices (Oct 2026)

**Why.** v7 places Breaking Bad's pieces and Fractura's synthetic bones, and
none of the scanned sets. Measured before any change:

- The trained term, `correspondence_loss`, is InfoNCE over the coincidence
  clusters `find_shared_points` finds (1e-5): one positive per vertex, its
  exact twin across the break, and every other break vertex a negative — the
  one an edge away included. What it teaches is to name single vertices.
- Scanned pieces share no vertex. On the converted meshes the model saw, the
  median gap to the nearest vertex of another piece (among vertices within 0.02
  of one) is 0.0095 on `bone_real`, 0.0111 on `fractura_real` and 0.0055 on
  Fantastic Breaks, against Breaking Bad's median edge of 0.0162; none
  coincides.
- Take the twins away and the model fails on Breaking Bad's own geometry: the
  29 synthetic scenes the first conversion thinned piece by piece fell from part
  accuracy 0.659 to 0.047, and Everyday's validation scenes (2+ pieces) under
  `--jitter 0.01 --drop 0.5` went from 29.41° to 87.74°, acc@5 0.740 to 0.164,
  reached 77.8% to 33.6%. On the scanned sets 0 to 3 of a scene's 221 to 545
  matches end in a verified fit; after §20's fix `fractura_real` is at part
  accuracy 0.000 (129.57°), and the embedding term reads 0.0000 there because
  nothing coincides.
- GARF, the reference (zero-shot on Fantastic Breaks: 10.62°, part accuracy
  91.00%), samples its points on each fragment on its own (Poisson disk), so
  the two sides of a break never share a point in its training either.

**What changed.** Everything is off by default; with it off, v7 trains,
evaluates and resumes as before.

- `nn/losses.py`: `proximity_loss`, InfoNCE whose positives are the vertices
  of other pieces of the same scene within `radius` in the assembled pose, in
  world units; every other vertex of the scene within `band × radius` — the
  other side just beyond the radius, the anchor's own neighbours — is neither a
  positive nor a negative. Anchors are subsampled as in `correspondence_loss`,
  and one whose positives all fell outside the sample is left out; match@1 is
  the share whose nearest candidate is a positive. With exact twins, a radius
  below their spacing and a band of 1 it is `correspondence_loss` (tested).
  `ReassemblyLoss(contact_radius, contact_band)` takes it whenever the batch
  supplies the anchors' positions.
- `mesh/correspondence.py`: `contact_vertices`, per piece, the vertices with
  another piece's vertex within the radius — the anchors. The loader computes
  them on the CPU in the assembled frame (`BreakingBadScenes.contact`); they
  travel as `SceneSample.contact` and `Batch.contact`.
- `training.py`, `_forward`: with `--contact_radius`, the anchors' positions
  are `target_vertices × unit + centroid` (world units), their groups the
  batch's fragment numbers and their scenes its scene numbers. A batch with no
  vertex in contact has no term, as one with no cluster.
- `data/remesh.py`: `remesh_pieces` draws each piece's share of faces from
  `[low, high]` first, collapses it (quadric edge collapse, `mesh/decimate.py`),
  tidies the result as the converter does (`clean_piece`: vertices merged,
  collapsed and repeated faces removed, every component kept, the winding
  kept), then adds the jitter. `--remesh LOW HIGH` and `--remesh_jitter S`
  apply it to the training split only, whenever a scene is loaded, per scene,
  epoch and seed (`_remeshed`); `pieces(i)` gives the meshes as the model got
  them, for the dumps.
- `--init_from CHECKPOINT`: `_start_from` loads the weights alone into a new
  run — a folder means its `best.pt`, else its `last.pt` — refusing an
  architecture that differs (naming the flags) and leaving a pre-v7 head
  behind. A run that resumes from its own checkpoint ignores it, and the
  first epoch's check against chance is skipped after it.
- `--evaluate --remesh LOW HIGH [--remesh_jitter S]` scores remeshed scenes
  (`_remesh0.5-0.9j0.001` in the file names, a report line,
  `summary["remesh"]`); the writer and `scripts/dump_prediction.py` dump the
  remeshed pieces. `remesh` and `remesh_jitter` are data settings of the
  training split only (`_TRAINING_ONLY`): an evaluation never takes them from
  its checkpoint.
- Refused: `--remesh_jitter` without `--contact_radius` (training: nothing
  would coincide, and coincidence is all v7's term reads); `--remesh` with
  `--label_method coincidence` (it finds a break by its coincident vertices);
  shares outside `0 < LOW <= HIGH <= 1`; `--contact_band` below 1.
- Kept: the coincidence clusters are still computed, so `probe_val`'s contact
  matrix reads as before.

**Measured** — `remesh_pieces` on seven Breaking Bad scenes (8,924–31,792
faces, 3–50 pieces), three draws each, at `0.5 0.9` and jitter 0.001:

| | |
|---|---|
| coincident vertices left | none (18–46% without the jitter) |
| median edge | ×1.18–1.48 |
| time a scene | 22–102 ms |
| within 0.015 of another piece: share of the break found / share found on it | 96–100% / 66–96% |
| within 0.01 | 91–100% / 74–97% |

The break is carried over: a remeshed vertex is on it when its nearest original
vertex of its own piece had a twin. On those scenes' 96 pieces the raw collapse
left no degenerate face, but 436 vertex pairs within 1e-6 of a piece's
diagonal (1e-8 to 1.3e-6 apart), which the tidy merges (868 faces dropped); no
piece was wound inward, before or after.

On slabs whose breaks share no vertex but their four corners — six stacks of
two or three, sampled on 9 × 9 and 10 × 10 grids by turns, the nearest vertex
across a break about 0.085 away on grids 0.22–0.25 apart — a model fitted for
120 epochs on the distance-based term (radius 0.15, band 2) placed 8 of the 9
scored pieces within 5° (mean 20.8°; untrained 146.4°). On v7's term the same
run matched the corner twins (match@1 0.89) and left 7 of the 9 wrong (96.8°).
That run is `test_the_model_learns_breaks_that_share_no_vertex`.

**Verified.** 588 tests (564 before; `tests/test_contact.py`, 24), clean
under `-W error`; each of 49 mutations of the new code fails a test; 128
version markers.

**Not yet measured.** No network has been trained with these settings: the
numbers above are of the data and of a toy, not of a result on the scanned
sets. The radius, band and range were set from these measurements, not tuned
on a run.

**Next.** Fine-tune from `checkpoint-final/best.pt` with `--contact_radius
0.015 --remesh 0.5 0.9 --remesh_jitter 0.001`; score it on Everyday's
validation scenes as stored and remeshed, on `bone_synthetic`, and on
`bone_real`, `fractura_real` and Fantastic Breaks.

### Measured, and a correction (Oct 17)

**The first fine-tune** (`checkpoint-contact`, reported by the user): from
`checkpoint-final/best.pt` with `--contact_radius 0.015 --remesh 0.5 0.9
--remesh_jitter 0.001 --lr 4e-4 --lr_min 9e-5`, 40 epochs of 800 steps of 2
scenes; `best.pt` is epoch 40, validation acc@5 0.662 (Everyday val, 2+
pieces, 8 patterns per object — not the scenes `checkpoint-final` validated
on, 2–20 pieces and 20 patterns per object, so not comparable with its
logged numbers). Scored with `--split all --modes_per_scene 0
--max_fragments 0`:

- `fractura_real`: part accuracy 0.056 (0.000 before), geodesic 129.62° per
  piece (129.57° before) and 131.04° per scene, acc@5, acc@10 and acc@30 all
  0.000 over 67 pieces, match@1 0.009, 412 matches a scene of which 6 are in
  verified fits; the embedding term, now computed by distance, 6.72. Its part
  accuracy is not a success: no piece is within 30°, and the scenes that score
  have wrongly turned pieces that pass the Chamfer test (`limb4` 1.0 at
  132.4°, `limb3` 0.5 at 124.0°).
- `bone_synthetic`: part accuracy 0.804 (0.821), geodesic per scene 28.22°
  (26.73°), per piece median 0.95° (0.34°) and acc@5 0.638 (0.663), matches a
  scene 944 (1,109), 599 verified (629), reached 0.781 (0.791); by category
  leg 0.883 (0.883), pig 0.856 (0.876), rib 0.840 (0.847), hip 0.784 (0.813),
  vertebra 0.519 (0.531). These pieces still share their breaks; trained to
  match within 0.015 rather than the twin, the fits are less exact.

**Correction: the jitter made the fracture labels.** The dihedral test reads
the jitter as fracture. On the seven Breaking Bad scenes, with training's
labels (`--sharp_threshold 0.9`), the share of vertices flagged:

| | as stored | remeshed | jitter 1e-5 | 1e-4 | 3e-4 | 1e-3 |
|---|---|---|---|---|---|---|
| flagged as fracture | 13–76% | 14–80% | 19–82% | 41–89% | 64–95% | 90–99.5% |
| coincident vertices left | 370–4,501 | 74–1,462 | 8–136 | 0–2 | 0 | 0 |

The labels place the cross-attention tokens and bound the vertices stage two
matches, so `checkpoint-contact` was trained with tokens over nearly the whole
surface while its validation, and every test, used the usual labels; and
`--evaluate --remesh ... --remesh_jitter 0.001` matched over nearly the whole
surface. Fixed: `data/remesh.py` splits `remesh_pieces` into
`collapse_pieces` and `jitter_pieces` — the same draws, so its output is
unchanged (identical on the seven scenes, two seeds and three settings) — and
`BreakingBadScenes` computes the labels on the collapsed pieces, before the
jitter, and everything else on the moved ones; `pieces` holds the moved ones.
The remeshing measurements above are unaffected. The labels were not measured
before the jitter was chosen; they should have been.

**The scans' breaks do not close.** Measured on the fine-tune's
`fractura_real` dumps (the meshes the model got, in the assembled pose), on the
seven Breaking Bad scenes as stored and as training remeshed them, and on one
`bone_real` object, `pig_bone_1` (the user's one-object copy of the HDF5:
1,650,918 faces, median edge 0.0014), at full resolution and thinned to 0.0162
as the converter thins (both read and cleaned by `data/external.py`); per
vertex, the distance to the other pieces' surface, in units of the object's
largest side:

| | Breaking Bad, stored | remeshed (training) | `fractura_real`, thinned | `pig_bone_1`, full | `pig_bone_1`, thinned |
|---|---|---|---|---|---|
| median edge | 0.0064–0.023 | 0.0082–0.028 | 0.016–0.017 | 0.0014 | 0.016 |
| of the vertices within 0.015, the share within 0.005 | 79–97% | 77–97% | 20–55% | 48% | 50% |
| of the vertices within 0.05: the gap's first quartile | 0 | 0.0006–0.0013 | 0.0038–0.0175 | 0.0054 | 0.0053 |
| of the vertices within 0.05: the median gap | 0–0.010 | 0.0015–0.011 | 0.011–0.029 | 0.015 | 0.016 |

- The thinning does not make the gap: on `pig_bone_1` every row is the same
  at full resolution and thinned.
- Nor does an offset of the stored pose: moving each piece of `pig_bone_1`
  rigidly onto the others (point-to-point ICP on its vertices within 0.05, 60
  iterations, full resolution) takes their median gap from 0.012–0.019 only to
  0.007–0.009, by motions of 1.3–12.7° and 0.015–0.070, still sliding.
- So the scanned surfaces themselves do not fit to within a few thousandths of
  the object; on Breaking Bad, remeshed or not, the nearest quarter of the
  facing vertices is within 0.0013. The fine-tune trained on that.
- This table replaces a first version measured to the nearest vertex of
  another piece rather than its surface. Without twins, that measure grows
  with the edge: on `pig_bone_1` thinned it put 13% of the contact within
  0.005 against 50% to the surface, and the first version gave
  `fractura_real` 5–22% where the surface gives 20–55%.

Also wrong, and not fixed: on a converted set the report's "N of them shapes
the checkpoint was trained on" (20 of 27, 75 of 93) counts the objects the
hashed split would put in training — there are no Breaking Bad split lists
under `external` — though none of them was in training. No score depends on it.

**Verified.** 589 tests, clean under `-W error`
(`test_the_noise_moves_the_vertices_not_the_fracture_labels`; `pieces` checked
against the scene's own vertices); each of 52 mutations of §21's code fails a test; 130 version markers.

**Next.** Training for scans needs Breaking Bad breaks that open like these
— the two sides of a break moved apart by gaps distributed like the scans',
checked against the table above before a run — and a contact radius that
reaches across them: 0.015 takes in half of the scans' vertices within 0.05.
Done the same day, below.

### Opening the breaks (Oct 17)

**What changed.** `--remesh_gap G` (with `--remesh`; off by default):

- `data/remesh.py`, `open_pieces`, between the collapse and the jitter. A
  scene draws a mean opening from `[0, G]`; each piece's break vertices
  (`break_masks`: those whose nearest stored vertex had a twin) move along the
  break's normal by `s(x) = m (1 + g(x))`, `m` half the scene's draw, `g` a
  smooth unit-variance random field (32 random Fourier features of a Gaussian
  of length 0.026), one per piece — inward where `s` is positive, outward into
  the other piece where negative — so the two sides stand `2m` apart on
  average and touch or overlap in places. The length is the scans': the gap of
  two vertices of a `fractura_real` piece correlates 0.96 at 0–0.01 apart, 0.84
  at 0.01–0.02, 0.51 at 0.02–0.04 and 0.02 at 0.04–0.08 (medians over the 27
  scenes); the field, 0.85 at 0.015, 0.51 at 0.03, 0.07 at 0.06.
- The direction is the break's normal averaged over the break with a Gaussian
  of width 0.026; the vertices within 3|s| of the break move as their nearest
  break vertex does, by a smoothstep weight from 1 at the break to 0 there;
  and wherever a face would still turn over, its corners move half as far,
  again until none does (zero after eight halvings; a face of no area is left
  out).
- `BreakingBadScenes._remeshed` returns the collapsed, the opened and the
  moved pieces: the fracture labels from the opened ones (before the jitter),
  the contact mask from the collapsed ones, and the term's positives found
  where the vertices were before the opening — `SceneSample.contact_points`,
  `Batch.contact_points`, read by `_forward` — so the two sides of an opened
  break stay each other's positives however far apart they move. A scene that
  is not remeshed carries none, and its positives are where its vertices are,
  as before.
- `--evaluate --remesh LOW HIGH --remesh_gap G` scores opened scenes (`g0.03`
  in the file names, the report line, `summary["remesh"]["gap"]`); refused
  without `--remesh`, or below 0; training refuses it without
  `--contact_radius`; never taken from a checkpoint (`_TRAINING_ONLY`).

**Measured** — five draws of each of the seven scenes at `--remesh 0.5 0.9
--remesh_jitter 0.001 --remesh_gap 0.03`, measured as the scans were (the
distance to the other pieces' surface, among the vertices within 0.05 of it);
the draws opened by 0.01 or more (a third of the scenes open by less):

| | six scenes (18 draws) | the spoon, 50 pieces (5) | `fractura_real` (27 scenes) |
|---|---|---|---|
| of the vertices within 0.015, the share within 0.005 | 26–57% (median 34%) | 49–61% | 20–55% (39%) |
| the gap's first quartile | 0.0047–0.0130 (0.0088) | 0.0019–0.0031 | 0.0038–0.0175 (0.0081) |
| the median gap | 0.0117–0.0226 (0.0190) | 0.0052–0.0075 | 0.011–0.029 (0.0184) |

A spread of 1.8 instead of 1 gave the six scenes 42–57% (52%) and a first
quartile of 0.0034–0.0069 (0.0045), farther from the scans. The spoon's pieces
are too thin for the step, and the guard holds them back.

Faces turned over (normal more than 90° from where it was), on the same 23
draws, by three versions of the opening, before the guard:

| | the break's faces | the faces beside it | all faces |
|---|---|---|---|
| each vertex along its own normal | 8–25% | 14–49% | 1.0–26% |
| along the averaged normal | 0–5% | 13–46% | 0.7–17% |
| and the vertices beside it moving with it | 0–5% | 0.5–9% | 0.03–2% (six), 4–9% (spoon) |

After the guard none is; it keeps 55–96% of the step on the six scenes and
29–54% on the spoon, and 0.1–1.6% of the faces turn by more than 60° (the
spoon, 3.7–6.3%). Over all 35 draws: the share of vertices labelled fracture
goes from 13–85% to 15–88% — at most 2.2 points more on the six, 5.6–8.4 on
the spoon; the positives, from the pieces before the opening, still find
96–100% of the break; the opening takes 13–85 ms a scene (the spoon,
140–171).

**Not yet measured.** No network has been trained with the opening.

**Verified.** 596 tests, clean under `-W error` (seven new in
`tests/test_contact.py`: the flat break opened along its normal, a field per
side, the mean opening; the vertices beside a break; no face turned over on a
rough break, where moved as asked 164 would be, and a face of no area left
alone; the break carried over to the collapse; the dataset's positives where
the pieces met; `contact_points` through the batch and the forward pass; the
script handing `--remesh_gap` to an evaluation); each of 85 mutations of §21's
code fails a test (one only after a test was added for it: on the test
scenes, whose breaks are open rims the collapse keeps, breaks found on the
collapse looked the same); 134 version markers.

**Next.** Fine-tune from `checkpoint-final/best.pt` (not `checkpoint-contact`,
whose labels were the jitter's) with `--contact_radius 0.015 --remesh 0.5 0.9
--remesh_jitter 0.001 --remesh_gap 0.03`; score it on Everyday's validation
scenes as stored and opened, on `bone_synthetic`, and on `bone_real`,
`fractura_real` and Fantastic Breaks.

### The opened fine-tune, and where real breaks stand (Oct 18)

**The run** (`checkpoint-open`, reported by the user): from
`checkpoint-final/best.pt` with `--contact_radius 0.015 --remesh 0.5 0.9
--remesh_jitter 0.001 --remesh_gap 0.03 --lr 4e-4 --lr_min 9e-5`, 80 epochs of
800 steps of 2 scenes; `best.pt` is epoch 65, validation acc@5 0.624 (Everyday
val as stored, 2+ pieces, 8 patterns per object). Scored with `--split all
--modes_per_scene 0 --max_fragments 0` unless named:

| | match@1 | acc@5 | acc@30 | part accuracy | matches a scene (in verified fits) |
|---|---|---|---|---|---|
| Everyday val, as stored (epoch 80's validation) | 0.534 | 0.608 | -- | -- | -- |
| Everyday val, opened (`--evaluate --remesh 0.5 0.9 --remesh_jitter 0.001 --remesh_gap 0.03`) | 0.323 | 0.161 | 0.308 | 0.745 | 193 (65) |
| `bone_synthetic`, as converted (breaks shared) | 0.676 | 0.638 | 0.650 | 0.799 | 871 (529) |
| `bone_synthetic`, opened | 0.551 | 0.212 | 0.493 | 0.694 | 549 (209) |
| `fractura_real` | 0.012 | 0.000 | 0.000 | 0.037 | 424 (4) |
| `bone_real` | 0.017 | 0.000 | 0.000 | 0.019 | 255 (2) |
| Fantastic Breaks (`FB_v1`, 150) | 0.012 | 0.000 | 0.000 | 0.027 | 173 (1) |

- The network matches synthetic breaks that share no vertex: opened, the
  first candidate is the partner for 32% (Everyday) and 55% (`bone_synthetic`)
  of the anchors, against 1.2-1.7% on the three scanned sets. Sharing no
  vertex is not why the scans fail; their breaks are.
- Opened breaks cost precision where matching holds: acc@5 0.61-0.64 as stored,
  0.16-0.21 opened, while acc@30 and part accuracy fall less. The matched
  points are a gap apart, so the fitted rotation is a few degrees off; fewer
  pairs pass the verification (reached 36% against 75% on Everyday val).
- The scans are clean meshes: all 94 pieces of the 27 `fractura_real` scenes
  are closed, of one component and wound outward (16 have a few non-manifold
  edges; Breaking Bad's 96 pieces, none) -- not a conversion fault.
- The dihedral labels, which place the tokens and bound the matching, find
  the scanned bones' breaks badly. Per scene, the break taken as the vertices
  within 0.03 of another piece's surface in the assembled pose: on the 16
  bones they label 52-87% of the surface, 7-33% is near a break, and 8-35% of
  the labelled vertices are on it (recall 67-92%); on the 11 ceramics and eggs,
  38-70% (recall 43-79%); on the seven Breaking Bad scenes as stored, 18-98%
  (median 38%, recall 69-97%), and as training opens them (two draws each),
  14-98% (34%, recall 76-99%). The ceramics and eggs fail as completely as the
  bones, so the labels are not the main cause.

**Next: real breaks in training.** Fantastic Breaks trains as it is: no
official lists, so `--split_source hash --val_frac 0.2 --test_frac 0.2` splits
its 150 objects (15 categories) by object into 87 train, 30 val and 33 test
(seed 0). Its pieces share no vertex, so the positives are by distance in the
assembled pose and the opening finds nothing to open; `--remesh 0.7 1.0`
keeps its training density near the scans': median edge over largest side
0.0161 as converted (30 scenes; 0.0041-0.0167), 0.0171 at 0.7-1.0, 0.0186 at
0.5-0.9, against `fractura_real`'s 0.0162 (0.0159-0.0170). `--steps_per_epoch
0` makes an epoch one pass. Checked here on CPU with the converted pieces
rebuilt from the Visualizer's dumps and a stand-in checkpoint of the same
architecture (2,754,784 parameters): the training starts from the weights
(87 train / 30 val, hashed split, positives within 0.015, remeshed 70%-100%),
an evaluation of the 33 test objects takes the split from the checkpoint, and
one of Everyday val needs `--data_from_flags --split_source official`
(otherwise it takes `--data_subsets FB_v1` from the checkpoint and finds no
scene). acc@5 picks `best.pt` and stays at its first epoch while it reads 0,
so `last.pt` is scored too. The commands are in the README ("Real breaks").
Not mixed with Breaking Bad yet: Breaking Bad's official split leaves every
object it does not list out of every split, so a mixed run needs code; worth
it if this one moves the scanned sets.

## 22 · Sim to real — true breaks, scan-like training, a fracture head (Oct 2026)

**Why.** Every run so far, the contact-radius fine-tune included, reaches
chance on the scanned sets (§21). This section is the plan from the review of
those runs, built. Every part is off by default and an old checkpoint
evaluates as it did, except that candidates are now spread over the break
(below). `docs/SIM2REAL.md` has the details and the run plan.

1. **The oracle.** `convert_external` writes each scan's marked break
   (`fractured_0/break_vertices.npy`; Fantastic Breaks' restoration takes the
   vertices within 0.03 of the marked break). `--label_method marked` makes
   the tokens and the matched vertices that true break, and with `--evaluate`
   it overrides the checkpoint's labels. §21 found the ceramics and eggs
   failing as completely as the bones and concluded that the labels are not
   the main cause; the oracle tests that directly.
2. **Stage two.** `--match_metric cosine` matches by direction, as the term
   trains. It is not the default because it measured worse: the slab model of
   `tests/test_contact.py`, trained once per seed and scored both ways, got
   20.8 deg against 59.8 deg (acc@5 0.89 against 0.67) on seed 0 and equal
   results on seeds 1 and 2 (9 pieces each). Candidates above 2,048 per
   fragment are now thinned by farthest-point sampling (unchanged by a turn of
   the fragment) instead of a stride through the vertex order.
3. **Scan-like training scenes.**
   - `--remesh_edge LOW HIGH` remeshes isotropically, using pymeshlab's
     explicit remesher with three passes; a 20k-face test mesh took 0.47 s,
     against 0.74 s for five passes, with the same edge spread.
   - Its crease preservation is off, because kept creases keep the shared
     vertices: of the fractured solid's 1,152, 1,350 still coincided at the
     default 30 degrees, 686 at 90 and 18 at 180.
   - `convert_external --isotropic` remeshes the scans to the same edge.
   - `--remesh_smooth` (Laplacian wear of the break band, capped at a quarter
     edge per step), `--remesh_noise` (normal ripple correlated over 0.02) and
     `--remesh_deform` (a slow bend per piece) join the opening and the jitter.
     The labels are read before the ripple and the jitter, and the positives
     before anything moves.
   - `--remesh_probability` and `--augment_probability`: one draw per scene.
4. **The fracture head** (`segment` in `--schedule`):
   - it reads VN invariants and is trained on the true break with a
     class-balanced focal loss, which reads 0.173 untrained;
   - it picks the tokens on the device, split by predicted-break counts and
     spread by farthest-point sampling, with ties broken by vertex order so a
     per-fragment turn changes nothing (float64);
   - half of the training batches take their tokens from the truth
     (`--segment_teacher`);
   - stage two matches the predicted break;
   - every scored epoch prints its precision and recall beside the dihedral
     labels'.
5. **Consistency** (`--w_consistency`). Each changed scene is also run as
   stored without a gradient, and its break vertices' embeddings are pulled
   toward their stored partners'. Between a scene and the same scene rotated
   such a term is identically zero, since the embedding is invariant.
6. **Fragment context** (`global` in `--schedule`): an equivariant pool per
   fragment, handed back to its vertices.
7. **`--init_partial`**: a v7 run's weights carried into such a schedule.

**The samples** are measured in `docs/SIM2REAL.md` §7:

- five Breaking Bad scenes;
- Fantastic Breaks' `00002`, decimated to 5% and stored one `.npz` per object;
- five Fractura real bones: `pig_bone_1`, and `limb1`, `limb4`, `vert1` and
  `vert6` from `fractura_real.hdf5`.

What they show:

- Breaking Bad's edges spread 11–16× (the scans' 3–4×), with 36–50% slivers,
  and every scene shares vertices across its breaks.
- The real pieces stand apart. Fractura's are a median 0.005–0.019 apart across
  the estimated break, and only 1–3% of the four solid bones' vertices are within
  0.005 of the other piece. The Fantastic Breaks pair is flush: the
  restoration's break lies a median 0.0003 from the broken piece's surface.
- **Roughness, corrected.** The plane-fit measure first used took in the far
  side of thin walls and the turned facets of narrow breaks. It read the pig
  bone's intact surface as rougher than its break, and Fantastic Breaks' break
  as 2–3.5× rougher than Breaking Bad's. The corrected measure counts only
  neighbours facing within 60° and fits a quadric, at the resolution the model
  reads (edge about 0.0145). With it:
  - §6.3's settings cover the intact surfaces: theirs measure 0.00022–0.00057,
    the bones' 0.00021–0.00037 and Fantastic Breaks' 0.00006.
  - The scans' breaks are as rough as the training breaks (0.00031–0.00058) or
    rougher. Three bones measure 0.00040–0.00044; `pig_bone_1`, `limb4` and
    Fantastic Breaks measure 0.00081–0.00118. No setting closes this gap. A
    detector forgives that direction, and the fracture head's report on the
    scans measures it.
  - Two of the five Breaking Bad scenes have intact surfaces as textured as
    their breaks, so roughness alone cannot tell break from intact even in
    training.
- The restoration is a boolean cut's mesh: 91% slivers, 341 edges shared by four
  or six faces. No remesher setting could even it out, so a remesh worse than
  its input is now refused. The piece is collapsed to the density instead, in
  training and in the converter, which counts it.
- Training remeshes without the remesher's surface check, at 0.50 s a scene
  against 1.38 s and the same edges.
- §6.3's settings bring the first Breaking Bad scene's edge spread to 1.8–1.9 and
  open its contact to a median 0.0085–0.0138.

**The real bones, and what they changed in the converter:**

- **None marks its break.** `pig_bone_1`'s `shared_faces` mark 24 debris faces
  per piece, the same faces in all four; the other four mark nothing, and their
  `full_mesh` is only the pieces put together. An assembly that marks no break
  now has one estimated from the pose: the faces whose nearest face of another
  piece is within 0.03 and faces them (face normals' dot below −0.5).
  - Face normals, because a vertex on a rim averages the break and the intact
    surface. Judged by vertex normals, recall was 0.11–0.62 on the Breaking Bad
    scenes and 0.21 on Fantastic Breaks.
  - By faces, it gives precision 0.91–0.98 and recall 1.00 on the five Breaking
    Bad scenes (mirror-image breaks, so easy), and 0.94 and 0.67 on Fantastic
    Breaks, a scan.
  - Drawn, it covers the limb bones' broken ends, the vertebrae's pedicles and
    the shards' split edges. Where pieces stand more than 0.03 apart it misses
    some: at 0.05 the vertebrae's and `limb4`'s estimates grow by a quarter to
    two thirds.
  - `break_source` in the CSV says which kind each assembly has.
- **A dense scan is thinned before remeshing by a quadric collapse that keeps
  the topology.** fast-simplification left 12–173 non-manifold edges per piece of
  `pig_bone_1`, which the remesher opened into 360 boundary edges. Now all five
  bones convert closed and manifold, in 4–72 s each.
- **Walls vary.** `pig_bone_1`'s shards are 0.012–0.016 thick, the other bones
  0.03–0.10. Converted at 0.0145, 43–71% of their break vertices are one edge
  from the intact side, against 81–83% in Breaking Bad's training scenes. So no
  finer edge is called for.
- **The dihedral test floods on them.** At the converted resolution it flags
  30–86% of each piece's vertices, with precision 0.07–0.41 against the
  estimate. At full resolution its precision is 0.03–0.27.

**Fantastic Breaks stored one `.npz` per object**, the sample's layout
(`num_pieces`, `piece_<i>_vertices`, `piece_<i>_faces`,
`piece_<i>_shared_faces`), is now read by `convert_external`, given one such
file or a folder of them.

**V8's run** (`docs/V8-RUN.md`) trains this from scratch: four intra layers,
the fracture head and the fragment context, three cross–intra pairs, on Everyday
with 3 stored and 5 perturbed scenes in every optimizer step of 8 (the 48/80 of
128 planned). A first run in steps of 128 stayed flat for 30 epochs: 750 steps,
2% of v7's. On the synthetic slabs the schedule itself learns (four seeds: 1–66°
against v7's 21–91° on dihedral labels), so the steps went back to v7's size.
Two changes make the split exact and reportable:

- **The sampler deals each scene its share** (`DealtShares`, `data/sampling.py`).
  Every step of `batch_size × accumulate` scenes on one GPU gets its shares
  spread evenly, so the split is exact in every step. One draw per scene gives
  80 of 128 give or take 5.5. The banner prints the counts per step.
- **The converter's report** prints, per subset, the break each assembly is
  scored against (marked or estimated) and the dihedral labels' precision and
  recall against it, at full resolution and converted. Where the dataset marks
  its break, it also prints the estimate's own precision and recall against the
  marks. The CSV has all of it per assembly.

The sequence was run end to end on the samples, on CPU and scaled down:

1. both datasets converted into their own folders;
2. preflight and one epoch of training with V8's flags;
3. Everyday val and an unseen subset scored, with their predictions zipped;
4. the converted Fractura objects and Fantastic Breaks scored, the oracle
   included;
5. figures and per-category tables drawn from all of it.

**Checked here.** No Breaking Bad dataset and no GPU are available in this
environment, beyond the samples above. `tests/test_sim2real.py` (28 tests) and
the whole suite, slow tests included, pass on CPU, and `--preflight` runs clean
with every change on the tests' fixture. No model has been trained or scored on
Breaking Bad or on the scans with these changes yet; the first numbers come from
the run in `docs/V8-RUN.md`.

### Why V8's runs stayed at chance (Oct 22)

**The run** (reported by the user): V8's flags in steps of 8 (`--batch_size 8
--micro_batch_scenes 4 --steps_per_epoch 400`, learning rate 1e-3), from
scratch, on one GPU. After 32 epochs, 12,800 steps:

- validation geodesic 126.6–126.9° in every epoch (chance 126.5°), acc@5 0.000,
  match@1 0.004–0.006;
- the training embedding term 6.41 → 5.88, the validation one 6.37 → 5.76;
- the fracture head precision 0.26 → 0.30–0.59, recall 0.62–0.99 (the dihedral
  labels: 0.35 and 1.00).

The first run, in steps of 128, was as flat. Sixteen times the steps did not
move it, so the steps were not what held it.

**Ruled out, on the Breaking Bad samples.** Four of the five have 2–20 pieces
(Bottle 15, Bowl 10, Cup 7, Mug 2).

- **The positives.** An embedding made from the true assembled positions
  scores match@1 1.00 on each scene, stored and remeshed. With four scenes in
  one pass it scores 0.97–0.99.
- **The other terms crowding out the embedding.** Untrained, the embedding
  term's gradient has norm 16–58, the head's 1–5 and the consistency term's
  0–2.4.
- **V8's schedule.** Trained on the four scenes as stored, two a step, it
  reached match@1 0.039 at step 175. v7's schedule reached 0.040 with the
  coincidence term and 0.034 with positives by distance.

**Cause 1: the anchors were drawn per pass, not per scene.** The contrastive
term drew 1,024 anchors from the whole forward pass. Every other anchor of the
pass was a negative. v7's runs had one scene a pass; V8's plan put four in each.
An anchor trains only if one of its positives is drawn with it, so four scenes
sharing a budget lost more than three quarters of what trains (SIM2REAL §2.4):

| | anchors with a drawn positive |
|---|---|
| one scene a pass, as stored | 3,176 |
| four in one pass, as stored | 643 (Bowl 38, Mug 15) |
| one scene a pass, remeshed | 2,971 |
| four in one pass, remeshed | 427 |

Three quarters of each anchor's negatives were also other objects' vertices,
which matching never compares. **Fixed:** both terms are computed scene by
scene. Each scene draws its own anchors, its anchors' negatives are its own
vertices, and its mean is weighted by its scored fragments. A pass of several
scenes now gives the step that one pass of each gives: the relative difference
is 5e-7, float32 round-off. `test_one_pass_of_two_scenes_steps_as_two_passes_of_one`
checks it; its predecessor asserted that the two differ, calling that "the
better behaviour". A pass of one scene is unchanged, its draw included, so
v7's runs and checkpoints are unaffected. With several scenes a pass, the
logged embedding term and match@1 now read per scene. Without other scenes'
negatives, the same model reads a lower term and a higher match@1 than
before, so the old logs cannot be compared with the new ones.

**Cause 2: what the stored scenes teach.** The same harness shows what each
training mix learns:

- V8's schedule at 32 channels, 512 tokens, two scenes a step, learning rate
  1e-3 on a cosine over 400 steps;
- scored every 25 steps at the 0.015 radius, on the four scenes as stored, on
  one fixed remeshed and perturbed draw of them, and on one fixed remeshed
  draw. Neither draw is ever trained on.

Chance is about 0.002–0.006.

| trained on (two scenes a step) | step | as stored | remeshed and perturbed | remeshed, unperturbed |
|---|---|---|---|---|
| as stored only | 200 | 0.055 (term 5.54 → 4.21) | 0.002 (term 6.14 → 8.20) | |
| 3 as stored, 5 remeshed and perturbed in 8 | 225 | 0.013–0.025 | 0.004 (term 5.75) | |
| all remeshed and perturbed | 200 | 0.005 | 0.005 (term 5.64) | |
| the same, positives within 0.03 | 125 | 0.005 | 0.002 | 0.001 |
| all remeshed, none perturbed | 400 | 0.016 | | 0.034–0.040 (term 6.10 → 4.92) |
| all remeshed, 5 in 8 perturbed | 150 | 0.005 | 0.004–0.008 | 0.002–0.008 (term 5.75) |

- **Stored scenes teach the mesh.** Their two sides share every break vertex
  and triangle, so a network can match them by tessellation alone, with no
  notion of shape. Trained on them only, it matched them better and better
  while it matched the same scenes meshed on their own worse than untrained:
  the term rose from 6.14 to 8.20. This is v7's record: match@1 0.53–0.68 on
  Breaking Bad and `bone_synthetic`, 0.012–0.017 on the scans (§21).
- **A new tessellation alone is learnable.** Remeshed and nothing else, the
  network learned to match pieces meshed on their own. On those, match@1 rose
  0.004 → 0.034–0.040 in 400 steps, and the scenes as stored rose with it
  (0.016), stored being just another mesh.
- **Every perturbation from the first step is not, yet.** Remeshed and
  perturbed, the scenes barely moved it in 200 steps. With V8's 3 stored in
  8, the stored ones carried what learning there was, and that was the mesh.
  Positives within 0.03 instead of 0.015 did not help.

**The new setting: every training scene remeshed, shape first.** Two stages
(`docs/V8-RUN.md` §2):

1. 12 epochs remeshed with nothing perturbed (`--remesh_probability 1
   --augment_probability 0`).
2. 28 epochs of the planned mix from stage 1's weights (`--augment_probability
   0.625 --w_consistency 0.5 --init_from`). That is 48 unperturbed and 80
   perturbed of every 128, with no stored scene among them.

`--val_remesh` validates both on the validation scenes remeshed the same way,
with one draw per scene repeated every epoch, so `best.pt` rewards matching by
shape (SIM2REAL §2.5).

The mix alone, from scratch, was slow. By step 150 it had barely moved its term
on the remeshed copies (5.83 → 5.75), with match@1 at noise level, while
remeshing alone had lowered its term from 6.10 to 5.30 and went on to 0.034–0.040
by step 400 (the mix was not run that far). Hence shape first. Stage 2 itself
is untested. The harness only
showed that stored-first is no start: stored-only weights carried into the mix
lost their stored match@1 within 25 steps (0.055 → 0.012), since they were the
mesh.

**Not measured here.** These are memorization checks on four scenes, a 32-channel
model and a few hundred steps. They rank the settings by how fast each starts
learning. They say nothing of what a full run reaches on unseen objects or on
the scans.

**Checked here.** 630 tests pass, slow ones included, and 162 version markers.
The new tests:

- one pass of two scenes steps as two passes of one (`tests/test_training.py`);
- another scene's vertices are neither positive nor negative, each scene keeps
  its own anchors, and the scene weights weight each scene's mean
  (`tests/test_contact.py`);
- `--val_remesh` remeshes every validation scene by one draw that every epoch
  repeats (`tests/test_sim2real.py`).

The harness scripts are not in the repository.
