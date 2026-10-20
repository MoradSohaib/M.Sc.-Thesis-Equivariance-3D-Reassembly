# Sim to real: the changes after v7, and how to train and test them

V7, trained on Breaking Bad (Everyday), reassembles unseen Breaking Bad scenes
and Fractura's synthetic bones well, and reaches chance on scanned pieces
(Fractura's real bones, Fantastic Breaks). This document covers the changes made
to attack that gap, in the order they were planned, and then the runs that test
them.

Every change is **off by default**. With no new flag, training, evaluation and
`probe_val.py` do exactly what v7 did. The one exception: stage two now spreads
its matching candidates over the break (§1.2), and that only applies to
fragments with more than 2,048 candidates. An old checkpoint loads and evaluates
as before.

| | What | Flag(s) | Where |
|---|---|---|---|
| 1 | Matching metric for stage two | `--match_metric euclidean\|cosine` | `assembly/translation.py` |
| 1 | Candidates spread over the break, not strided by index | (always) | `assembly/translation.py` |
| 1 | True breaks written beside converted scenes; the oracle | `--label_method marked` | `data/scene.py`, `data/external.py`, `training.py` |
| 2 | Isotropic remeshing, in training and conversion | `--remesh_edge LOW HIGH`, `convert_external --isotropic` | `data/remesh.py`, `data/external.py` |
| 2 | Breaks worn, rippled, bent (plus v7's opening and jitter) | `--remesh_smooth`, `--remesh_noise`, `--remesh_deform` | `data/remesh.py` |
| 2 | One draw per scene: stored, remeshed, or perturbed | `--remesh_probability`, `--augment_probability` | `training.py` |
| 3 | Fracture head: learns the break, picks the tokens and the matched vertices | `segment` in `--schedule`, `--w_segment`, `--segment_gamma`, `--segment_teacher`, `--segment_threshold` | `nn/context.py`, `nn/model.py`, `training.py` |
| 4 | Consistency between a scene and itself as stored | `--w_consistency` | `nn/losses.py`, `training.py` |
| 5 | Fragment context for every vertex | `global` in `--schedule` | `nn/context.py` |
| – | Carry a v7 run's weights into the new schedule | `--init_partial` with `--init_from` | `training.py` |

---

## 1. Stage two: the metric, the candidates, the oracle

### 1.1 `--match_metric` (default `euclidean`)

The contrastive term compares embeddings by cosine, so nothing trains their
length. Stage two matched by Euclidean distance on the raw embeddings, which
also weighs the length. `--match_metric cosine` matches by direction alone.

It is **not** the default, because it was measured and did not win. One
model was trained on the synthetic slabs of `tests/test_contact.py`, whose breaks
share no vertex, and then scored both ways:

| seed | Euclidean: mean / acc@5 | cosine: mean / acc@5 |
|---|---|---|
| 0 | 20.8° / 0.89 | 59.8° / 0.67 |
| 1 | 2.0° / 0.89 | 2.1° / 0.89 |
| 2 | 39.7° / 0.78 | 39.3° / 0.78 |

Each seed scores 9 pieces, so seed 0 differs by 2 pieces. The length is not
noise: it is the same function of the geometry on both sides of a break.

The setting only affects inference and is never taken from a checkpoint. So the
same checkpoint can be scored both ways on the scans, where lengths may drift
(§6.2).

### 1.2 Candidates spread over the break

Stage two matches at most 2,048 vertices per fragment. V7 thinned a larger
candidate set by striding through the vertex order, which follows the scanner on
a dense scan, so some regions were kept and others skipped. Now farthest-point
sampling spreads the subset evenly over the break.

It is deterministic and does not depend on the fragment's pose: it starts at the
vertex farthest from the candidates' centroid. Breaking Bad fragments rarely
have more than 2,048 candidates, so their numbers do not move.

### 1.3 The true break, and the oracle (`--label_method marked`)

`scripts/convert_external.py` now writes `fractured_0/break_vertices.npy` beside
each converted scene: one bool per vertex. It holds the vertices of the faces
the dataset marks as break:

- Fractura: `shared_faces`.
- Fantastic Breaks: `meta_<k>.npz`'s `mask`.

The marks are carried through the cleaning and thinning. Fantastic Breaks marks
the broken object only, so the restoration takes the vertices within 0.03 of the
marked break (`BREAK_REACH`); real gaps are 0.011–0.029. The CSV report gains
`marked_pieces` and `marked_vertices`. **Scenes converted before this change
have no such file and must be converted again.**

`--label_method marked` uses that true break, wherever it is known. On
converted scenes it is the dataset's marks. On Breaking Bad it is the coincident
vertices, carried over by nearest vertex to remeshed pieces (which
`coincidence` cannot label). With `--evaluate`, an explicit `--label_method`
overrides the checkpoint's. **This is the oracle experiment:** the
cross-fragment tokens and the vertices stage two matches are the true break,
whatever the checkpoint would have labelled. Read it like this:

- If the scans' matches recover under the oracle, the fracture labels were what
  failed on scans, and the fracture head (§3) is the fix.
- If they stay at chance, the descriptors are the problem, and §2 and §4 come
  first.

---

## 2. The scans' geometry, in training

### 2.1 Isotropic remeshing (`--remesh_edge LOW HIGH`; `convert_external --isotropic`)

An intra layer reaches one ring of the mesh. So what the network sees around a
vertex is so many *edges* wide, and its physical size follows the tessellation.
Breaking Bad's pieces come from a simulation, with long thin triangles; scans are
meshed evenly.

`--remesh_edge LOW HIGH` remeshes each piece of every training scene
isotropically, on its own, to an edge length drawn per piece from `[LOW, HIGH]`
in world units. It replaces `--remesh`'s quadric collapse; the two cannot be
combined. `scripts/convert_external.py --isotropic` remeshes the scans the same
way. It is added to `--reference DATA` or `--edge E`, and remeshes to that edge
instead of thinning. Seen at one edge length, Breaking Bad and the scans have
one scale and one triangulation. `0.012 0.018` brackets Breaking Bad's median
edge, 0.0145 of an object.

This uses pymeshlab's explicit isotropic remesher, with three passes in training
and five in conversion. On a 20k-face test mesh, three passes took 0.47 s and
five took 0.74 s, with the same edge spread.

Crease preservation is **off** (feature angle 180°), because preserving creases
keeps the shared vertices. On the tests' fractured solid (1,152 shared vertices
before), the number still shared after remeshing to 0.12 was:

| Feature angle | Vertices still shared |
|---|---|
| 30° (the default) | 1,350 |
| 90° | 686 |
| 180° (none kept) | 18 |

It needs `pip install pymeshlab`. On Linux the remeshing plugin also needs the
OpenGL library: `apt-get install -y libopengl0`. Without it, the error message
says so.

### 2.2 Breaks worn, rippled, bent

These come in addition to v7's `--remesh_gap` (opening) and `--remesh_jitter`
(independent noise). All of them apply after `--remesh` or `--remesh_edge` and
are refused without one. `--remesh 1 1` remeshes nothing but allows them.

- **`--remesh_smooth N`: wear.** Up to `N` Laplacian smoothing steps, drawn per
  piece, on the break and the ring of vertices beside it. The sharp simulated rim
  rounds off and the finest roughness goes. Each step is capped at a quarter of
  the piece's median edge, so a rim vertex next to a long thin triangle is not
  dragged across the surface. No face ends up turned over.
- **`--remesh_noise S`: scanner ripple.** Displacement along the normal by a
  smooth random field, correlated over about 0.02 (where the jitter is
  independent per vertex). Its standard deviation is drawn per scene from
  `[0, S]`.
- **`--remesh_deform S`: slow bend.** Each piece is bent by its own smooth field
  (length scale 0.3), as two scans registered with their own small errors do
  not fit exactly. Keep it well under stage two's RANSAC tolerance, 0.05 of the
  largest piece's radius.

The order is: remesh, open, wear, bend; then the fracture labels are read; then
ripple and jitter. The dihedral test reads noise as fracture, so the labels come
before it. The distance-based term's positives are taken before anything
moved. Validation is never augmented. With `--evaluate`, `--remesh_edge` (plus
`--remesh_gap` and `--remesh_jitter`) scores remeshed scenes.

### 2.3 One draw per scene (`--remesh_probability`, `--augment_probability`)

One uniform draw per scene and epoch is compared with both shares:

- the scene is remeshed if the draw is below `--remesh_probability`;
- its breaks are perturbed (opened, worn, bent, rippled, jittered) if the draw is
  below `--augment_probability`.

The rest are left as stored.

- **Both at 0.625:** 48 of 128 scenes as stored and 80 remeshed and perturbed,
  as originally planned.
- **`--remesh_probability 1 --augment_probability 0.625`:** every scene is
  remeshed and 62.5% are perturbed. This is the recommended variant. It keeps
  the stored scenes' shared vertices and mirror-image tessellation out of
  training, since no scan has them.

The draw has its own random stream, so with both shares at 1 the remeshing
draws what it always drew.

---

## 3. The fracture head (`segment`)

The dihedral test chooses the cross-attention tokens and the vertices stage two
matches. It over-labels 3.9× on Breaking Bad (precision 0.24, recall 0.93) and
floods on noisy scans. `segment` in `--schedule` adds a head that learns the
break instead:

```
--schedule intra intra intra intra segment global cross intra cross intra cross intra
```

- **What it reads.** The vector features' invariants (`VNInvariant`), then an
  MLP, then one logit per vertex. Whether a vertex is on a break does not change
  with the fragment's pose.
- **What it is trained on.** The true break: on Breaking Bad, the vertices a
  piece shares with another, carried over to remeshed pieces. On converted
  scenes it is the marks (§1.3); scenes without known truth are left out of the
  term.
- **The loss.** Focal binary cross-entropy (`--segment_gamma`, default 2; 0 is
  plain BCE). Each fragment's break and non-break vertices are weighted to count
  equally, since a break is about a tenth of the vertices. Untrained (p = 0.5) it
  reads ln 2 / 4 = 0.173. Weighted by `--w_segment` (1).
- **The tokens.** The network picks them itself, on the GPU, after the head:
  - `--tokens_per_scene` per scene, split over the fragments by how many
    predicted-break vertices each has;
  - spread by farthest-point sampling within each fragment;
  - near-ties broken by vertex order, so the choice does not change with any
    fragment's pose (tested in float64, per fragment);
  - a fragment the head finds nothing on still gets its 8 likeliest vertices.

  The data loader no longer picks tokens for such a model.
- **Teacher forcing.** `--segment_teacher` (0.5) is the share of training
  batches whose tokens come from the true break rather than the prediction. The
  cross layers learn on good tokens while the head is still poor. Validation
  and `--evaluate` always use the prediction.
- **Stage two.** It matches the vertices the head predicts
  (p > `--segment_threshold`, 0.5) instead of the dihedral labels.
  `--label_method marked` (the oracle) replaces both the tokens and the matched
  vertices with the true break.
- **Reported.** Every scored epoch prints, for example, `fracture head: precision
  P recall R (dihedral labels: precision P' recall R')`: the head and the labels
  it replaces, against the same truth. These are exact counts over the epoch.
  `--evaluate` prints them for converted scenes whose marks were written (§1.3).

A schedule with `segment` must put every `cross` layer after it, and may contain
it once.

---

## 4. The consistency term (`--w_consistency`)

Each training scene that was remeshed or perturbed also brings itself as stored
(`SceneSample.view`). It is tumbled the same way, and every vertex knows the
stored vertex it came from: the nearest one, to where it was before anything
moved it. The network runs on the stored scene without a gradient. The term then
pulls each break vertex's embedding toward its stored partner's (1 − cos):

```
--w_consistency 0.5
```

It is not a loss between a scene and the same scene rotated: the embedding is
invariant to every fragment's pose by construction
(`test_the_embedding_does_not_depend_on_the_perturbation`, drift < 1e-10), so
such a loss is exactly zero for any weights. Remeshing and perturbing are what
change the embedding, so that is what this term teaches the network to ignore.

It costs a second forward pass (no backward) and a second copy of each changed
scene in the loader. It needs `--remesh` or `--remesh_edge`. A constant
embedding would also score 0; the contrastive term beside it prevents that.

## 5. The fragment context (`global`)

`global` in `--schedule` pools each fragment's vector features (an equivariant
mean) and hands the pool back to every vertex through a VN layer:

```
x + W_o · act(W_x x + W_g · mean_fragment(x))
```

followed by a VN layer norm. It turns with its own fragment, ignores every
other, and starts near the identity. It gives each vertex its place in the whole
piece, beyond the few rings the intra layers reach.

## Carrying a v7 run into the new schedule (`--init_partial`)

`--init_from v7/best.pt --init_partial True` loads the weights whose names and
shapes match and starts the rest fresh, naming them. Intra and cross layers are
matched by their order among their own kind. So in the schedule above, the first
five intra layers take v7's five, wherever those sat. It is a warm start, not
the same network; training from scratch is the clean comparison.

---

## 6. Running it

### 6.0 Setup (Kaggle or any Linux machine)

```bash
pip install fast-simplification pymeshlab h5py
apt-get install -y libopengl0          # pymeshlab's remeshing plugin needs it
python -m scripts.check_version        # every marker present, the copy is complete
python -c "import sys; sys.path.insert(0, 'src'); from reassembly.data.remesh import _pymeshlab; _pymeshlab(); print('pymeshlab ok')"
```

Below, `data` is your Breaking Bad root (`--root_dir`), as in the README. Add
your usual flags to every training command: `--hidden_channels 128`,
`--num_gpus`, `--batch_size`, `--epochs`, `--time_budget_hours`, and so on.
**An evaluation takes the architecture from the checkpoint, so its flags do not
need repeating.**

### 6.1 Convert the scans again, so they carry their true breaks

Run it twice: once thinned as before (for v7 and the oracle), and once
isotropic (for the new model).

```bash
python -m scripts.convert_external Fractura/bone_real.hdf5 Fractura/fractura_real.hdf5 \
    path/to/FB_v1 --out external --reference data
python -m scripts.convert_external Fractura/bone_real.hdf5 Fractura/fractura_real.hdf5 \
    path/to/FB_v1 --out external_iso --reference data --isotropic
```

Use your own file names. Each source becomes a subset named after the file:
`bone_real`, `fractura_real`, `FB_v1`.

### 6.2 The oracle on the v7 checkpoint (no training)

```bash
for S in bone_real fractura_real FB_v1; do
  base="--root_dir external --evaluate --checkpoint checkpoint-final/best.pt --data_subsets $S --split all --modes_per_scene 0 --max_fragments 0"
  python -m scripts.train $base --checkpoint_dir eval/v7-$S
  python -m scripts.train $base --label_method marked --checkpoint_dir eval/v7-$S-oracle
  python -m scripts.train $base --match_metric cosine --checkpoint_dir eval/v7-$S-cosine
done
```

Compare `match@1`, `acc@5` and part accuracy between `eval/v7-$S` and
`eval/v7-$S-oracle`. A large rise means the fracture labels were what failed on
scans, and the fracture head is the fix. No rise means the descriptors are the
problem. Run the same three on `checkpoint-open/best.pt`, the contact-radius
fine-tune.

### 6.3 Train the sim-to-real model

```bash
S2R="--root_dir data --checkpoint_dir checkpoint-s2r \
  --schedule intra intra intra intra segment global cross intra cross intra cross intra \
  --contact_radius 0.015 \
  --remesh_edge 0.012 0.018 --remesh_gap 0.03 --remesh_smooth 3 --remesh_deform 0.003 \
  --remesh_noise 0.001 --remesh_jitter 0.0005 --augment_probability 0.625 \
  --w_consistency 0.5 --num_workers 4"
python -m scripts.train $S2R --preflight      # memory, losses at init, time per step
python -m scripts.train $S2R                  # plus your usual flags
```

Preflight times the steps on your machine. Isotropic remeshing, the stored view
and the consistency pass all add time, and none of it was measured on Breaking
Bad in this environment, which has no dataset. The flags carry over to every
session: rerun the same command and it resumes from `last.pt`. On Kaggle, point
`--resume` at the previous session's output.

To warm-start from v7 instead, add `--init_from checkpoint-final/best.pt
--init_partial True`.

### 6.4 Test it

```bash
# Breaking Bad validation, as stored and as scan-like
python -m scripts.train --root_dir data --evaluate --checkpoint checkpoint-s2r/best.pt --split val --checkpoint_dir eval/s2r-val
python -m scripts.train --root_dir data --evaluate --checkpoint checkpoint-s2r/best.pt --split val \
    --remesh_edge 0.012 0.018 --remesh_gap 0.03 --remesh_jitter 0.0005 --checkpoint_dir eval/s2r-val-scanlike
# the scans, converted isotropically; then the oracle on the same scenes
for S in bone_real fractura_real FB_v1; do
  base="--root_dir external_iso --evaluate --checkpoint checkpoint-s2r/best.pt --data_subsets $S --split all --modes_per_scene 0 --max_fragments 0"
  python -m scripts.train $base --checkpoint_dir eval/s2r-$S
  python -m scripts.train $base --label_method marked --checkpoint_dir eval/s2r-$S-oracle
done
```

What to read in each report:

- **`match@1`, `acc@5` and part accuracy.** Compare against the v7 numbers from
  §6.2.
- **`fracture head: precision … recall …`, against the dihedral labels.** This
  shows whether the head beats the labels on the scans' true breaks.
- **The oracle gap.** If the oracle still beats the head, part of what remains
  is segmentation.

### 6.5 Afterwards: Fantastic Breaks, and ablations

The README's Fantastic Breaks fine-tune runs unchanged on `external_iso`. Use
`--init_from checkpoint-s2r/best.pt`, the same `--schedule` and
`--remesh_edge 0.012 0.018`, and drop `--remesh`. The converted marks then train
the fracture head on real breaks.

Ablations, one change at a time against `checkpoint-s2r`:

- no `--w_consistency`;
- no `segment` (dihedral labels);
- no `global`;
- `--remesh 0.5 0.9` instead of `--remesh_edge`;
- `--augment_probability 1`;
- both shares at 0.625 (the original 48/80 plan).

## 7. Tests

`tests/test_sim2real.py` (24 tests) checks:

- **Stage two:** cosine against Euclidean matching on scrambled lengths, and
  candidate spread and pose independence.
- **The true break:** written and read back through the loader's reordering;
  completed on unmarked pieces; returned by the converter; and labels under
  `marked`, stored and remeshed.
- **The augmentations:** wear, ripple and bend with no face turned over and only
  the break band moving; and the label and positive order.
- **Isotropic remeshing:** closed, edges near the target, almost no shared
  vertex left, long thin triangles evened out.
- **The per-scene draw and refused settings.**
- **The network:** per-fragment equivariance with `segment` and `global`,
  float64 and ties included; token budget and mask; the fracture term's and the
  consistency term's reference values; stage two matching the predicted break;
  teacher forcing and the oracle.
- **Whole runs:** the stored view through collate and the forward pass; a whole
  run with every change trained, validated and evaluated (with and without the
  oracle); `--init_partial`; an old checkpoint loading as before.

The whole suite passes, slow tests included: CPU, torch 2.14, pymeshlab
2025.7.
