# Sim to real: the changes after v7, and how to train and test them

V7, trained on Breaking Bad (Everyday), reassembles unseen Breaking Bad scenes
and Fractura's synthetic bones well, and reaches chance on scanned pieces
(Fractura's real bones, Fantastic Breaks). This document covers the changes made
to attack that gap, in the order they were planned, and then the runs that test
them.

Every change is **off by default**. With no new flag, training, evaluation and
`probe_val.py` do exactly what v7 did. Two exceptions:

- stage two now spreads its matching candidates over the break (§1.2), and that
  only applies to fragments with more than 2,048 candidates;
- the embedding term is computed scene by scene (§2.4). That only changes a
  forward pass of more than one scene (`--micro_batch_scenes` above 1), which
  v7's runs did not use.

An old checkpoint loads and evaluates as before.

| | What | Flag(s) | Where |
|---|---|---|---|
| 1 | Matching metric for stage two | `--match_metric euclidean\|cosine` | `assembly/translation.py` |
| 1 | Candidates spread over the break, not strided by index | (always) | `assembly/translation.py` |
| 1 | True breaks written beside converted scenes; the oracle | `--label_method marked` | `data/scene.py`, `data/external.py`, `training.py` |
| 2 | Isotropic remeshing, in training and conversion | `--remesh_edge LOW HIGH`, `convert_external --isotropic` | `data/remesh.py`, `data/external.py` |
| 2 | Breaks worn, rippled, bent (plus v7's opening and jitter) | `--remesh_smooth`, `--remesh_noise`, `--remesh_deform` | `data/remesh.py` |
| 2 | Exact shares per optimizer step: stored, remeshed, or perturbed | `--remesh_probability`, `--augment_probability` | `data/sampling.py`, `training.py` |
| 2 | The embedding term scene by scene: each scene its own anchors and negatives | (always) | `nn/losses.py`, `training.py` |
| 2 | Validation remeshed, so `best.pt` rewards matching by shape | `--val_remesh` | `training.py` |
| 3 | Fracture head: learns the break, picks the tokens and the matched vertices | `segment` in `--schedule`, `--w_segment`, `--segment_gamma`, `--segment_teacher`, `--segment_threshold` | `nn/context.py`, `nn/model.py`, `training.py` |
| 4 | Consistency between a scene and itself as stored | `--w_consistency` | `nn/losses.py`, `training.py` |
| 5 | Fragment context for every vertex | `global` in `--schedule` | `nn/context.py` |
| – | Carry a v7 run's weights into the new schedule | `--init_partial` with `--init_from` | `training.py` |
| – | The converter: one-assembly `.npz` files read; a break no dataset marks, estimated; dense scans thinned keeping their topology | (always) | `data/external.py`, `scripts/convert_external.py` |

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
- Fantastic Breaks: `meta_<k>.npz`'s `mask` in its folder layout. In the
  one-`.npz`-per-object layout of the sample (§7), each piece's
  `piece_<i>_shared_faces`.

The marks are carried through the cleaning and thinning. The folder layout marks
the broken object only, so the restoration takes the vertices within 0.03 of the
marked break (`BREAK_REACH`); real gaps are 0.011–0.029.

**Fractura's real bones mark nothing.** In `pig_bone_1` the `shared_faces`
mark 24 faces per piece: the same faces in all four pieces, around the origin,
which are the debris the cleaning drops. In `limb1`, `limb4`, `vert1` and
`vert6` they are −1 throughout. Their `full_mesh` is only the pieces put
together, so it does not help either.

So an assembly that marks no break gets one estimated from its assembled pose
(`estimated_breaks`). A face counts as break when the nearest face of another
piece, centroid to centroid, is within 0.03 and faces it: the two face normals'
dot product is below −0.5. Two choices in that rule matter:

- **Face normals, not vertex normals.** A vertex on a break's rim averages the
  break and the intact surface. Judged by vertex normals, the rims and every
  narrow break went unmarked: recall 0.11–0.62 on five Breaking Bad scenes and
  0.21 on Fantastic Breaks.
- **Facing, not proximity alone.** The shards of `pig_bone_1` are thin and lie
  against each other, and 44–91% of each one's vertices are within 0.03 of
  another piece.

The estimate was checked where the break is known:

| Sample | Precision | Recall |
|---|---|---|
| Breaking Bad, five scenes (the coincident vertices) | 0.91–0.98 | 1.00 |
| Fantastic Breaks `00002` (its marks) | 0.94 | 0.67 |

Breaking Bad is the easy case: the two sides of a simulated break are mirror
images. Fantastic Breaks is a scan, and the better test.

On the five real bones, drawn piece by piece, the estimate covers the limb
bones' broken ends, the vertebrae's pedicles (where the arch broke from the
body), and the shards' split edges. It is conservative: what it calls break is
break, which is what an oracle needs. But it misses the parts of a break whose
two sides stand more than 0.03 apart. At a reach of 0.05, the estimates of the
vertebrae and of `limb4` grow by a quarter to two thirds (limb1's not at all).
**On the real bones, the oracle and the fracture head's precision and recall are
measured against this estimate**, so the head's precision there is a lower
bound.

The CSV report gains `marked_pieces`, `marked_vertices` and `break_source`
(`marked` or `estimated`), and the summary counts the estimated assemblies.
**Scenes converted before this change have no such file and must be converted
again.**

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

Training remeshes without pymeshlab's check of the distance to the input
surface. On the Breaking Bad sample (15 pieces, 34k faces) that took 0.50 s
against 1.38 s, at the same edges (median 1.03 of the target, 0.1% slivers). All
it gives up is the sharpest corners, rounded by up to 0.8 of an edge at the 99th
percentile instead of 0.4, as a scan rounds them. The converter remeshes once
per scan, so it keeps the check.

A scan more than 3× denser than the target is first thinned to half the target
edge, by pymeshlab's quadric collapse **with its topology kept**. The faster
collapse v7 thins with (fast-simplification) does not keep it. On `pig_bone_1`
it left 12–173 edges shared by more than two faces in each of the four pieces,
and the remesher carried them on and opened them into holes (360 boundary
edges). With the topology kept there are none: the bone converts closed and
manifold, with 1% slivers, in 72 s for its 1.65M faces.

A remesh that comes out worse than its input is refused:

- more than 20% slivers;
- a median edge outside 0.6–1.6× the target;
- more than 0.2% more non-manifold edges.

The piece is then collapsed instead to the faces an even mesh of that edge would
have. Its tessellation stays its own, and the converter also keeps its topology.
Both training and the converter do this, and the converter counts the piece
(`isotropic_failed` in the CSV). Kept as it was, a full-resolution scan would
stay many times denser than the training scenes.

This catches Fantastic Breaks' restorations. In the sample (§7) the restoration
is a boolean cut's mesh, with 91% slivers and 341 edges shared by four or six
faces, which no setting of the remesher could even out. Collapsed, it keeps 88%
slivers and has 429 such edges. Every collapse tried did about as well: 324–588
such edges, 79–92% slivers.

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

### 2.3 Exact shares per step (`--remesh_probability`, `--augment_probability`)

Each training scene gets one number in [0, 1) per epoch, compared with both
shares:

- the scene is remeshed if the number is below `--remesh_probability`;
- its breaks are perturbed (opened, worn, bent, rippled, jittered) if the number
  is below `--augment_probability`.

The rest are left as stored.

The sampler deals those numbers (`DealtShares`, `data/sampling.py`). It knows
where each optimizer step starts: a step is `batch_size × accumulate`
consecutive scenes on one GPU. Within each step it spreads the numbers evenly,
`(k + u) / step` for a shuffled `k`. So every step holds exactly the planned
count, not just on average. One independent draw per scene would give 80 of 128
give or take 5.5. Every GPU deals its own steps. An interrupted epoch restarts
from its first step and is dealt the same again.

- **Both at 0.625:** 3 of every 8 scenes as stored and 5 remeshed and perturbed
  in every step: 48 and 80 of 128, as originally planned. V8's first runs used
  it.
- **`--remesh_probability 1 --augment_probability 0.625`:** every scene is
  remeshed and 62.5% are perturbed: 48 unperturbed and 80 perturbed of 128.
  This keeps the stored scenes' shared vertices and mirror-image tessellation
  out of training, since no scan has them. It is V8's second stage now, in
  steps of 8, after a first stage with `--augment_probability 0`: the stored
  scenes taught the network to match by mesh, and perturbed scenes from the
  first step taught it almost nothing (§2.5).

The banner prints the counts per step. A scene rebuilt outside training, by a
script, draws its own number from its own stream, as before. With both shares at
1 nothing is dealt, and the remeshing draws what it always drew.

### 2.4 The embedding term, scene by scene

The contrastive term subsamples its anchors, 1,024 at a time, because its
similarity matrix is quadratic in them. It used to draw them from the whole
forward pass, and every other anchor of the pass was a negative. v7's runs had
one scene a pass, so nothing changed for them. V8's first runs put four in each
(`--micro_batch_scenes 4`), and that cut the term down:

- An anchor trains only if one of its positives is drawn with it. Four scenes
  sharing one budget each got about a quarter of the anchors, and far fewer
  than a quarter with a positive drawn too.
- Three quarters of each anchor's negatives were other objects' vertices.
  Matching never compares two scenes, so those negatives taught nothing.

On the four Breaking Bad samples with 2–20 pieces, the anchors with a drawn
positive:

| | Bottle | Bowl | Cup | Mug | All |
|---|---|---|---|---|---|
| One scene a pass, as stored | 823 | 1,018 | 1,001 | 335 | 3,176 |
| Four in one pass, as stored | 472 | 38 | 118 | 15 | 643 |
| One scene a pass, remeshed | 908 | 743 | 969 | 351 | 2,971 |
| Four in one pass, remeshed | 97 | 223 | 87 | 20 | 427 |

Both terms (`proximity_loss`, `correspondence_loss`) are now computed scene by
scene:

- each scene draws its own 1,024 anchors;
- an anchor's negatives are its own scene's vertices;
- each scene's mean is weighted by its scored fragments (its fragments less the
  anchor), the weight a pass of that scene alone carries in the step.

So a pass of several scenes gives the step one pass of each gives, to float32
round-off and the anchors' draw. `--micro_batch_scenes` changes the memory, not
what is trained. A pass of one scene is unchanged, its draw included.

`tests/test_training.py::test_one_pass_of_two_scenes_steps_as_two_passes_of_one`
compares the two steps (relative difference 5e-7; until this change the test
asserted that they differ). `tests/test_contact.py` checks that another scene's
vertices are neither positives nor negatives, that each scene keeps its own
budget, and the weighting.

### 2.5 Validation remeshed (`--val_remesh`)

Stored Breaking Bad scenes share their break vertices, so the two sides of a
break carry the same triangles. A network can match them by their mesh alone,
with no notion of shape. Measured on the samples (V8's schedule, the four
2–20-piece scenes, two a step), trained on the scenes as stored only:

| step | as stored: embedding term, match@1 | the same scenes remeshed: embedding term, match@1 |
|---|---|---|
| 0 | 5.54, 0.006 | 6.14, 0.002 |
| 75 | 4.84, 0.017 | 6.70, 0.003 |
| 150 | 4.32, 0.043 | 7.99, 0.001 |

The more it learned the stored scenes, the worse it did on the same scenes
meshed on their own: the term rose well above its untrained value. That is
v7's record too: match@1 0.53–0.68 on Breaking Bad and `bone_synthetic`,
0.012–0.017 on the scans (`PROJECT-STATE.md` §21).

Remeshed and nothing else, the scenes did teach it to match by shape: on the
remeshed copies, match@1 rose 0.004 → 0.034–0.040 in 400 steps. Remeshed and
perturbed from the first step, they barely moved it (0.005 at step 200). That
is why V8 trains in two stages, shape first (`docs/V8-RUN.md` §2).

Validation as stored rewards the mesh, and `best.pt` is chosen on it.
`--val_remesh` validates on every validation scene remeshed by the training
settings instead, `--augment_probability` of them perturbed too. Each scene
gets one draw, its own, and every epoch repeats it, so the epochs are compared
on the same scenes. An evaluation
never takes it from a checkpoint: it is remeshed only when asked
(`--evaluate --remesh_edge ...`, §6.4). Changing it on a resume resets the
best-so-far, as a change of `--max_fragments` does.

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

**V8's run** — convert, train from scratch with 48 stored and 80 perturbed scenes
in every step of 128, test on Breaking Bad's unseen objects, draw the figures,
test on the converted scans — is in `docs/V8-RUN.md`, command by command. This
section is the general plan it follows.

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

A source can be a GARF `.hdf5` file, a Fantastic Breaks folder
(`model_b_<k>.ply`, `model_r_<k>.ply`, `meta_<k>.npz`), or assemblies stored
one per `.npz` file: `num_pieces` and `piece_<i>_vertices`, `piece_<i>_faces`,
`piece_<i>_shared_faces`, as in the Fantastic Breaks sample. Pass one such file,
or a folder of them. An assembly that marks no break, as the real bones do,
gets its break estimated (§1.3), and the summary says so: `marks no break: N of
M -- their break is ESTIMATED`.

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
  --remesh_noise 0.001 --remesh_jitter 0.0005 --augment_probability 0.625 --val_remesh True \
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

## 7. The samples, measured

The samples:

- **Breaking Bad**: five scenes, of 2 to 64 pieces.
- **Fantastic Breaks**: one pair, `00002`. Its two pieces come in one `.npz` in
  GARF's per-piece layout, decimated to 5% of the scan; the full-resolution
  pairs were too large to share.
- **Fractura real**: five bones.
  - `pig_bone_1`, from `bone_real_small.hdf5`: four thin shards.
  - `limb1`, `limb4`, `vert1` and `vert6`, from `fractura_real.hdf5`: two pieces
    each. That file's 27 objects (16 bones, 8 ceramics, 3 eggs) are all in its
    `val` split.

That is a handful per dataset, so the numbers are ranges over these samples, not
dataset statistics. Lengths are fractions of the object's largest side.

### 7.1 As stored

| | Breaking Bad (5) | Fantastic Breaks `00002` | Fractura real (5) |
|---|---|---|---|
| pieces | 2–64 | 2 | 2–4 |
| median edge | 0.0079–0.0232 | 0.0085 | 0.0014–0.0058 |
| edge spread (p90/p10) | 10.8–15.7 | 4.2 | 3.0–3.1 |
| triangle shape: median / slivers | 3.6–4.9 / 36–50% | 3.1 / 30% | 2.0 / 6–7% |
| closed and manifold | all | broken piece: closed, 8 non-manifold edges; restoration: 341 non-manifold edges, 91% slivers | all, once cleaned |
| break marked | coincident vertices, 7–44% | `shared_faces`: 3.8% and 29.3% | none; estimated at 1–34% per piece (§1.3) |
| dihedral labels vs that break: precision / recall | 0.10–0.68 / 1.00 | 0.28 / 0.70 | 0.03–0.27 / 0.10–0.49 |
| vertices shared across the break | 272–6,539 | 0 | 0 |
| gap across the break (median) | 0 | 0.0003 (restoration to broken), 0.0065 (broken to the coarser restoration) | 0.0049–0.0185 |
| wall thickness (median per piece) | 0.012–0.066 | 0.028 and 0.044 | 0.012–0.016 (`pig_bone_1`), 0.03–0.10 (the others) |

Wall thickness is the distance from a surface point to the nearest point of the
same piece that faces the other way. `pig_bone_1` is stored as one vertex per
face corner, plus Blender's cube; the four from `fractura_real.hdf5` are stored
as closed meshes.

- **The tessellations differ most.** Breaking Bad's edges spread 11–16×, and four
  in ten of its triangles are slivers. The scans are meshed evenly.
- **The real pieces stand apart.** Across the estimated break, Fractura's are a
  median 0.005–0.019 apart, and only 1–3% of the four solid bones' vertices are
  within 0.005 of the other piece. The Fantastic Breaks pair is flush. Training's
  opening (`--remesh_gap 0.03`, calibrated on 27 Fractura scenes) covers that
  range.
- **Walls vary.** `pig_bone_1`'s shards are as thin as the thinnest Breaking Bad
  scene's pieces (64 pieces, 0.012). The other bones are solid.
- **At full resolution the dihedral test is of no use on the scans:** precision
  0.03–0.27 against the estimate.

### 7.2 Roughness, measured so geometry stays out

The first measure of roughness here was the median distance of surface samples
within 0.02 from their fitted plane. It mixed geometry in: on a thin wall the
ball takes in the far side of the wall, and on a narrow break the facets turned
another way. It read the bone's intact surface as 0.0054, rougher than its
break, and Fantastic Breaks' break as 2–3.5× rougher than Breaking Bad's.

The measure below fixes that in two ways:

- only neighbours facing within 60° of the centre count;
- a quadric is fitted instead of a plane, so curvature drops out.

The table is at the resolution the model reads, every sample remeshed
isotropically to about 0.0145. The scans' break is the one they mark, or the
estimate.

| Quadric residual within 0.02, edge ~0.0145 | break | intact |
|---|---|---|
| Fractura real, five bones, `--isotropic` | 0.00040–0.00091 | 0.00021–0.00037 |
| Fantastic Breaks `00002`, `--isotropic` | 0.00118 | 0.00006 |
| Breaking Bad, five scenes, `--remesh_edge 0.012 0.018` alone | 0.00038–0.00057 | 0.00005–0.00053 |
| Breaking Bad, five scenes, §6.3's settings | 0.00031–0.00058 | 0.00022–0.00057 |

- **The intact surfaces are covered.** The bones' fall within §6.3's range, and
  Fantastic Breaks' is smoother than any training scene's. At full resolution
  `pig_bone_1`'s intact surface is four times as textured as Breaking Bad's
  (0.00051 against 0.00013). But that texture is finer than the edge, and
  remeshing removes it (0.00021). So no roughening of intact surfaces is needed.
- **The scans' breaks are as rough as the training breaks, or rougher.** Three
  of the bones match them (0.00040–0.00044). `pig_bone_1` (0.00081), `limb4`
  (0.00091) and Fantastic Breaks (0.00118) are rougher than any training scene
  (0.00031–0.00058). No setting closes that: on the first scene, without the
  wear (`--remesh_smooth 0`) the training breaks measure 0.00043–0.00061, and
  without the opening 0.00037–0.00041. A fracture detector forgives this
  direction, and §6.4's fracture-head report measures it on the scans directly.
- **Roughness alone does not tell break from intact, even in training.** Two of
  the five Breaking Bad scenes have intact surfaces as textured as their breaks
  (0.00042–0.00057).

### 7.3 What the training settings make of the first Breaking Bad scene

| | edge spread | slivers | shared | roughness, break / intact | gap, contact region (median) | dihedral P / R |
|---|---|---|---|---|---|---|
| as stored | 14.8 | 39% | 3,478 | 0.00057 / 0.00013 | 0 | 0.62 / 1.00 |
| v7's recipe: collapse 0.5–0.9, jitter 0.001, gap 0.03 | 8.4 | 33% | 0 | 0.00085–0.00100 / 0.00050–0.00051 | 0.0045–0.0064 | 0.21 / 1.00 |
| `--remesh_edge 0.012 0.018` alone | 1.7 | 0% | 8 | 0.00039–0.00042 / 0.00009 | 0.0008–0.0010 | 0.46 / 0.97 |
| §6.3's settings | 1.8–1.9 | 0–1% | 0 | 0.00037–0.00055 / 0.00024–0.00029 | 0.0085–0.0138 | 0.43 / 0.95–0.97 |

Roughness is §7.2's measure, at each setting's own resolution.

The recommended settings do three things: they even out the tessellation,
remove the shared vertices, and open the contact as the scans' is open. V7's
independent jitter roughens intact surfaces past every scan's (0.00050 against
at most 0.00037).

### 7.4 The scans as the model gets them

Converted with `--isotropic` at 0.0145:

- **Fractura's bones** come out closed and manifold, in 4–72 s each. `limb4` has
  1,632 vertices, `vert6` 10,713 and `pig_bone_1` 7,062.
  - **A break is one or two edges across.** 43–71% of the break vertices are one
    edge from the intact side; the solid bones' breaks are wider than the
    shards'. In Breaking Bad's training scenes the figure is 81–83% (the first
    scene, at `--remesh_edge 0.012 0.018`). So no finer edge is called for. At
    0.007, `pig_bone_1` and Breaking Bad widen alike (63% against 48–50%), and
    the vertex count quadruples.
  - **The dihedral test floods.** It flags 30–86% of each piece's vertices, with
    precision 0.07–0.41 against the estimate and recall 0.53–0.99. V7's thinned
    conversion of `pig_bone_1` floods as much (47–91% of each piece's vertices).
    This is what the fracture head (§3) and the oracle (§1.3) are for.
- **Fantastic Breaks** comes out as 20,249 vertices. Its break is wider: 18% of
  its break vertices are one edge from the intact side. The remesher refuses the
  restoration, so it is collapsed to the density (§2.1), and the dihedral test
  flags 90% of it (14% of the broken piece).

## 8. Tests

`tests/test_sim2real.py` (28 tests) checks:

- **Stage two:** cosine against Euclidean matching on scrambled lengths, and
  candidate spread and pose independence.
- **The true break:** written and read back through the loader's reordering;
  completed on unmarked pieces; returned by the converter; estimated where a
  dataset marks none; and labels under `marked`, stored and remeshed.
- **The augmentations:** wear, ripple and bend with no face turned over and only
  the break band moving; and the label and positive order.
- **Isotropic remeshing:** closed, edges near the target, almost no shared
  vertex left, long thin triangles evened out; a remesh worse than its input
  refused, then collapsed, in training and by the converter, which counts it;
  a dense scan thinned with its topology kept before it is remeshed.
- **The shares and refused settings:** one draw per scene where nothing is
  dealt; through the training loader, every step dealt its exact split of
  stored and perturbed scenes, honoured by the scene (`tests/test_sampling.py`
  checks the dealing itself).
- **The network:** per-fragment equivariance with `segment` and `global`,
  float64 and ties included; token budget and mask; the fracture term's and the
  consistency term's reference values; stage two matching the predicted break;
  teacher forcing and the oracle.
- **Whole runs:** the stored view through collate and the forward pass; a whole
  run with every change trained, validated and evaluated (with and without the
  oracle); `--init_partial`; an old checkpoint loading as before.

`tests/test_external.py` adds the one-`.npz`-per-object layout: read, other
`.npz` files passed over, and converted by the script.

The whole suite passes, slow tests included: CPU, torch 2.14, pymeshlab
2025.7.
