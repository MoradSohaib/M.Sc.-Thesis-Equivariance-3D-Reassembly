# V8: convert, train, test — the commands

This is the V8 run in order:

1. Convert Fractura and Fantastic Breaks.
2. Train from scratch on Breaking Bad's Everyday subset.
3. Test on every Breaking Bad object the training did not see.
4. Draw the per-category figures.
5. Test on the converted scans.

The whole sequence was run end to end on the samples of each dataset, on CPU
and scaled down, with the flags below. `docs/SIM2REAL.md` explains what each
flag does and why.

## 0. Setup

```bash
git clone -b V8 https://github.com/MoradSohaib/M.Sc.-Thesis-Equivariance-3D-Reassembly.git
cd M.Sc.-Thesis-Equivariance-3D-Reassembly
pip install -r requirements.txt
apt-get install -y libopengl0          # pymeshlab's remeshing plugin needs it (Linux)
python -m scripts.check_version        # every marker present: the copy is complete
```

Set the four paths once per session:

```bash
BB=/path/to/breaking_bad        # the unzipped Breaking Bad: holds everyday_compressed/,
                                # artifact_compressed/, other_compressed/ (and data_split/)
FB=/path/to/fantastic_breaks    # the unzipped Fantastic Breaks
FRACTURA=/path/to/fractura      # holds fractura_real.hdf5, bone_real.hdf5, bone_synthetic.hdf5
OUT=/path/to/output             # writable: converted data, checkpoints, results, figures
mkdir -p $OUT/converted
```

On Kaggle, the inputs are under `/kaggle/input` (read-only) and `OUT` must be
under `/kaggle/working`. Keep Breaking Bad's `data_split/` folder inside `$BB`.
Without it the split falls back to a hashed one, and the numbers stop being
comparable with published ones (preflight warns).

## 1. Convert Fractura and Fantastic Breaks (before training)

This step needs no GPU. Each dataset goes into its own folder, and each source
becomes one subset named after its file: `fractura_real`, `bone_real`,
`bone_synthetic`, and Fantastic Breaks under the name given with `--subsets`.

```bash
python -m scripts.convert_external \
    $FRACTURA/fractura_real.hdf5 $FRACTURA/bone_real.hdf5 $FRACTURA/bone_synthetic.hdf5 \
    --out $OUT/converted/fractura --edge 0.015 --isotropic --workers 2 \
    2>&1 | tee $OUT/converted/fractura.log

python -m scripts.convert_external $FB --subsets fantastic_breaks \
    --out $OUT/converted/fantastic_breaks --edge 0.015 --isotropic --workers 2 \
    2>&1 | tee $OUT/converted/fantastic_breaks.log
```

What it does to each assembly:

1. Cleans every piece to one closed surface.
2. Scales the object so its largest side is 1, as Breaking Bad's objects are.
3. Remeshes every piece isotropically to an edge of 0.015, the middle of
   training's `--remesh_edge 0.012 0.018`.
   - A full-resolution scan is first thinned with its topology kept.
   - A piece the remesher cannot work on is collapsed to that density instead,
     and counted.

`$FB` can be a Fantastic Breaks folder (`model_b_<k>.ply`, `model_r_<k>.ply`,
`meta_<k>.npz`) or a folder of one-`.npz`-per-object files. Point it at the
folder that holds the object folders, or the `.npz` files.

**The precision and recall it reports.** For each subset, the summary prints:

- **The break each assembly is scored against.** Either the dataset's marks
  (Fantastic Breaks, and Fractura's synthetic set if its `shared_faces` are
  filled; no synthetic sample was checked here) or, where the dataset marks none
  (Fractura's real objects), the break estimated from the assembled pose
  (`marks no break: N of M -- their break is ESTIMATED`, `docs/SIM2REAL.md`
  §1.3). The summary says which, per subset.
- **The dihedral labels against that break.** These are the labels the model's
  stage two falls back on without the fracture head. Precision and recall are
  given at full resolution and after conversion, as medians over the
  assemblies.
- **The estimate against the marks.** Wherever a dataset does mark its break,
  the estimate's own precision and recall, so you can see how far to trust it on
  the sets that do not mark one.

`$OUT/converted/<dataset>/<subset>.csv` has every number per assembly:

| Columns | What |
|---|---|
| `break_source` | `marked` or `estimated` |
| `precision_full`, `recall_full` | the dihedral labels against the break, full resolution |
| `precision`, `recall` | the same after conversion |
| `estimate_precision`, `estimate_recall` | the estimate against the marks, where marked |

It also has the face counts, edge lengths, and edges left open or shared by more
than two faces.

Expect hours: a 1.65M-face object takes about 70 s. Each worker holds one whole
assembly, and the largest Fractura objects have 3–6M faces, so raise
`--workers` only with the memory for it.

## 2. Train V8 from scratch

The architecture is four intra layers, then the fracture head (`segment`) and
the fragment context (`global`), then three cross–intra pairs.

**Every training scene is remeshed.** None is used as stored: stored scenes
teach the network to match by mesh rather than by shape (below, and
`docs/SIM2REAL.md` §2.5). The run has two stages:

1. **Shape, 12 epochs.** Remeshed, nothing perturbed. On the samples this was
   the one setting that learned to match pieces meshed on their own.
2. **The planned mix, 28 epochs, from stage 1's weights.** 48 of every 128
   scenes unperturbed and 80 perturbed, 3 and 5 of every 8 in each optimizer
   step. The split is exact in every step, not just on average: the sampler
   deals each scene its share (`docs/SIM2REAL.md` §2.3).

"Perturbed" means every setting of §6.3:

- opened (`--remesh_gap`);
- worn smooth (`--remesh_smooth`);
- jittered (`--remesh_jitter`);
- rippled (`--remesh_noise`);
- bent (`--remesh_deform`).

From the first step, those scenes barely moved the network on the samples.
Hence the first stage.

```bash
COMMON="--root_dir $BB --data_subsets everyday_compressed --max_fragments 20 \
  --schedule intra intra intra intra segment global cross intra cross intra cross intra \
  --hidden_channels 128 \
  --contact_radius 0.015 \
  --remesh_edge 0.012 0.018 --remesh_gap 0.03 --remesh_smooth 3 --remesh_jitter 0.0005 \
  --remesh_noise 0.001 --remesh_deform 0.003 --remesh_probability 1 --val_remesh True \
  --num_gpus 1 --batch_size 8 --micro_batch_scenes 4 --steps_per_epoch 400 \
  --val_steps 160 --num_workers 4 --time_budget_hours 11"

# Stage 1: shape
STAGE1="$COMMON --checkpoint_dir $OUT/checkpoint-v8-shape --augment_probability 0 --epochs 12"
python -m scripts.train $STAGE1 --preflight   # memory, losses at init, time per step, sessions
python -m scripts.train $STAGE1

# Stage 2: the mix, from stage 1's last epoch
STAGE2="$COMMON --checkpoint_dir $OUT/checkpoint-v8c --augment_probability 0.625 \
  --w_consistency 0.5 --epochs 28 --init_from $OUT/checkpoint-v8-shape/last.pt"
python -m scripts.train $STAGE2
```

- **The banners state the split.** Stage 1 prints `8 remeshed, 0 of them
  perturbed, none as stored` and `validation remeshed too, every scene, 0.0% of
  them perturbed`. Stage 2 prints `8 remeshed, 5 of them perturbed, none as
  stored (dealt by the sampler)` and `62.5% of them perturbed`. On two GPUs,
  keep `--batch_size 8` (per GPU) and halve `--steps_per_epoch` to 200.
- **What to watch in stage 1.** The training `embedding` term should fall from
  its first value and validation `match@1` should rise above ~0.005 within the
  first epochs. If neither moves by epoch 5, stop and read "If the curves stay
  flat" below before going on.
- **Stage 2 starts fresh** from the weights only: new optimizer, new schedule
  and warmup, new history (`--init_from`). It takes `last.pt`, because
  `best.pt` stays at its first epoch for as long as validation acc@5 reads 0.
  It is untested; on the samples the first stage alone learned fastest. If
  stage 2 loses what stage 1 learned (validation `match@1` falling for several
  epochs), stage 1's checkpoint is the better model for the scans.
- **Small steps, many of them.** An epoch is 400 steps of 8 scenes, 3,200 scenes,
  about one pass over Everyday's training split. So the 40 epochs are 16,000
  optimizer steps; v7's defaults gave 32,000 steps of 4 scenes. Each scene
  brings thousands of vertex pairs to the embedding term, so a step of 8 is
  already a low-noise gradient. The learning rate and weight decay stay v7's
  (`--lr 1e-3`, `--weight_decay 1e-4`, warmup on). `--val_steps 160` is 1,280
  validation scenes (`--val_steps` counts steps of `--batch_size`).
- **`best.pt` is chosen on the remeshed validation scenes** (`--val_remesh`), so
  it rewards matching by shape. Score `last.pt` too.
- **To keep 48 scenes as stored**, as first planned, run one stage with
  `--remesh_probability 0.625 --augment_probability 0.625 --w_consistency 0.5`
  and no `--val_remesh`. It will learn Breaking Bad's meshes, which helps on
  Breaking Bad's own test sets and not on the scans.
- **Memory.** `--micro_batch_scenes` is the only batch setting peak memory
  depends on. Raise it if preflight shows the headroom: it is faster and trains
  the same thing (the embedding term is computed scene by scene).
- **Across sessions.** Rerun the same command and it resumes from its
  `--checkpoint_dir`'s `last.pt`. On Kaggle, add the previous session's output
  as an input and append `--resume /kaggle/input/<that output>/<checkpoint dir>`.
  An interrupted epoch restarts from its first step.
- **What each stage writes.** `best.pt` (the epoch with the highest validation
  acc@5), `last.pt`, and `history.json`.

### Why the first two V8 runs stayed at chance

Both runs, in steps of 128 and then of 8, kept validation at chance (geodesic
126.6–126.9°, acc@5 0.000, match@1 0.004–0.006) for 30 epochs. Measured on the
Breaking Bad samples (`docs/PROJECT-STATE.md` §22), two things held them there:

1. **A fault in the embedding term, now fixed.** It drew its 1,024 anchors per
   forward pass, not per scene, and an anchor trains only if one of its
   positives is drawn with it. With four scenes a pass (`--micro_batch_scenes
   4`) that left a fifth to a seventh of the anchors that train (643 against
   3,176 on the samples as stored), and three quarters of each anchor's
   negatives were other objects' vertices, which matching never compares. Each
   scene now draws its own anchors and its own negatives; a pass of four trains
   as four passes of one.
2. **The stored scenes teach the mesh, not the shape.** Their two sides share
   every break vertex and triangle. Trained on them, the network matched them
   by tessellation, and matched the same scenes meshed on their own worse than
   untrained. That is also why v7 is at chance on the scans. Remeshed and
   unperturbed, the scenes taught matching by shape (match@1 on independently
   meshed copies 0.004 → 0.034–0.040 in 400 steps). Remeshed and perturbed
   from the first step, they barely moved it. Hence every scene remeshed, and
   shape first, the perturbations after.

The old logs cannot be compared with new ones: per scene, the logged embedding
term reads lower and match@1 higher for the same model.

- **If the curves stay flat anyway**, read the epoch lines before changing
  anything:
  - Whether the training `embedding` loss falls. If it does not, the
    optimization is the problem.
  - What `fracture head: precision … recall …` reads. Stage two matches what the
    head predicts.
  - Whether the matching works when the break is given: score the checkpoint on
    Everyday val with `--label_method marked`, as in §3. If that is far better,
    the embedding has learned and the head has not yet.

  On the synthetic slabs of `tests/test_contact.py` (four seeds each, the same
  training), V8's schedule with the default head reached 1–66° (median 32°,
  acc@10 0.56–1.00), against 21–91° (median 68°) for v7's schedule on dihedral
  labels. A head weighted ten times more (`--w_segment 10`) segmented far better
  (AUC 0.89–0.98) but matched worse (80–110°), so leave `--w_segment` at 1.

## 3. Test on every Breaking Bad object the training did not see

With Breaking Bad's split lists, an Everyday object is either train or val. So
the unseen objects are Everyday's val split and every object of the subsets not
trained on. Every break pattern of 2–20 pieces is scored (`--modes_per_scene 0`).

```bash
CKPT=$OUT/checkpoint-v8c/best.pt
TEST="--root_dir $BB --evaluate --checkpoint $CKPT --modes_per_scene 0 --max_fragments 20"

python -m scripts.train $TEST --data_subsets everyday_compressed --split val \
    --checkpoint_dir $OUT/eval/everyday-val --predictions $OUT/predictions/everyday-val
python -m scripts.train $TEST --data_subsets artifact_compressed --split all \
    --checkpoint_dir $OUT/eval/artifact --predictions $OUT/predictions/artifact
python -m scripts.train $TEST --data_subsets other_compressed --split all \
    --checkpoint_dir $OUT/eval/other --predictions $OUT/predictions/other
```

Each run writes `<split>_metrics.json` and `<split>_report.txt` to its
`--checkpoint_dir`. The report has the four benchmark numbers, overall and per
category:

| Paper's name | In the report | Unit |
|---|---|---|
| RMSE(R) | `Euler RMSE` (residual convention, as GARF reports it; the geodesic angle beside it) | degrees |
| RMSE(T) | `RMSE(T)` | object's largest side = 1; papers quote ×10⁻² |
| CD | `Chamfer (CD)` | squared distance; papers quote ×10⁻³ |
| PA | `part accuracy` | share of pieces with their own Chamfer distance below 0.01 |

The largest piece is the anchor, set to its true pose, as in GARF. Each run also
writes every scene's prediction, `$OUT/predictions/<name>/<subset>-<split>.zip`.
The zip holds one `.npz` per scene (matched rotations, placement, everything to
redraw it) and a CSV with one row per scene. Each `.npz` is a dump that
`python -m scripts.visualize_reassembly --dump <file>.npz --mode compare`
animates.

Optional:
- **The patterns of 21 pieces and more:** the same commands with
  `--min_fragments 21 --max_fragments 0`, each with its own `--checkpoint_dir`
  and `--predictions`.
- **A `volume_constrained-*` subset**, if your copy has one, with `--split all`.
  Its Everyday copy holds training shapes broken differently, and the report
  says how many.
- **The same test on scan-like scenes:** add `--remesh_edge 0.012 0.018
  --remesh_gap 0.03 --remesh_jitter 0.0005`.

## 4. Figures per object category

```bash
python -m scripts.make_figures --out $OUT/figures/v8-training --history V8=$OUT/checkpoint-v8c/history.json
python -m scripts.make_figures --out $OUT/figures/breaking_bad \
    --eval "Everyday val=$OUT/eval/everyday-val/val_metrics.json" \
    --eval "Artifact=$OUT/eval/artifact/all_metrics.json" \
    --eval "Other=$OUT/eval/other/all_metrics.json"
```

Each call writes PNG and PDF figures, their tables and an index, and zips the
folder beside it (`<out>.zip`):
- **Training:** the losses, validation acc@5/10/30, and the error per category
  and epoch.
- **Evaluation:** the rotation error by object type (box and violin), the share
  within each threshold, part accuracy by type, and the scores against pieces
  per scene.
- **Tables:** `tables/by_type.csv` and `tables/assembly_by_type.csv` have every
  number per category: part accuracy, RMSE(T), Chamfer, geodesic and Euler RMSE.

For one zip per subset, call it once per subset.

## 5. The final challenge: Fractura and Fantastic Breaks

```bash
SCAN="--evaluate --checkpoint $CKPT --split all --modes_per_scene 0 --max_fragments 0"

for S in fractura_real bone_real bone_synthetic; do
  python -m scripts.train $SCAN --root_dir $OUT/converted/fractura --data_subsets $S \
      --checkpoint_dir $OUT/eval/$S --predictions $OUT/predictions/$S
done
python -m scripts.train $SCAN --root_dir $OUT/converted/fantastic_breaks --data_subsets fantastic_breaks \
    --checkpoint_dir $OUT/eval/fantastic_breaks --predictions $OUT/predictions/fantastic_breaks

python -m scripts.make_figures --out $OUT/figures/scans \
    --eval "Fractura real=$OUT/eval/fractura_real/all_metrics.json" \
    --eval "Fractura bone real=$OUT/eval/bone_real/all_metrics.json" \
    --eval "Fractura bone synthetic=$OUT/eval/bone_synthetic/all_metrics.json" \
    --eval "Fantastic Breaks=$OUT/eval/fantastic_breaks/all_metrics.json"
```

Each report also prints the fracture head's precision and recall against the
scans' break (`fracture head: precision … recall …`), beside the dihedral
labels'. On Fractura's real objects that break is the estimate, so the head's
precision there is a lower bound (`docs/SIM2REAL.md` §1.3).

The oracle is optional and recommended. It reruns each test with the true (or
estimated) break in place of the fracture head's
(`docs/SIM2REAL.md` §1.3, §6.4):

```bash
for S in fractura_real bone_real bone_synthetic; do
  python -m scripts.train $SCAN --root_dir $OUT/converted/fractura --data_subsets $S \
      --label_method marked --checkpoint_dir $OUT/eval/$S-oracle
done
python -m scripts.train $SCAN --root_dir $OUT/converted/fantastic_breaks --data_subsets fantastic_breaks \
    --label_method marked --checkpoint_dir $OUT/eval/fantastic_breaks-oracle
```

If the oracle's numbers are far better than the head's, segmentation is what
remains to fix. If they are close, the descriptors are.
