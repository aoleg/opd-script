# Rebuild of kroma-v0.3.1-turbo-opd: report

I rebuilt kroma-v0.3.1-turbo-opd from Krea 2 Raw, Krea 2 Turbo and the Kroma teacher, and compared the result with the published file element by element. They match in 12,820,073,035 of 12,820,073,036 elements. The one element that differs sits exactly on the trim threshold of one tensor, where the last fp32 digit of the threshold decides it.

The first run used a plain top-10 % mask and matched 99.82 % of elements. The misses were all on the threshold, and they carried 10 % of the size of the change, because a top-10 % threshold is a large value. The published file uses the 0.9 quantile of |Δ| with linear interpolation and a strict comparison, which drops values equal to the threshold. bf16 deltas have many such ties. With that rule the rebuild is exact.

## Files

| Role | File |
|---|---|
| RAW | `raw.safetensors` (Krea 2 Raw) |
| TURBO | `turbo.safetensors` (Krea 2 Turbo) |
| TEACH | `kroma-sensei-booru-e6-teacher-velocity-v0.3.safetensors` |
| OPD | `kroma-v0.3.1-turbo-opd.safetensors` (published) |
| Output | `kroma-v0.3.1-turbo-opd-rebuilt.safetensors` |

RAW, TURBO, OPD and the output share one layout: 256 bf16 and 174 fp32 tensors. TEACH stores three of the fp32 tensors in bf16.

## Blocks

| Tensors | Identical elements | Error relative to the change |
|---|---|---|
| `blocks.0` to `blocks.27` | 100 % | 0 |
| `tproj` | 100 % | 0 |
| `tmlp` | 100 % (one element off) | 3.8 × 10⁻⁴ |
| `first`, `last` | 100 % | 0 |
| `txtfusion`, `txtmlp` (copied from TEACH) | 100 % | 0 |
| All | 1 in 1.3 × 10¹⁰ off | 1.3 × 10⁻⁵ |

The differing element is in `tmlp.2.weight`. It is off by 8.0 × 10⁻³, which is that tensor's threshold τ.

## Math

For each tensor W outside the text side:

- Δ = W_TEACH − W_RAW, in fp32
- τ = the 0.9 quantile of |Δ|, linear interpolation
- W_OPD = W_TURBO + Δ where |Δ| > τ, W_TURBO elsewhere

On the text side, W_OPD = W_TEACH. The `trim 0.9` field in the OPD metadata is this quantile.

## How the threshold was found

| Tensor | dtype | OPD keeps | Exact top 10 % keeps |
|---|---|---|---|
| `blocks.0.attn.wq` | bf16 | 9.986 % | 10.054 % |
| `blocks.13.mlp.down` | bf16 | 9.907 % | 10.153 % |
| `tproj.1` | fp32, bf16 delta | 9.892 % | 10.022 % |
| `blocks.5.mod.lin` | fp32 | 10.002 % | 9.999 % |
| `blocks.3.prenorm.scale` | fp32 | 10.010 % | 9.993 % |

In every tensor the OPD mask is a clean magnitude cut: all kept |Δ| are above all dropped |Δ|. In bf16 tensors OPD keeps less than 10 %, and the gap equals the group of ties at the threshold. In fp32 tensors OPD keeps slightly more than 10 %: 615 of 6,144 and 3,687 of 36,864. Those are exactly the counts above an interpolated 0.9 quantile.

## The model card and the weights

The published OPD file holds no training of its own: it is a merge of three public files by one formula. The model card at huggingface.co/lodestones/Kroma describes it as the result of on-policy distillation. The author's later account places that distillation inside the teacher.

### What the card says

I read the raw README and the file list through the Hub API, last modified 11 October 2026.

- The repository has three models: `kroma-v0.3-base`, `kroma-sensei-booru-e6-teacher-velocity-v0.3` and the recommended `kroma-v0.3.1-turbo-opd`, plus one LoRA.
- OPD is described as on-policy distillation: the student samples its own trajectories, the teacher corrects it on those states, and the model stays close to its original distribution. The card promises Turbo speed with no distillation tax: no mode collapse, no washed-out detail, no prompts that stop working.
- Settings for OPD: 8 to 12 steps, CFG 1.0 to 1.5, shift 1.15.
- LoRAs trained on the teacher or the base load on OPD unchanged.
- History covers v0.2 only: a continued full fine-tune, with the Krea 2 Turbo delta merged as a rank 512 LoRA. Nothing is said about how v0.3 and v0.3.1 were made, at what resolution or on what data. The download table and the workflow still name a v0.2 file that the repository no longer has.

### What the weights show

1. OPD is rebuilt to one element in 12.8 billion from Krea 2 Turbo, Krea 2 Raw and the teacher: Turbo plus the teacher's change above the 0.9 quantile, text side from the teacher. The OPD metadata names the same inputs and `trim 0.9`.
2. 90 % of OPD's elements equal the official Turbo bit for bit. Any training, on-policy or otherwise, would move nearly all of them.
3. The speed comes from Krea's own distillation. The Turbo delta in OPD is complete (coefficient 1.00 in every block) and carries guidance distillation (baked-in w of about 3 to 6). Kroma adds no distillation of its own.
4. The teacher holds the base whole (coefficient 0.99) and adds a change as large as the base's own training. That change spreads over about four times more directions than the base's and touches nearly every element. The author describes the teacher as a soup of thousands of sparse adapters plus his OPD of v0.3, and the weights fit that: no single LoRA or LoKr is behind it. Its norm, modulation and bias tensors changed too, so it had dense updates as well. It defaults to booru-style anime, works at about 1 MP and breaks composition at 2.2 MP. It shows no distillation and no baked-in guidance. "Velocity" in its name fits any Krea 2 model.

| Card | Weights |
|---|---|
| OPD trained by on-policy distillation | A merge of Turbo and the trimmed teacher − Raw difference; any distillation sits inside the teacher |
| Student trained on its own trajectories under the teacher | Every weight follows from three public files and one formula |
| Turbo speed with no distillation tax | The speed is the official Krea 2 Turbo, taken whole with all its traits |
| Stays close to the original distribution | True by construction: 90 % of the weights are Turbo |
| LoRAs from teacher or base load on OPD | Plausible: OPD holds Turbo and a tenth of the teacher's elements, all Krea 2 |
| v0.2: Turbo delta as a rank 512 LoRA | v0.3.1 takes the full Turbo delta; the card does not say so |

### What can be said

By the author's account, the on-policy distillation ran from his adapter soup into v0.3, and its result sits inside the teacher. The weights can neither confirm nor rule out that step. What they show is narrower: the published `kroma-v0.3.1-turbo-opd` is a fixed function of `raw`, `turbo` and the teacher. Anyone can rebuild it in four minutes with this script, and the merge added no training.

For use, this means:

- OPD's good behaviour comes almost entirely from Krea 2 Turbo. Kroma contributes a tenth of the teacher's elements, about half the energy of its change, plus the teacher's text side.
- OPD inherits part of the teacher's faults: the booru default and the composition errors on large canvases. Both showed in raw-opd, which is OPD minus the Turbo delta.
- A better result needs no new training. `kroma-base-trimD05-comp050-basevec` uses the same kind of sparse teacher merge on top of the whole base, halves it in the composition blocks and keeps the base's vectors, and renders cleaner.

### What the author says

The author answered twice: in [discussion #28](https://huggingface.co/lodestones/Kroma/discussions/28) and in a comment on this investigation. His pipeline, in plain words:

1. Start from Kroma v0.3 base.
2. Train thousands of sparse adapters on the image backbone at once. Each adapter updates its own set of salient weights, picked by gradient size, and the sets are kept apart to cut interference. He dropped LoRA because merged LoRAs drowned each other out.
3. Distill that soup of adapters back into v0.3 on-policy, to sharpen it. This is his OPD.
4. Move the resulting changes to Turbo with a top-K pick, budgeted at about 10 % of the DiT and all of the text side.

The teacher is the soup of all of it, his OPD of v0.3 included. His OPD distills knowledge into v0.3; the Turbo speed comes from the merge. The public sparse adapter code is in [RamTorchRun 16bea9e](https://github.com/lodestone-rock/RamTorchRun/commit/16bea9e9c8afb2d9b19b33645fa62524e17e6204); he says the version he uses is messier. That code trains one adapter on `nn.Linear` weights only, keeps norms and biases frozen, picks a global top-k mask by gradient with no per-layer quota, gives each matrix the budget of a rank 8 LoRA, and trains on 1024 px buckets.

| Author | Weights |
|---|---|
| The teacher is thousands of sparse adapters | Fits: its extra change is high rank, Gaussian, and touches 98.6 % of the elements |
| Adapters trained at 1024 px | Fits the composition failure at 2.2 MP |
| Adapters update matrices only | The teacher also changes all 165 norm, modulation and bias tensors (median 2.3 %, biases up to 12 %), so it had dense updates as well |
| OPD budgeted at 10 % of the DiT | That 10 % is the merge trim: exactly 10 % per tensor, vectors included. A global salient mask has no per-layer quota |
| OPD changes stay selectable by top-K because they got more updates | The teacher's extra change has no separate strong group: its top 10 % holds 0.44 of the energy and its top 1 % holds 0.086, as Gaussian noise does. Only `blocks.27.attn.wo` shows one |
| The OPD changes went to Turbo | The merge took the top 10 % of teacher − Raw. Inside that mask about 30 % of the energy is v0.3's own training and about 30 % the teacher's extra change; the rest is where the two add up |

### What Kroma v0.3 base is

The base looks like ordinary dense fine-tuning. Its change from Raw sits in far fewer directions than the teacher's extra change, and it moves every vector and the input layers hard.

| | v0.3 base (base − Raw) | Teacher's extra change (teacher − base) |
|---|---|---|
| Effective rank over dimension, median | 0.081 | 0.339 |
| Stable rank, median | 11.1 | 79.5 |
| Rank for 90 % of the energy over dimension, median | 0.19 | 0.39 |
| Energy in the top 10 % / top 1 % of elements | 0.445 / 0.087 | 0.443 / 0.086 |
| 1-D tensors changed | 165 of 165, median 2.5 %, `mod.lin` up to 11.6 % | 165 of 165, median 2.3 % |
| Input projection `first.weight` | 52 % | 8 % |
| Timestep MLP `tmlp.0.weight` | 78 % | 19 % |

Thousands of sparse adapters that cover the model spread over many directions, the way the teacher's extra change does. The base concentrates its change, as full fine-tuning does, and the card calls v0.2 a continued full fine-tune. Rank alone cannot prove the method: a few large dense adapters merged together would look similar.

## Run

Final rebuild 294 s on an RTX 5090 (first run 212 s), comparison about 2 minutes. Logs: `run_log_topk.txt` (first run), `run_log.txt` (final run), `diagnose_log.txt` (mask study), `count_diff_log.txt` (element count).
