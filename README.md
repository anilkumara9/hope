<!-- Copyright 2026 Google LLC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0) -->
# HOPE: Hilbert Operator for Progressive Encoding & DEFT Transfer Learning

This repository contains the reference implementations of **HOPE** (*Hilbert
Operator for Progressive Encoding*) and **DEFT** (*Dispersed Elastic
Fine-Tuning*):

1. **Data-Free Progressive Model Compression (`hope.py`, `compress_resnet50.py`,
   `hope_baselines.py`, `plot_compression_curves.py`):** HOPE compresses an
   ImageNet-pretrained ResNet-50 (Keras) through a sequence of analytical steps
   (pruning a neuron via `PRUNE`, merging two neurons into one synthesized
   parent neuron via `MERGE`, or evicting the residual branch of a bottleneck
   block via `EVICT`), selected greedily by a rate-distortion criterion. Every
   step is recorded together with the information needed to undo it. The
   resulting *curriculum* lets the compressed network be decoded step by step to
   any intermediate density, or all the way back to the pretrained network
   (lossless up to float32 round-off). HOPE is strictly data-free: every cost is
   evaluated in closed form from the weights and the stored BatchNorm parameters
   and statistics. It uses no training images, no calibration data, and no
   fine-tuning; ImageNet is only needed to measure the accuracy of the
   compressed networks. On ResNet-50, HOPE operates on the 16 bottleneck blocks
   (32 quadruplets, 7,552 neurons); the full curriculum has 3,066 steps (2,983
   prunes, 67 merges, 16 block evictions) and goes from density 1.0 down to
   0.1897, where every residual branch has been evicted.
2. **Cross-Domain Transfer Learning via Dispersed Elastic Fine-Tuning
   (`deft.py`):** DEFT uses HOPE's data-free progressive encoding curriculum on
   a pretrained source network to partition each hidden layer into a frozen
   **Universal Core** ($\mathcal{F}_{\mathrm{core}}$) and a plastic **Slack**
   ($\mathcal{F}_{\mathrm{slack}}$). By severing
   upstream-slack-to-downstream-core connections at initialization ($t = 0$) and
   restricting gradient updates to slack-destination parameters during target
   fine-tuning ($t > 0$), DEFT bounds the one-time initialization shock by the
   capacity of the slack subspace and guarantees exact zero dynamic forgetting
   on the source task across all fine-tuning epochs (see
   [Cross-domain transfer learning with DEFT
   (`deft.py`)](#cross-domain-transfer-learning-with-deft-deftpy)).

For full derivations, see [*Hilbert Operator for Progressive Encoding (HOPE): A
Mathematical Framework for Deconstructing Learned Representations in Deep
Networks*](https://arxiv.org/abs/2607.21366) ([Citation](#citation)).

## Method

HOPE treats each hidden neuron of a bottleneck block (a convolution filter with
its BatchNorm (BN), its ReLU, and its slice of the next convolution) as a rank-1
Hilbert–Schmidt operator $f_i = g_i \otimes w_{\mathrm{out},i}$. The norm of
this operator (the neuron's *capacity*) and the inner products between neurons
are computed in closed form under a Gaussian surrogate built from the stored BN
parameters, under which the pre-activation of neuron $i$ is
$y_i \sim \mathcal{N}(\beta_i, \gamma_i^2)$. Three actions are available:

| Action | What it does | Cost $\mathcal{J}$ |
|---|---|---|
| `PRUNE` | removes one neuron $i$ | $\mathcal{J}^\circ_{\mathrm{prune}}(i) = \dfrac{N\,\lVert f_i \rVert_\circ}{E_a - \lVert f_i \rVert_\circ}$ |
| `MERGE` | replaces two neurons $(i, j)$ of a layer by one synthesized parent neuron $f_p = s^{\ast}\psi^{\ast}$ | $\mathcal{J}^\circ_{\mathrm{merge}}(i,j) = \dfrac{N\sqrt{\lVert f_i - f_p \rVert_\circ^2 + \lVert f_j - f_p \rVert_\circ^2}}{E_a - \lVert f_i \rVert_\circ - \lVert f_j \rVert_\circ + \lVert f_p \rVert_\circ}$ |
| `EVICT` | removes the residual branch of a bottleneck block (internal layers $l \in \{1, 2\}$) | $\mathcal{J}_{\mathrm{evict}} = \sum_{l \in \{1,2\}} N_l\,\dfrac{E_a^{(l)}}{E_{\mathrm{init}}^{(l)}}$ |

At every step the encoder executes the admissible action with the smallest
distortion rate $\mathrm{DR} = \mathcal{J} / \Delta P^{\mathrm{init}}$, the cost
per released parameter, until no action is left. Each executed action is stored
as an `InstructionPacket`; the packets in reverse order form the curriculum.

Components of the method:

- **Closed-form ReLU kernels.** With $c_i = \beta_i / |\gamma_i|$, the mean
  $m_i = |\gamma_i|\phi(c_i) + \beta_i\Phi(c_i)$ and second moment
  $K(i,i) = (\gamma_i^2 + \beta_i^2)\Phi(c_i) + \beta_i|\gamma_i|\phi(c_i)$ of
  each ReLU activation have closed forms, and the covariance
  $C(i,j) = K(i,j) - m_i m_j$ of two biased ReLUs is evaluated exactly through
  the bivariate normal CDF $\Phi_2(x, y; \rho)$, computed with Owen's T
  function.
- **Mean-centred costs and parent synthesis.** Capacities
  $\lVert f_i \rVert_\circ = \lVert w_{\mathrm{out},i} \rVert\sqrt{v_i}$, prune
  and merge costs, and the synthesis of merge parents are computed on the
  zero-mean fluctuation subspace $\mathcal{H}_0^\perp$, i.e. with the centred
  kernels $v_i = C(i,i) = K(i,i) - m_i^2$ and $C(i,j) = K(i,j) - m_i m_j$.
- **Optimal parent neuron.** A merge writes a parent neuron
  $f_p = s^{\ast}\psi^{\ast}$ whose input direction $u_c$, output direction
  $v^{\ast}$, optimal scale $s^{\ast}$ and physical parameters (raw filter, BN
  $\gamma_p, \beta_p, \mu_p, \sigma_p^2$) are computed in closed form.
- **Consumer BatchNorm mean compensation.** The mean shift that a prune or merge
  causes at the input of the downstream BN is absorbed into that BN's running
  mean ($\mu^b \leftarrow \mu^b - m_i \sum_t \chi_t w_{t,i}$ for a prune;
  $\mu^b\leftarrow\mu^b+\sum_t\chi_t(m_p w'_{t,p}-m_i w_{t,i}-m_j w_{t,j})$ for
  a merge), with zero-padding tap fractions $\chi_t$ for 3×3 consumers.
- **Coherence-weighted block eviction.** When a residual branch is evicted, its
  kernels and BN tensors are zeroed except the $\beta$ of its last BN, which
  keeps the coherence-weighted Wiener shift $\delta_3^{\ast} = \alpha_3 \beta_3$
  with $\alpha_3 = \beta_3^2 / (\gamma_3^2 + \beta_3^2)$.
- **Rate-distortion greedy selection.** All candidate actions of all layers
  compete through their distortion rate
  $\mathrm{DR} = \mathcal{J} / \Delta P^{\mathrm{init}}$; merge candidates are
  scored in $O(1)$ per pair from a decoupled all-pairs cache, and each executed
  action updates only the acting layer's local cache ($O(N)$ for a merge, $O(1)$
  for a prune or eviction).

## Repository contents

| File | Description |
|---|---|
| [`hope.py`](hope.py) | The HOPE library: quadruplet discovery, encoder and decoder. Depends only on NumPy and SciPy. |
| [`hope_baselines.py`](hope_baselines.py) | The seven data-free structured compression baselines (`L1_IN`, `L1_JOINT`, `BN_SCALE`, `FPGM`, `L1_JOINT_BIASCORR`, `WS_COSINE_MERGE`, `SYNFLOW`), producing `InstructionPacket` curricula compatible with `hope.hope_decode`. Depends only on NumPy and SciPy. |
| [`plot_compression_curves.py`](plot_compression_curves.py) | Standalone plotting script and library utility that renders the `(density, accuracy)` comparison figure across HOPE and the seven baselines. |
| [`compress_resnet50.py`](compress_resnet50.py) | Command-line runner: compresses Keras' pretrained ResNet-50 with HOPE, any individual baseline, or all methods (`--method ALL`), records top-1 accuracy versus density, saves the compressed weights and curriculum, and checks lossless decoding. |
| [`deft.py`](deft.py) | Dispersed Elastic Fine-Tuning (DEFT): cross-domain transfer learning pipeline (20-class CIFAR-100 $\to$ 10-class SVHN) on an 8-layer BN-VGG network (`CNN_8Layer`), together with `Head-Only FT`, `Standard Full FT`, `EWC`, and `PEFT (BN-Tuning)` baselines (Section 10.2 & Appendices H–I). |
| [`test_hope.py`](test_hope.py) | Self-test of `hope`, `hope_baselines`, and `plot_compression_curves` on synthetic ResNet and mixed Conv2D/Dense networks (NumPy and SciPy only). |
| [`requirements.txt`](requirements.txt) | Python dependencies. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution and CLA guidelines. |
| [`LICENSE`](LICENSE) | Apache License 2.0. |
| [`.gitignore`](.gitignore) | Ignores Python caches, the runner's default and example output directories, and evaluation-cache files. |

## Installation

```bash
python3 -m venv hope_venv
source hope_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt         # library, self-test, ResNet-50 runner, and DEFT benchmark
pip install "numpy>=2.0" "scipy>=1.13"  # minimal install: library and self-test only
```

- `hope.py`, `hope_baselines.py`, and the self-test need only **NumPy ≥ 2.0**
  and **SciPy**. NumPy 2 is required: the arithmetic follows NumPy 2's scalar
  type-promotion rules (NEP 50); with NumPy 1.x the encoder can differ in the
  last bits and in tie-breaks between candidate actions, so the reference
  results may not be reproduced.
- The ResNet-50 runner (`compress_resnet50.py`) additionally needs Keras 3,
  TensorFlow, and TensorFlow Datasets (versions in
  [`requirements.txt`](requirements.txt)). TensorFlow and TensorFlow Datasets
  read and preprocess ImageNet (once, into a cache) whatever the Keras backend.
  Optional: matplotlib (accuracy-versus-density plot), JAX or PyTorch (other
  Keras backends), pytest.
- The cross-domain transfer benchmark ([`deft.py`](deft.py)) uses **JAX**,
  **Optax**, **Keras 3** (with `KERAS_BACKEND=jax`), **TensorFlow**, and
  **TensorFlow Datasets** (all specified in
  [`requirements.txt`](requirements.txt)). `deft.py` automatically detects
  whether an NVIDIA GPU or multi-core CPU is available at runtime
  (`detect_hardware_backend()`) and configures XLA flags accordingly.
- Tested with Python 3.13, NumPy 2.4.1 / SciPy 1.17.0 (the library also with
  NumPy 2.5.1 / SciPy 1.18.0), Keras 3.13 with the TensorFlow 2.20, JAX 0.8, and
  PyTorch 2.13 backends, Optax 0.2.8, and TensorFlow Datasets 4.9.9. The core
  compression library itself uses only Python 3.9-compatible syntax.

## Quick start

### 1. Self-test (no Keras, no ImageNet)

```bash
python test_hope.py          # or: python -m pytest test_hope.py
```

The fourteen tests run in a few seconds. They check quadruplet discovery, HOPE
encoding to completion with all three action types, mixed 4D Conv2D and 2D Dense
quadruplets with DEFT curriculum metadata, that the input weights are
not modified, the density history, that encoding is deterministic and
independent of the evaluation callback, lossless full decoding, that progressive
decoding retraces every intermediate encoder state, in-place single-step
decoding, a pickle round trip of the curriculum, incremental CSV output, all
seven structured compression baselines (including lossless decoding), and the
curve-plotting utility.

### 2. Compress ResNet-50 without ImageNet

```bash
python compress_resnet50.py --output_dir runs/encode_only --eval_stride 0
```

Keras downloads the pretrained weights (or pass
`--weights path/to/resnet50_weights_tf_dim_ordering_tf_kernels.h5`). The script
runs the encoder to completion, saves the compressed weights and the curriculum,
and checks that decoding the full curriculum restores the pretrained weights.
The log includes lines such as

```
... Encoding finished: 3066 steps {'EVICT': 16, 'PRUNE': 2983, 'MERGE': 67}, final density 0.1897, ...
... Decoding check PASSED: max |decoded - pretrained| = 5.96e-08 over 320 tensors (0.1s).
```

### 3. Accuracy along the curriculum (ImageNet)

The runner reads the ImageNet 2012 validation split through TensorFlow Datasets,
which requires a one-time manual download: register at
<https://image-net.org/download-images>, download `ILSVRC2012_img_val.tar` (the
training archive is not needed) and prepare the dataset:

```bash
tfds build imagenet2012 --manual_dir /path/to/directory/with/the/tar   # add --data_dir DIR for a custom location
```

(see the
[TFDS `imagenet2012`
page](https://www.tensorflow.org/datasets/catalog/imagenet2012) for details).
Then:

```bash
python compress_resnet50.py --output_dir runs/resnet50                          # default Keras backend (HOPE)
python compress_resnet50.py --output_dir runs/resnet50 --method ALL             # HOPE + all 7 baselines
python compress_resnet50.py --output_dir runs/resnet50 --backend jax --batch_size 32
python compress_resnet50.py --output_dir runs/resnet50 --data_dir DIR           # if prepared with --data_dir DIR
```

With the defaults (`--eval_stride 10`, `--num_eval_images 10144`) the runner
evaluates the pretrained network (step 0), the network after every 10 encoding
steps, and the final network: 308 evaluations in total. The first run
preprocesses the evaluation images once into a 6.1 GB cache (about 0.6 MB per
image, see `--cache_dir`) that later runs reuse.

### 4. Plotting comparison curves

To plot the `(density, accuracy)` curves from a directory containing
`results_HOPE.csv` and any subset of the baseline CSV files
(`results_L1_IN.csv`, `results_L1_JOINT.csv`, `results_BN_SCALE.csv`,
`results_FPGM.csv`, `results_L1_JOINT_BIASCORR.csv`,
`results_WS_COSINE_MERGE.csv`, `results_SYNFLOW.csv`):

```bash
python plot_compression_curves.py --data_dir runs/resnet50 --output runs/resnet50/accuracy_vs_density.png
```

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--method` | `HOPE` | `HOPE`, one of the 7 baselines (`L1_IN`, `L1_JOINT`, `BN_SCALE`, `FPGM`, `L1_JOINT_BIASCORR`, `WS_COSINE_MERGE`, `SYNFLOW`), or `ALL` |
| `--output_dir` | `hope_resnet50_output` | directory for all outputs |
| `--weights` | `imagenet` | `imagenet` (downloaded by Keras) or the path of a Keras ResNet-50 weights file |
| `--backend` | `KERAS_BACKEND` or the Keras configuration file | Keras backend: `tensorflow`, `jax` or `torch` |
| `--eval_stride` | `10` | evaluate every this many encoding steps; `0` disables evaluation (no ImageNet needed) |
| `--num_eval_images` | `10144` | number of validation images, taken from the start of the split (at most 50,000) |
| `--batch_size` | `64` | inference batch size |
| `--data_dir` | TFDS default (usually `~/tensorflow_datasets`) | TFDS data directory with the prepared `imagenet2012` dataset |
| `--tfds_name` | `imagenet2012` | TFDS dataset name, optionally with a version, e.g. `imagenet2012:5.1.0` (the version used for the results below) |
| `--cache_dir` | `$XDG_CACHE_HOME/hope` or `~/.cache/hope` | directory of the preprocessed evaluation images |
| `--no_plot` | off | do not save the accuracy-versus-density plot |
| `--log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

Exit codes: 0 success, 1 decoding check failed, 2 setup error (for example,
ImageNet not prepared), 130 interrupted.

### Outputs

Written to `--output_dir`:

| File | Content |
|---|---|
| `density_accuracy.csv` | columns `step,density,top1_accuracy` (density and accuracy as fractions); step 0 is the pretrained network. Rows are appended as soon as they are computed, so partial results survive an interruption. Only written in single-method runs when evaluation is enabled (`--eval_stride > 0`). |
| `results_<METHOD>.csv` | two-column CSV (`density,accuracy`) for the evaluated method(s) (all eight methods when `--method ALL`), ready for `plot_compression_curves.py`. Only written when evaluation is enabled (`--eval_stride > 0`). |
| `compressed_weights.npz` | final compressed weights, keyed by Keras weight path such as `conv2_block1_1_conv/kernel` (about 103 MB; single-method runs only) |
| `curriculum.pkl` | pickled dict with the curriculum (key `"curriculum"`, decoding order), the quadruplets, the weight paths and the initial number of nonzero parameters (about 125 MB; single-method runs only); unpickling requires `hope` to be importable |
| `summary.json` | configuration, software versions, number of steps and actions, final density, pretrained accuracy, lossless decoding check and timings (single-method runs only; `--method ALL` skips these per-model artifacts and the decoding check) |
| `accuracy_vs_density.png` | accuracy-versus-density plot (only when evaluation is enabled and `--no_plot` is not set; single-method runs require `matplotlib`, while `--method ALL` falls back to Pillow if `matplotlib` is absent) |

## Using the library

`hope` works on a plain dictionary that maps Keras weight paths to NumPy arrays
(kernels in `(height, width, in_channels, out_channels)` layout):

```python
import collections

import keras
import hope
import hope_baselines
import plot_compression_curves

# Pretrained ResNet-50 as {Keras weight path: NumPy array}, e.g. "conv2_block1_1_conv/kernel".
model = keras.applications.ResNet50(weights="imagenet")
weights = {v.path: keras.ops.convert_to_numpy(v) for v in model.weights}

# The quadruplets to compress: two per bottleneck block, 32 for ResNet-50.
quadruplets = hope.find_bottleneck_quadruplets(weights)


def eval_fn(current_weights):
    """Optional. Called every `eval_stride` steps with the current weights, which must not be modified."""
    model.set_weights([current_weights[v.path] for v in model.weights])
    return my_accuracy(model)  # your own evaluation; the returned float is recorded in `history`


compressed, curriculum, history = hope.hope_encode(
    weights, quadruplets, eval_fn=eval_fn, eval_stride=10, output_file="results_HOPE.csv"
)
print(len(curriculum), collections.Counter(packet.action for packet in curriculum))
# 3066 Counter({'PRUNE': 2983, 'MERGE': 67, 'EVICT': 16})

# Progressive decoding: the network after the first k encoding steps ...
k = 570
weights_k = hope.hope_decode(compressed, curriculum, num_steps=len(curriculum) - k)
# ... and the pretrained network, restored up to float32 round-off.
restored = hope.hope_decode(compressed, curriculum)

# Run all 7 data-free baselines and plot the comparison figure:
hope_baselines.run_all_baselines(weights, quadruplets, eval_fn=eval_fn, eval_stride=10, output_dir=".")
plot_compression_curves.plot_compression_curves(data_dir=".", save_path="accuracy_vs_density.png")
```

- `find_bottleneck_quadruplets(layer_names)` accepts Keras layer names or weight
  paths and returns, for every bottleneck block `X`, the quadruplets
  `(X_1_conv, X_1_bn, X_2_conv, X_2_bn)` and
  `(X_2_conv, X_2_bn, X_3_conv, X_3_bn)` (producer convolution and BN, consumer
  convolution and BN), in network order.
- `hope_encode(weights, quadruplets, eval_fn=None, eval_stride=10,
  output_file=None)` runs the encoder until no admissible action is left (there
  is no target density) and does not modify `weights`. It returns the fully
  compressed weights, the curriculum (one `InstructionPacket` per step, in
  decoding order: `curriculum[0]` undoes the last encoding step) and `history`,
  the list of `(density, eval_fn value)` pairs recorded every `eval_stride`
  steps (the value is 0.0 without `eval_fn`). If `output_file` is given (e.g.
  `"results_HOPE.csv"`), the `(density, accuracy)` trajectory (including the
  initial network at density `1.0` and the final compressed network) is also
  written incrementally as a two-column CSV (`density,accuracy`). Density is the
  number of nonzero entries over all tensors divided by its initial value.
- `baseline_encode(weights, quadruplets, benchmark_method="L1_IN", ...)` in
  `hope_baselines.py` runs any of the seven data-free baselines (`L1_IN`,
  `L1_JOINT`, `BN_SCALE`, `FPGM`, `L1_JOINT_BIASCORR`, `WS_COSINE_MERGE`,
  `SYNFLOW`) and returns `(compressed_weights, curriculum, results)`, where
  `results` records `(density, eval_fn value)` pairs at steps `0..4` (starting
  at `density = 1.0`), every `eval_stride` steps, and the final compressed
  network; `run_all_baselines(...)` runs all seven and writes
  `results_<METHOD>.csv`.
- `hope_decode(weights, curriculum, num_steps=None)` returns a copy of `weights`
  with the first `num_steps` packets applied (all if `None`);
  `hope_decode_step(weights, packet)` applies one packet in place. Decoding
  reproduces the encoder's intermediate networks up to float32 round-off in the
  restored producer filters.
- Each `InstructionPacket` has an `action` (`"PRUNE"`, `"MERGE"` or `"EVICT"`),
  `indices` (PRUNE: `(-1, i, conv)`; MERGE: `(i, j, conv)`, with the parent
  written into neuron `i` and neuron `j` removed; EVICT: the block's three
  convolutions), the `quadruplet` acted upon, and copies of the tensors needed
  to undo the step.

The runner's outputs are decoded the same way, for example to obtain the network
after 570 encoding steps or to walk through the curriculum one step at a time:

```python
import pickle

import numpy as np
import hope  # must be importable to unpickle the curriculum

compressed = dict(np.load("runs/resnet50/compressed_weights.npz"))
with open("runs/resnet50/curriculum.pkl", "rb") as f:
    curriculum = pickle.load(f)["curriculum"]

weights_570 = hope.hope_decode(compressed, curriculum, num_steps=len(curriculum) - 570)

current = {name: array.copy() for name, array in compressed.items()}
for packet in curriculum:  # from the fully compressed network back to the pretrained one
    hope.hope_decode_step(current, packet)
```

## Evaluation protocol

This is the protocol implemented by `compress_resnet50.py` and used for the
results below:

- **Images:** the first 10,144 images (`--num_eval_images`) of the ImageNet 2012
  validation split as served by TensorFlow Datasets (`imagenet2012`, version
  5.1.0), in TFDS order without shuffling.
- **Preprocessing:** resize to 224×224 with `tf.image.resize` (bilinear, no
  antialiasing, aspect ratio not preserved, no crop), then
  `keras.applications.resnet50.preprocess_input` ("caffe" mode: RGB to BGR and
  subtraction of the ImageNet channel means), in float32.
- **Metric:** top-1 accuracy of Keras' ResNet-50 (1000-way classifier) with the
  evaluated weights loaded by `model.set_weights`.
- **Density:** the number of nonzero entries over all weight tensors of the
  network (including biases, BN parameters and statistics, and the parts that
  HOPE never modifies: stem, shortcut projections, and classifier), divided by
  the same count for the pretrained network (25,610,216).
- **Step:** step 0 is the pretrained network; step *t* is the network after *t*
  encoding steps.

Accuracies measured with this simple preprocessing are not directly comparable
with numbers obtained with other preprocessing (such as resize and central crop)
or on the full validation set.

## Reference results

Top-1 accuracy of the HOPE-compressed ResNet-50 on the first 10,144 ImageNet
validation images, at selected density milestones along the curriculum:

| Density | Top-1 accuracy |
|---:|---:|
| 1.0000 (pretrained) | 71.25% |
| 0.9499 | 69.48% |
| 0.9010 | 66.92% |
| 0.8520 | 64.87% |
| 0.7261 | 55.70% |
| 0.6055 | 52.05% |
| 0.5917 | 51.09% |
| 0.5523 | 49.12% |
| 0.4928 | 45.28% |
| 0.4537 | 39.32% |
| 0.4184 | 33.88% |
| 0.3729 | 26.58% |
| 0.3382 | 18.88% |
| 0.2817 | 9.45% |
| 0.2296 | 4.35% |
| 0.1897 | 0.71% |

## Math-to-code map

All code symbols below refer to [`hope.py`](hope.py).

| Quantity | Formula | Code |
|---|---|---|
| BN-folded input weights and bias; joint parameter rows | $w_{\mathrm{in},i}^{\mathrm{eff}} = \frac{\gamma_i}{\sqrt{\sigma_i^2 + \epsilon}}\,w_{\mathrm{in},i}$, $b_i = \beta_i - \frac{\gamma_i}{\sqrt{\sigma_i^2 + \epsilon}}\,\mu_i$, row $[w_{\mathrm{in},i}^{\mathrm{eff}}, w_{\mathrm{out},i}]$ | `read_quadruplet`, `BN_EPSILON` |
| Quadruplets $Q_1$, $Q_2$ of a bottleneck block | $Q_1 = (K_1, \mathrm{BN}_1, K_2, \mathrm{BN}_2)$, $Q_2 = (K_2, \mathrm{BN}_2, K_3, \mathrm{BN}_3)$ | `find_bottleneck_quadruplets`, `Quadruplet` |
| ReLU mean $m_i$ and variance $v_i = C(i,i)$ under $y_i \sim \mathcal{N}(\beta_i, \gamma_i^2)$ ($c_i = \beta_i / |\gamma_i|$) | $m_i = |\gamma_i|\phi(c_i) + \beta_i\Phi(c_i)$, $K(i,i) = (\gamma_i^2 + \beta_i^2)\Phi(c_i) + \beta_i|\gamma_i|\phi(c_i)$, $v_i = K(i,i) - m_i^2$ | `relu_mean_scalar`, `relu_variance`, `relu_variance_scalar` |
| Warped correlation $\hat\rho_{ij}$ | $\kappa = \frac{\rho_{\mathrm{eff}}}{1 - \rho_{\mathrm{eff}}^2}\frac{|\gamma_i|}{\lVert w_{\mathrm{in},i}^{\mathrm{eff}} \rVert}\frac{|\gamma_j|}{\lVert w_{\mathrm{in},j}^{\mathrm{eff}} \rVert}$, $\hat\rho_{ij} = \frac{2\kappa}{1 + \sqrt{1 + 4\kappa^2}}$ | `warped_correlation` (vectorized in `build_merge_cache`) |
| Bivariate normal CDF $\Phi_2$ via Owen's T | $\Phi_2(x, y; \rho) = \tfrac{1}{2}(\Phi(x) + \Phi(y)) - T(x, a_x) - T(y, a_y) - \tfrac{1}{2}\mathbf{1}[xy < 0]$ | `bivariate_normal_cdf` |
| Centred cross-kernel $C(i,j)$ | $K(i,j) = |\gamma_i||\gamma_j|\big[(c_i c_j + \hat\rho)\Phi_2 + c_i\phi(c_j)\Phi(c_{i\mid j}) + c_j\phi(c_i)\Phi(c_{j\mid i}) + \frac{\sqrt{1-\hat\rho^2}}{2\pi}e^{-\frac{c_i^2 - 2\hat\rho c_i c_j + c_j^2}{2(1-\hat\rho^2)}}\big]$, $C(i,j) = K(i,j) - m_i m_j$ | `relu_covariance`, `relu_covariance_scalar` |
| Centred capacity | $\lVert f_i \rVert_\circ = \lVert w_{\mathrm{out},i} \rVert \sqrt{v_i}$, $E_a = \sum_{k \in \mathrm{active}} \lVert f_k \rVert_\circ$ | `neuron_capacities`, `CostManager.neuron_capacity` |
| Centred prune and merge costs $\mathcal{J}^\circ_{\mathrm{prune}}$, $\mathcal{J}^\circ_{\mathrm{merge}}$ | $\mathcal{J}^\circ_{\mathrm{prune}} = \frac{N\lVert f_i \rVert_\circ}{E_a - \lVert f_i \rVert_\circ}$, $\mathcal{J}^\circ_{\mathrm{merge}} = \frac{N\sqrt{\lVert f_i - f_p \rVert_\circ^2 + \lVert f_j - f_p \rVert_\circ^2}}{E_a - \lVert f_i \rVert_\circ - \lVert f_j \rVert_\circ + \lVert f_p \rVert_\circ}$ | `_prune_candidate`, `_merge_candidate` |
| Parameter footprints $\Delta P^{\mathrm{init}}$, $\Delta P_{\mathrm{evict}}$ | $\Delta P_l^{\mathrm{init}} = n_l + c_l + 4$, $\Delta P_{\mathrm{evict}} = \sum_{l \in \{1,2\}} N_l \Delta P_l^{\mathrm{init}} + 3 d_{\mathrm{amb}}$ | `_greedy_scan`, `_evict_candidate` |
| Distortion rate and action selection | $\mathrm{DR}(a) = \mathcal{J}(a) / \Delta P^{\mathrm{init}}(a)$, $a^{\ast} = \arg\min_a \mathcal{J}(a) / \Delta P^{\mathrm{init}}(a)$ | `_greedy_scan` |
| Parent direction $u_c$ (polarity check) and output direction $v^{\ast}$ | $u_c = \arg\max_{u \in \{\pm \hat u\}} \frac{\lVert C(u,i)w_{\mathrm{out},i} + C(u,j)w_{\mathrm{out},j} \rVert}{\sqrt{C(u,u)}}$, $v^{\ast} = \frac{C(u_c,i)w_{\mathrm{out},i} + C(u_c,j)w_{\mathrm{out},j}}{\lVert C(u_c,i)w_{\mathrm{out},i} + C(u_c,j)w_{\mathrm{out},j} \rVert}$ | `merge_pair_geometry`, `build_merge_cache` |
| Optimal parent scale $s^{\ast}$, capped at $S = \lVert f_i \rVert_\circ + \lVert f_j \rVert_\circ$ | $s^{\ast} = \min\!\big(\max\!\big(0, \frac{a + b E_{\mathrm{rem}}}{2 E_{\mathrm{rem}} + b}\big), S\big)$ with $a = \lVert f_i \rVert_\circ^2 + \lVert f_j \rVert_\circ^2$, $b = \langle \psi^{\ast}, f_i + f_j \rangle_\circ$, $E_{\mathrm{rem}} = \max(E_a - S, \epsilon)$ | `_merge_candidate` |
| Decoupled all-pairs merge cache | $E_a$-independent pair quantities $(S, a, b, \lVert \psi^{\ast} \rVert_\circ, \lVert \psi^{\ast} \rVert_\circ^2, \langle \psi^{\ast}, f_i + f_j \rangle_\circ)$ | `build_merge_cache`, `update_merge_cache_row`, `CostManager` |
| Input/output scaling with the Frobenius ratio $R_F = \lVert W_{\mathrm{in}} \rVert_F / \lVert W_{\mathrm{out}} \rVert_F$ | $\tilde w_{\mathrm{in},p} = \sqrt{s^{\ast} R_F}\,u_c$, $w_{\mathrm{out},p} = \sqrt{s^{\ast}/R_F}\,C(u_c,u_c)^{-1/2} v^{\ast}$ | `merge_pair_geometry`, `_synthesize_parent` |
| Parent's raw filter and BN parameters $\gamma_p$, $\beta_p$, $\mu_p$, $\sigma_p^2$ | $\gamma_p = \sqrt{c_1^2 \gamma_i^2 + c_2^2 \gamma_j^2 + 2 c_1 c_2 \hat\rho_{ij}|\gamma_i||\gamma_j|}$, $\beta_p = c_1 \beta_i + c_2 \beta_j$, $\sigma_p^2 = \max(0, \gamma_p^2 - \epsilon)$, $\mu_p = \frac{\beta_p - b_p}{\gamma_p / \sqrt{\sigma_p^2 + \epsilon}}$ | `_synthesize_parent`, `span_coefficients`, `_execute_merge` |
| Consumer BN mean compensation $\Delta\mu^b$ and tap fractions $\chi_t$ | prune: $\Delta\mu^b = -m_i \sum_t \chi_t w_{t,i}$; merge: $\Delta\mu^b = \sum_t \chi_t (m_p w'_{t,p} - m_i w_{t,i} - m_j w_{t,j})$; $3\times 3$: $\chi = r r^\top$, $r = (\frac{H-1}{H}, 1, \frac{H-1}{H})$ | `_execute_prune`, `_execute_merge`, `_compensate_consumer_mean`, `consumer_tap_fractions` |
| Eviction cost $\mathcal{J}_{\mathrm{evict}}$ and its anchor $E_{\mathrm{init}}$ | $\mathcal{J}_{\mathrm{evict}} = \sum_{l \in \{1,2\}} N_l \frac{E_a^{(l)}}{E_{\mathrm{init}}^{(l)}}$ | `_evict_candidate`, `CostManager.initial_capacity` |
| Coherence-weighted Wiener shift $\delta_3^{\ast} = \alpha_3 \beta_3$ | $\beta_3 \leftarrow \delta_3^{\ast} = \alpha_3 \beta_3$, $\alpha_3 = \frac{\beta_3^2}{\gamma_3^2 + \beta_3^2}$ | `_execute_evict` |
| Encoding loop and capacity ledger | greedy loop over $a^{\ast} = \arg\min_a \mathcal{J}(a) / \Delta P^{\mathrm{init}}(a)$ | `hope_encode`, `_initialize_caches`, `_greedy_scan`, `_collapse_evicted_layers` |
| Decoding | reverse curriculum application ($w_{\mathrm{raw}} = w_{\mathrm{in}}^{\mathrm{eff}} / (\gamma/\sqrt{\sigma^2+\epsilon})$) | `InstructionPacket`, `hope_decode_step`, `hope_decode` |

## Cross-domain transfer learning with DEFT (`deft.py`)

[`deft.py`](deft.py) implements **Dispersed Elastic Fine-Tuning (DEFT)** from
Section 10.2, Appendix H, Appendix I, and Algorithm 3 of
[the HOPE paper](https://arxiv.org/abs/2607.21366). DEFT
runs HOPE's progressive encoder (`hope.hope_encode`) on a pretrained source
network to completion and uses the resulting pruning costs to partition each
hidden layer into a frozen **Universal Core** $\mathcal{F}_{\mathrm{core}}$ and
a plastic **Slack** $\mathcal{F}_{\mathrm{slack}}$:

- **Merge-pair core placement
  ($\mathcal{M} \subseteq \mathcal{F}_{\mathrm{core}}$):** Every neuron that
  participates in a `MERGE` step (both the surviving root $i$ and the absorbed
  neuron $j$) is placed in the frozen core ($E_i = E_j = 0$) and excluded from
  the threshold pool $\mathcal{C}$, so that every slack neuron is an unmerged
  neuron priced on its exact original source weights.
- **Binary elasticity ($E_i \in \{0, 1\}$):** Over all unmerged pruned neurons
  $i \notin \mathcal{M}$ with remaining mean-centred capacity
  $E_b^{(i)} > \epsilon$ ($\epsilon = 10^{-5}$), DEFT computes the $P$-th
  percentile threshold $J_P = \operatorname{Percentile}(\mathcal{C}, P)$ of
  $\mathcal{J}_{\mathrm{prune}}^{(i)}=N^{(i)}\lVert f_i\rVert_\circ/E_b^{(i)}$
  and assigns
  $$
  E_i = \mathbf{1}[i \notin \mathcal{M} \wedge E_b^{(i)} > \epsilon
    \wedge \mathcal{J}_{\mathrm{prune}}^{(i)} < J_P].
  $$
- **Structural mask at initialization ($t = 0$):** All weight connections from
  upstream slack neurons ($E_{\mathrm{in},j} > 0$) to downstream core neurons
  ($E_{\mathrm{out},u} = 0$) and to the source classification head are severed
  at $t = 0$ ($\mathbf{W}_0 = \mathbf{M} \odot \mathbf{W}_{\mathrm{source}}$,
  Axiom I.1). By Theorem I.1, the static shock injected into each downstream
  core is bounded by the capacity of the severed slack neurons.
- **Exact source preservation ($t > 0$):** Target gradients (with
  $5 \times 10^{-4}$ $L_2$ weight decay on tensors of rank $> 1$ included inside
  $\mathcal{L}_{\mathrm{target}}$) are multiplied by
  $E_{\mathrm{out}} \in \{0, 1\}$ before the optimizer step (Axiom I.2). Core
  parameters and severed weights never change; when evaluated with the masked
  source head and source BN running statistics (Axiom I.3), source retention is
  invariant across all fine-tuning epochs (Theorem I.2 & Corollary I.1).

### Running the CIFAR-100 $\to$ SVHN benchmark

```bash
python deft.py --smoke_test    # quick end-to-end check (~2 min on CPU; skips pre-sweeps, 1 scenario)
python deft.py                 # full benchmark (5 pre-sweeps + 4 trials x 5 tasks = 20 scenarios)
python deft.py --skip_presweep # full 20-scenario benchmark using locked paper hyperparameters
```

`deft.py` automatically downloads/loads `cifar100` and `svhn_cropped` via
TensorFlow Datasets, runs the hyperparameter pre-sweep on an independent tuning
task (`seed=999`, unless `--skip_presweep` or `--smoke_test` is passed), and
evaluates 4 independent trials $\times$ 5 tasks = 20 cross-domain scenarios
(20-class CIFAR-100 sampled from 4 superclasses $\to$ 10-class SVHN; see
`python deft.py --help` for all CLI flags) across DEFT and four baselines:

| Method | Target Acc (SVHN) | Source Retention (CIFAR-100) | H-Score |
|---|---:|---:|---:|
| **DEFT (Ours)** ($P^{\ast}=30$, $\eta^{\ast}=0.005$) | $89.88 \pm 0.59\%$ | $61.18 \pm 4.05\%$ | $\mathbf{72.72 \pm 2.77\%}$ |
| **Head-Only FT** | $36.01 \pm 3.06\%$ | $62.74 \pm 3.84\%$ | $45.63 \pm 2.47\%$ |
| **Standard Full FT** | $92.66 \pm 0.27\%$ | $12.61 \pm 2.78\%$ | $22.08 \pm 4.32\%$ |
| **EWC**$^{\dagger}$ ($\lambda^{\ast}=10^{7}$, $\eta^{\ast}=10^{-4}$) | $84.38 \pm 19.37\%$ | $41.98 \pm 11.49\%$ | $55.81 \pm 13.95\%$ |
| **PEFT (BN-Tuning)** | $78.99 \pm 0.98\%$ | $6.80 \pm 1.70\%$ | $12.46 \pm 2.84\%$ |

$^{\dagger}$ EWC diverged numerically (NaN at epoch 1) in 1 of the 20 scenarios,
which is counted as 0; over the other 19 scenarios it reaches $88.82 \pm 0.66\%$
target, $44.19 \pm 6.42\%$ source, and $58.75 \pm 5.68\%$ H-Score. The EWC
strength must be searched up to $\lambda = 10^{8}$ because the per-example
empirical Fisher of a converged source network is tiny (mean $\approx 10^{-8}$);
with $\lambda \leq 50$ EWC is indistinguishable from Full FT.

### DEFT math-to-code map

All code symbols below refer to [`deft.py`](deft.py) (with progressive encoding delegated to [`hope.py`](hope.py)).

| Quantity / Mechanism | Formula / Reference | Code (`deft.py`) |
|---|---|---|
| Hardware (CPU/GPU) & runtime environment auto-detection | Automatic XLA flag and device configuration | `_configure_runtime_environment` (L215–247), `detect_hardware_backend` (L271–295) |
| 8-layer BN-VGG network (`CNN_8Layer`) & quadruplet discovery | 7 sequential `(producer, producer_bn, consumer, consumer_bn)` quadruplets (Appendix H.2) | `create_cnn` (L550–646), `find_prunable_blocks` (L649–670) |
| Progressive HOPE encoding of the source network | Greedy rate-distortion tournament over all 2,944 prunable neurons (Sections 7–8, Algorithm 1) | `encode_source_curriculum` (L690–724; delegates to `hope.hope_encode`) |
| Merge-pair core set $\mathcal{M}$ and pruning-cost pool $\mathcal{C}$ | $\mathcal{C} = \{\mathcal{J}_{\mathrm{prune}}^{(i)} = N^{(i)}\lVert f_i\rVert_\circ / E_b^{(i)} \mid i \notin \mathcal{M},\, E_b^{(i)} > 10^{-5}\}$ (Section 10.2.1) | `_unpack_curriculum_entry` (L2072–2100), `generate_elasticity_map` (L2103–2245) |
| Global percentile threshold $J_P$ and binary elasticity $E_i$ | $J_P = \operatorname{Percentile}(\mathcal{C}, P)$, $E_i = \mathbf{1}[i \notin \mathcal{M} \wedge E_b^{(i)} > 10^{-5} \wedge \mathcal{J}_{\mathrm{prune}}^{(i)} < J_P]$ (Section 10.2.1, Def. I.1) | `generate_elasticity_map` (L2103–2245) |
| Structural initialization mask $\mathbf{M}$ (Axiom I.1, Theorem I.1) | $M_{u,j} = 0$ if $E_{\mathrm{in},j} > 0 \wedge E_{\mathrm{out},u} = 0$ else $1$; $\mathbf{W}_0 = \mathbf{M} \odot \mathbf{W}_{\mathrm{source}}$ | `apply_null_space_initialization` (L741–824) |
| Masked source classification head (Axioms I.1 & I.3) | $w_{\mathrm{head},j} \leftarrow \mathbf{1}[E_{\mathrm{in},j} = 0]\,w_{\mathrm{head},j}$ (bias kept) | `precompute_masked_source_head` (L2412–2468) |
| Downstream gradient elasticity $\mathbf{E}_{\mathrm{out}}$ (Axiom I.2) | $\tilde{\mathbf{g}}_t = \mathbf{E}_{\mathrm{out}} \odot \nabla_{\mathbf{W}} \mathcal{L}_{\mathrm{target}}(\mathbf{W}_t)$ | `generate_target_elasticity_list` (L2324–2375), `train_step_deft_jit` (L2251–2306) |
| Source evaluation view (Axiom I.3, Theorem I.2, Corollary I.1) | Masked source head + original source BN running statistics $(\mu_{\mathrm{src}}, \sigma_{\mathrm{src}}^2)$ | `get_source_eval_weights_fast` (L2471–2511) |
| Exact per-example diagonal Fisher (EWC baseline, Appendix H.4) | $\mathrm{FIM}_i = \mathbb{E}_{x \sim \mathcal{X}_{\mathrm{source\_train}}}[(\partial \mathcal{L}_{\mathrm{CE}} / \partial \theta_i)^2]$ | `_batch_fisher_jit` (L846–871), `compute_diagonal_fim` (L874–904), `train_step_ewc_jit` (L910–955) |
| Hyperparameter pre-sweep, CLI flags, and 20-scenario benchmark (Appendix H.5, Alg. 3) | $\mathrm{H\text{-}Score} = \frac{2\,\mathrm{Acc}_{\mathrm{val}}^{\mathrm{target}}\,\mathrm{Acc}_{\mathrm{val}}^{\mathrm{source}}}{\mathrm{Acc}_{\mathrm{val}}^{\mathrm{target}} + \mathrm{Acc}_{\mathrm{val}}^{\mathrm{source}}}$ | `find_optimal_deft_hyperparams` (L978–1227), `parse_args` (L2541–2741), `run_deft_benchmark` (L2744–3738) |

## Implementation notes and limitations

- **ResNet-50 at 224×224, Keras naming.** Quadruplets are found from Keras layer
  names (`X_1_conv`, `X_1_bn`, `X_2_conv`, …, matched by substring). The
  zero-padding tap fractions $\chi_t$ of a 3×3 consumer need its feature-map
  size, which is inferred from the first five characters of its layer name
  (`RESNET50_FEATURE_MAP_SIZE`: conv2 → 56, conv3 → 28, conv4 → 14, conv5 → 7,
  any other prefix → 14), i.e. for 224×224 inputs. Only 1×1 consumers and 3×3
  consumers with stride 1 and padding 1 are supported. Other architectures or
  resolutions need these two parts adapted.
- **What is compressed.** Only the bottleneck blocks. The stem, the shortcut
  projections and the classifier are never modified (they still count in the
  density). Convolution biases are not folded into the BN-folded weights; all
  bottleneck convolution biases of Keras' pretrained ResNet-50 are zero.
- **BN epsilon.** BN folding uses $\epsilon = 10^{-5}$ (`BN_EPSILON`), while the
  BN layers of Keras' ResNet-50 use $1.001 \times 10^{-5}$. The value is kept
  for exact reproducibility of the reference results; the encoder and the
  decoder use the same value.
- **Decoding is lossless up to float32 round-off, not bit-exact.** All tensors
  are restored bit-exactly except the producer filters of pruned and merged
  neurons, which are recovered by inverting the BN folding,
  $w_{\mathrm{raw}}=w_{\mathrm{eff}}/(\gamma/\sqrt{\sigma^2+\epsilon}+10^{-9})$.
  On ResNet-50 the maximum absolute error of the fully decoded network is
  5.96e-08 (the runner's check fails above 1e-5). The inversion cannot recover
  the filter of a neuron whose BN scale $\gamma/\sqrt{\sigma^2+\epsilon}$ is
  zero or not large compared with $10^{-9}$.
- **Reproducibility.** The encoder is deterministic, and its result depends
  neither on the evaluation callback nor on the number of BLAS threads. The
  final compressed ResNet-50 and its decoded network are bit-identical with
  NumPy 2.4.1 / SciPy 1.17.0 and NumPy 2.5.1 / SciPy 1.18.0; other BLAS/LAPACK
  builds may round differently in the last bits.
- **Software gauge of the parent neuron.** The parent's input/output scaling
  puts the whole factor $K_{\mathrm{self}}^{-1/2}$ into the outgoing weights,
  $\tilde w_{\mathrm{in}} = \sqrt{s^{\ast} R_F}\, u_c$ and
  $w_{\mathrm{out}} = \sqrt{s^{\ast}/R_F}\, K_{\mathrm{self}}^{-1/2} v^{\ast}$,
  instead of splitting it symmetrically as $K_{\mathrm{self}}^{-1/4}$ on both
  factors. The operator and its capacity are the same; only the ratio of the
  input and output weight norms differs.
- **Other details** in which the code differs from the paper's idealized
  formulas (least-squares span coefficients for the parent's BN statistics,
  numerical guards, and parameter footprints counted from tensor shapes) are
  documented where they occur in `hope.py`.
- **Run time and memory** (shared 64-core CPU server, no GPU). Encoding
  ResNet-50 with 4 BLAS threads took 7 to 10 minutes in our runs (up to 13
  minutes when the machine was heavily loaded), and the encoder itself needs
  less than 1 GB of memory; decoding the whole curriculum takes about 0.1 s. One
  evaluation on 10,144 images (batch 64) took 150–275 s with the TensorFlow
  backend, 200–220 s with JAX and 445–520 s with PyTorch, so the default run
  with 308 evaluations takes roughly 13 to 24 hours on such a CPU with
  TensorFlow or JAX. Evaluation dominates: use a larger `--eval_stride` or an
  accelerator for faster runs.
- Keras 3.13 prints a harmless `FutureWarning` about `np.object` when it is
  imported with NumPy ≥ 2.4.

## Citation

Hossein Mobahi and Peter L. Bartlett. *Hilbert Operator for Progressive Encoding
(HOPE): A Mathematical Framework for Deconstructing Learned Representations in
Deep Networks.* arXiv preprint
[arXiv:2607.21366](https://arxiv.org/abs/2607.21366), 2026.

```tex
@article{mobahi2026hope,
  title={Hilbert Operator for Progressive Encoding ({HOPE}): A Mathematical Framework for Deconstructing Learned Representations in Deep Networks},
  author={Mobahi, Hossein and Bartlett, Peter L.},
  journal={arXiv preprint arXiv:2607.21366},
  eprint={2607.21366},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.21366},
  year={2026}
}
```

## License and disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
