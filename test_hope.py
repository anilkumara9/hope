# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""HOPE (Hilbert Operator for Progressive Encoding): unit and integration test suite.

Fast self-test of the public API of ``hope`` on a tiny synthetic ResNet-style
network.
It needs only NumPy and SciPy (no Keras, no ImageNet) and runs in a few seconds:

    python test_hope.py
    python -m pytest test_hope.py

Checked: quadruplet discovery from weight paths and from layer names; encoding
to completion with every
action type (PRUNE, MERGE, EVICT) exercised; the input weights are not modified;
the density history is
monotone and consistent with ``eval_fn`` / ``eval_stride``; encoding is
deterministic and independent of
``eval_fn``; full decoding is lossless; progressive decoding retraces every
intermediate encoder state;
``hope_decode_step`` works in place; the curriculum survives a pickle round
trip.
"""

import functools
import hashlib
import itertools
import os
import pickle
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Tuple

import hope
import hope_baselines
import numpy as np
import plot_compression_curves

WeightsDict = Dict[str, np.ndarray]

NUM_BLOCKS = (
    2  # Bottleneck blocks conv2_block1, conv2_block2 (two quadruplets each).
)
WIDTH = 16  # Block input/output channels.
MID = 8  # Bottleneck channels (neurons of the quadruplets).
ACTIONS = ("PRUNE", "MERGE", "EVICT")
# Producer kernels are restored as (k * s) / s in float32, so they may differ
# from the original by a few ulps; every other tensor must be restored
# bit-exactly.
RTOL, ATOL = 1e-6, 1e-7

EXPECTED_QUADRUPLETS = [
    (
        f"conv2_block{b}_{i}_conv",
        f"conv2_block{b}_{i}_bn",
        f"conv2_block{b}_{i + 1}_conv",
        f"conv2_block{b}_{i + 1}_bn",
    )
    for b, i in itertools.product(range(1, NUM_BLOCKS + 1), (1, 2))
]


def make_synthetic_resnet_weights(seed: int = 0) -> WeightsDict:
  """Builds a tiny Keras-style ResNet weights dictionary.

  Tensor names, shapes (HWIO kernels), conv biases and ordering mimic
  ``keras.applications.ResNet50``:
  ``conv2_blockK_{1,2,3}_conv/{kernel,bias}``,
  ``conv2_blockK_{1,2,3}_bn/{gamma,beta,moving_mean,moving_variance}``, plus
  the projection shortcut ``conv2_block1_0_*`` of the first block. Neurons 0 and
  1 of ``conv2_block1_1_conv`` are made near-duplicates so that the encoder also
  performs a MERGE.

  Args:
    seed: Random seed for deterministic synthetic weight generation.

  Returns:
    Dictionary mapping Keras-style weight paths to float32 NumPy arrays.
  """
  rs = np.random.RandomState(
      seed
  )  # Legacy generator: its stream is frozen across NumPy versions.
  w: WeightsDict = {}

  def conv(name: str, k: int, c_in: int, c_out: int) -> None:
    w[f"{name}/kernel"] = (
        rs.standard_normal((k, k, c_in, c_out)) * np.sqrt(2.0 / (k * k * c_in))
    ).astype(np.float32)
    w[f"{name}/bias"] = (0.01 * rs.standard_normal(c_out)).astype(np.float32)

  def bn(name: str, c: int) -> None:
    w[f"{name}/gamma"] = rs.uniform(0.5, 1.5, c).astype(np.float32)
    w[f"{name}/beta"] = (0.5 * rs.standard_normal(c)).astype(np.float32)
    w[f"{name}/moving_mean"] = (0.5 * rs.standard_normal(c)).astype(np.float32)
    w[f"{name}/moving_variance"] = rs.uniform(0.5, 2.0, c).astype(np.float32)

  conv("conv1_conv", 7, 3, WIDTH)
  bn("conv1_bn", WIDTH)
  for b in range(1, NUM_BLOCKS + 1):
    p = f"conv2_block{b}"
    conv(f"{p}_1_conv", 1, WIDTH, MID)
    bn(f"{p}_1_bn", MID)
    conv(f"{p}_2_conv", 3, MID, MID)
    bn(f"{p}_2_bn", MID)
    if b == 1:
      conv(f"{p}_0_conv", 1, WIDTH, WIDTH)
    conv(f"{p}_3_conv", 1, MID, WIDTH)
    if b == 1:
      bn(f"{p}_0_bn", WIDTH)
    bn(f"{p}_3_bn", WIDTH)
  w["predictions/kernel"] = (0.1 * rs.standard_normal((WIDTH, 10))).astype(
      np.float32
  )
  w["predictions/bias"] = np.zeros(10, np.float32)

  k1, k2 = w["conv2_block1_1_conv/kernel"], w["conv2_block1_2_conv/kernel"]
  k1[..., 1] = 1.01 * k1[..., 0]
  k2[:, :, 1, :] = 0.99 * k2[:, :, 0, :]
  for param in ("gamma", "beta", "moving_mean", "moving_variance"):
    w[f"conv2_block1_1_bn/{param}"][1] = w[f"conv2_block1_1_bn/{param}"][0]
  return w


def _hash(weights: WeightsDict) -> str:
  h = hashlib.sha256()
  for k in sorted(weights):
    v = np.ascontiguousarray(weights[k])
    h.update(k.encode())
    h.update(str(v.dtype).encode())
    h.update(str(v.shape).encode())
    h.update(v.tobytes())
  return h.hexdigest()


def _copy(weights: WeightsDict) -> WeightsDict:
  return {k: v.copy() for k, v in weights.items()}


def _nonzeros(weights: WeightsDict) -> int:
  return int(sum(np.count_nonzero(v) for v in weights.values()))


def _is_producer_kernel(key: str) -> bool:
  return key.endswith("_1_conv/kernel") or key.endswith("_2_conv/kernel")


def _assert_restored(
    restored: WeightsDict,
    reference: WeightsDict,
    what: str,
    producer_kernels: Tuple[str, ...] = (),
) -> None:
  """Asserts that ``restored`` matches ``reference`` within float32 round-off."""
  assert set(restored) == set(reference), f"{what}: different tensor names"
  for k, ref in reference.items():
    got = restored[k]
    assert (
        got.dtype == ref.dtype and got.shape == ref.shape
    ), f"{what}: {k} dtype/shape changed"
    assert np.count_nonzero(got) == np.count_nonzero(
        ref
    ), f"{what}: {k} sparsity pattern differs"
    if _is_producer_kernel(k) or k in producer_kernels:
      np.testing.assert_allclose(
          got, ref, rtol=RTOL, atol=ATOL, err_msg=f"{what}: {k}"
      )
    else:
      assert np.array_equal(got, ref), f"{what}: {k} not restored bit-exactly"


@functools.cache
def _encoded() -> Tuple[
    WeightsDict,
    List[Any],
    WeightsDict,
    List[Any],
    List[Tuple[float, float]],
    List[WeightsDict],
]:
  """Encodes the synthetic network once, recording the encoder state after every step (eval_stride=1)."""
  weights = make_synthetic_resnet_weights()
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  states: List[WeightsDict] = []

  def record(current: WeightsDict) -> float:
    states.append(_copy(current))
    return float(len(states))

  final, curriculum, history = hope.hope_encode(
      weights, quadruplets, eval_fn=record, eval_stride=1
  )
  return weights, quadruplets, final, curriculum, history, states


# ----------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------


def test_find_bottleneck_quadruplets_from_weight_paths():
  """Tests quadruplet discovery from Keras weight paths."""
  weights = make_synthetic_resnet_weights()
  got = [tuple(q) for q in hope.find_bottleneck_quadruplets(list(weights))]
  assert got == EXPECTED_QUADRUPLETS, got


def test_find_bottleneck_quadruplets_from_layer_names():
  """Tests quadruplet discovery from bare layer names and partial lists."""
  layer_names = [
      "input_layer",
      "conv1_pad",
      "conv1_conv",
      "conv1_bn",
      "conv1_relu",
      "pool1_pad",
      "pool1_pool",
  ]
  for b in range(1, NUM_BLOCKS + 1):
    p = f"conv2_block{b}"
    layer_names += [
        f"{p}_1_conv",
        f"{p}_1_bn",
        f"{p}_1_relu",
        f"{p}_2_conv",
        f"{p}_2_bn",
        f"{p}_2_relu",
    ]
    layer_names += ([f"{p}_0_conv"] if b == 1 else []) + [f"{p}_3_conv"]
    layer_names += ([f"{p}_0_bn"] if b == 1 else []) + [
        f"{p}_3_bn",
        f"{p}_add",
        f"{p}_out",
    ]
  layer_names += ["avg_pool", "predictions"]
  got = [tuple(q) for q in hope.find_bottleneck_quadruplets(layer_names)]
  assert got == EXPECTED_QUADRUPLETS, got
  # Repeated names / a mix of layer names and weight paths give the same result.
  mixed = layer_names[:10] + [n + "/kernel" for n in layer_names] + layer_names
  assert [
      tuple(q) for q in hope.find_bottleneck_quadruplets(mixed)
  ] == EXPECTED_QUADRUPLETS
  # A quadruplet is reported only if all four of its layers exist.
  partial = [n for n in layer_names if n != "conv2_block2_3_bn"]
  assert [
      tuple(q) for q in hope.find_bottleneck_quadruplets(partial)
  ] == EXPECTED_QUADRUPLETS[:-1]


def test_encode_runs_to_completion():
  """Tests that encoding consumes all bottleneck neurons and uses all actions."""
  weights, quadruplets, final, curriculum, history, states = _encoded()
  assert len(curriculum) > 0 and len(states) == len(curriculum) == len(history)
  actions = [p.action for p in curriculum]
  assert set(actions) <= set(ACTIONS), set(actions)
  assert set(actions) == set(ACTIONS), (
      "the synthetic network should exercise every action type, got"
      f" {sorted(set(actions))}"
  )
  for packet in curriculum:
    assert tuple(packet.quadruplet) in quadruplets
    assert packet.indices is not None
  # Every bottleneck neuron has been consumed: all bottleneck kernels are zero
  # at the end.
  for quad in quadruplets:
    for conv in (quad[0], quad[2]):
      assert np.count_nonzero(final[f"{conv}/kernel"]) == 0, conv
  # Tensors outside the bottleneck branches are never modified.
  for k in (
      "conv1_conv/kernel",
      "conv1_bn/gamma",
      "conv2_block1_0_conv/kernel",
      "predictions/kernel",
  ):
    assert np.array_equal(final[k], weights[k]), k


def test_encode_does_not_mutate_input():
  """Tests that ``hope_encode`` leaves the input weights dictionary unchanged."""
  weights = make_synthetic_resnet_weights()
  before = _hash(weights)
  hope.hope_encode(weights, hope.find_bottleneck_quadruplets(list(weights)))
  assert _hash(weights) == before


def test_density_history_is_monotone_and_consistent():
  """Tests monotone non-increasing density history and callback values."""
  weights, _, final, _, history, states = _encoded()
  total = _nonzeros(weights)
  densities = [float(d) for d, _ in history]
  assert all(b <= a for a, b in zip(densities, densities[1:])), densities
  assert 0.0 < densities[-1] < densities[0] < 1.0, densities
  # The history reports density = nonzeros / nonzeros(pretrained) and the value
  # returned by eval_fn.
  for t, ((density, value), state) in enumerate(zip(history, states), 1):
    assert density == _nonzeros(state) / float(total)
    assert float(value) == float(t)
  assert densities[-1] == _nonzeros(final) / float(total)


def test_eval_stride():
  """Tests ``eval_stride`` subsampling and callback-independent determinism."""
  weights = make_synthetic_resnet_weights()
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  _, _, ref_final, ref_curriculum, ref_history, _ = _encoded()
  calls: List[int] = []

  def counter(current: WeightsDict) -> float:
    calls.append(_nonzeros(current))
    return 0.5

  stride = 3
  final, curriculum, history = hope.hope_encode(
      weights, quadruplets, eval_fn=counter, eval_stride=stride
  )
  assert len(calls) == len(curriculum) // stride == len(history)
  assert [v for _, v in history] == [0.5] * len(history)
  assert [d for d, _ in history] == [d for d, _ in ref_history][
      stride - 1 :: stride
  ][: len(history)]
  # The evaluation callback has no influence on the encoding (deterministic,
  # data-free).
  assert _hash(final) == _hash(ref_final)
  assert [p.action for p in curriculum] == [p.action for p in ref_curriculum]
  final_no_eval, curriculum_no_eval, _ = hope.hope_encode(weights, quadruplets)
  assert _hash(final_no_eval) == _hash(ref_final)
  assert [list(p.indices) for p in curriculum_no_eval] == [
      list(p.indices) for p in ref_curriculum
  ]


def test_lossless_decode():
  """Tests full lossless reconstruction of pretrained weights via ``hope_decode``."""
  weights, _, final, curriculum, _, _ = _encoded()
  final_hash = _hash(final)
  decoded = hope.hope_decode(final, curriculum)
  assert _hash(final) == final_hash, "hope_decode must not modify its input"
  _assert_restored(decoded, weights, "full decode")


def test_progressive_decode_retraces_the_encoder():
  """Tests that partial decoding matches every intermediate encoder state."""
  weights, _, final, curriculum, _, states = _encoded()
  trajectory = [
      weights
  ] + states  # trajectory[t] = encoder state after t steps.
  num_steps = len(curriculum)
  for k in range(num_steps + 1):
    partial = hope.hope_decode(final, curriculum, num_steps=k)
    _assert_restored(
        partial, trajectory[num_steps - k], f"decode of {k} packets"
    )
  # hope_decode works on a copy, even when no packet is applied.
  untouched = hope.hope_decode(final, curriculum, num_steps=0)
  assert all(not np.shares_memory(untouched[k], final[k]) for k in final)


def test_decode_step_in_place():
  """Tests single-step in-place decoding via ``hope_decode_step``."""
  _, _, final, curriculum, _, states = _encoded()
  current = _copy(final)
  returned = hope.hope_decode_step(current, curriculum[0])
  assert returned is current
  _assert_restored(
      current,
      states[-2] if len(states) > 1 else make_synthetic_resnet_weights(),
      "one decode step",
  )


def test_curriculum_pickle_roundtrip():
  """Tests that pickling and unpickling a curriculum preserves exact decoding."""
  # compress_resnet50.py stores the curriculum with pickle; decoding the
  # reloaded copy must be identical.
  _, _, final, curriculum, _, _ = _encoded()
  reloaded = pickle.loads(  # pylint: disable=g-unsafe-pickle-load
      pickle.dumps(curriculum, protocol=pickle.HIGHEST_PROTOCOL)
  )
  assert _hash(hope.hope_decode(final, reloaded)) == _hash(
      hope.hope_decode(final, curriculum)
  )


def test_output_file_csv():
  """Tests incremental CSV writing in ``hope_encode``."""
  weights = make_synthetic_resnet_weights()
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = os.path.join(tmpdir, "results_HOPE.csv")
    _, curriculum, history = hope.hope_encode(
        weights,
        quadruplets,
        eval_fn=lambda _: 0.75,
        eval_stride=3,
        output_file=csv_path,
    )
    assert os.path.isfile(csv_path)
    with open(csv_path, "r") as f:
      lines = [line.strip() for line in f if line.strip()]
    assert lines[0] == "density,accuracy"
    expected_rows = 1 + len(history) + (1 if len(curriculum) % 3 != 0 else 0)
    assert len(lines) - 1 == expected_rows
    first_d, first_a = [float(x) for x in lines[1].split(",")]
    assert first_d == 1.0 and first_a == 0.75

    no_eval_dir = os.path.join(tmpdir, "no_eval")
    hope_baselines.run_all_baselines(
        weights,
        quadruplets,
        eval_fn=None,
        eval_stride=10,
        output_dir=no_eval_dir,
    )
    assert not os.listdir(no_eval_dir)
    direct_csv = os.path.join(no_eval_dir, "direct.csv")
    hope_baselines.baseline_encode(
        weights,
        quadruplets,
        eval_fn=None,
        eval_stride=10,
        output_file=direct_csv,
    )
    assert not os.path.exists(direct_csv)


def test_all_seven_baselines_encode_and_lossless_decode():
  """Tests encoding and lossless decoding across all seven baseline methods."""
  weights = make_synthetic_resnet_weights()
  before_hash = _hash(weights)
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  total = _nonzeros(weights)

  for method in hope_baselines.BASELINE_METHODS:
    final, curriculum, history = hope_baselines.baseline_encode(
        weights,
        quadruplets,
        benchmark_method=method,
        eval_fn=lambda _: 0.5,
        eval_stride=4,
        output_file="",
    )
    assert _hash(weights) == before_hash, f"{method} mutated input weights"
    assert (
        len(curriculum) == 4 * MID
    ), f"{method}: expected {4 * MID} steps, got {len(curriculum)}"
    densities = [float(d) for d, _ in history]
    assert densities[0] == 1.0
    assert densities[-1] == _nonzeros(final) / total
    assert all(
        b <= a for a, b in zip(densities, densities[1:])
    ), f"{method}: non-monotone density"
    assert _nonzeros(final) < total
    if method == "WS_COSINE_MERGE":
      assert any(
          p.action == "MERGE" for p in curriculum
      ), "WS_COSINE_MERGE should perform MERGE actions"
    decoded = hope.hope_decode(final, curriculum)
    _assert_restored(decoded, weights, f"baseline {method} full decode")


def test_plot_compression_curves():
  """Tests CSV curve loading and PNG rendering in ``plot_compression_curves``."""
  weights = make_synthetic_resnet_weights()
  quadruplets = hope.find_bottleneck_quadruplets(list(weights))
  with tempfile.TemporaryDirectory() as tmpdir:
    hope_csv = os.path.join(tmpdir, "results_HOPE.csv")
    hope.hope_encode(
        weights,
        quadruplets,
        eval_fn=lambda _: 0.8,
        eval_stride=8,
        output_file=hope_csv,
    )
    hope_baselines.run_all_baselines(
        weights,
        quadruplets,
        eval_fn=lambda _: 0.6,
        eval_stride=8,
        output_dir=tmpdir,
    )
    for fname in hope_baselines.DEFAULT_BASELINE_FILES.values():
      arr = plot_compression_curves.load_density_accuracy_file(
          os.path.join(tmpdir, fname)
      )
      assert arr is not None and arr.ndim == 2 and arr.shape[1] == 2
    png_path = os.path.join(tmpdir, "accuracy_vs_density.png")
    try:
      import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

      before_rc = dict(plt.rcParams)
    except ImportError:
      plt = None
      before_rc = None
    try:
      from PIL import Image as _pil_image  # pylint: disable=g-import-not-at-top

      has_pil = _pil_image is not None
    except ImportError:
      has_pil = False
    plot_compression_curves.plot_compression_curves(
        data_dir=tmpdir, save_path=png_path
    )
    if plt is not None and before_rc is not None:
      assert dict(plt.rcParams) == before_rc
    if plt is not None or has_pil:
      assert os.path.isfile(png_path) and os.path.getsize(png_path) > 0

    # Also verify graceful fallback when neither matplotlib nor Pillow is
    # installed (pure numpy + scipy minimal environment).
    no_lib_png = os.path.join(tmpdir, "no_libs.png")
    blocked_mods = ("matplotlib", "matplotlib.pyplot", "PIL", "PIL.Image")
    saved_mods = {m: sys.modules.get(m) for m in blocked_mods}
    saved_present = {m: m in sys.modules for m in blocked_mods}
    try:
      for m in blocked_mods:
        sys.modules[m] = None  # type: ignore[assignment]
      res = plot_compression_curves.plot_compression_curves(
          data_dir=tmpdir, save_path=no_lib_png
      )
      assert res == (None, None)
      assert not os.path.exists(no_lib_png)
    finally:
      for m in blocked_mods:
        if saved_present[m]:
          sys.modules[m] = saved_mods[m]  # type: ignore[assignment]
        else:
          sys.modules.pop(m, None)


def test_mixed_conv_dense_quadruplets_and_deft_metadata():
  """Tests 4D Conv -> 2D Dense -> 2D Dense (`bn_b=None`) encoding, decoding, and DEFT packet fields."""
  rng = np.random.default_rng(7)
  weights: WeightsDict = {
      "conv1/kernel": rng.normal(0.0, 0.2, size=(3, 3, 3, 6)).astype(
          np.float32
      ),
      "bn1/gamma": rng.uniform(0.5, 1.5, size=(6,)).astype(np.float32),
      "bn1/beta": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn1/moving_mean": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn1/moving_variance": rng.uniform(0.3, 1.8, size=(6,)).astype(
          np.float32
      ),
      "bottleneck/kernel": rng.normal(0.0, 0.2, size=(6, 6)).astype(np.float32),
      "bn_bot/gamma": rng.uniform(0.5, 1.5, size=(6,)).astype(np.float32),
      "bn_bot/beta": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn_bot/moving_mean": rng.normal(0.0, 0.3, size=(6,)).astype(np.float32),
      "bn_bot/moving_variance": rng.uniform(0.3, 1.8, size=(6,)).astype(
          np.float32
      ),
      "output/kernel": rng.normal(0.0, 0.2, size=(6, 4)).astype(np.float32),
      "output/bias": rng.normal(0.0, 0.1, size=(4,)).astype(np.float32),
  }
  # Make neurons 0 and 1 in both layers near-duplicates so MERGE triggers on
  # both 4D->2D and 2D->2D (`bn_b=None`) quadruplets.
  weights["conv1/kernel"][..., 1] = weights["conv1/kernel"][..., 0] * 1.001
  weights["bottleneck/kernel"][1, :] = (
      weights["bottleneck/kernel"][0, :] * 0.999
  )
  weights["bottleneck/kernel"][:, 1] = (
      weights["bottleneck/kernel"][:, 0] * 1.001
  )
  weights["output/kernel"][1, :] = weights["output/kernel"][0, :] * 0.999
  for bn in ("bn1", "bn_bot"):
    for p in ("gamma", "beta", "moving_mean", "moving_variance"):
      weights[f"{bn}/{p}"][1] = weights[f"{bn}/{p}"][0] * 1.001

  quadruplets: List[hope.Quadruplet] = [
      ("conv1", "bn1", "bottleneck", "bn_bot"),
      ("bottleneck", "bn_bot", "output", None),
  ]
  compressed, curriculum, _ = hope.hope_encode(
      weights,
      quadruplets,
      eval_stride=0,
      feature_map_sizes={"conv1": 32.0},
  )
  assert len(curriculum) == 12
  assert any(p.action == "MERGE" for p in curriculum)
  assert any(
      p.action == "MERGE" and p.quadruplet[3] is None for p in curriculum
  )
  for p in curriculum:
    assert p.n_active >= 1
    assert p.norm > 0.0
    assert p.e_b >= hope.CAPACITY_FLOOR
    if p.quadruplet[3] is None:
      assert p.consumer_mean_before is None
  decoded = hope.hope_decode(compressed, curriculum)
  _assert_restored(
      decoded,
      weights,
      "mixed conv/dense decode",
      producer_kernels=("conv1/kernel", "bottleneck/kernel"),
  )


def _run_all() -> int:
  """Runs all ``test_*`` functions in this module and returns an exit code."""
  tests = [
      (name, fn)
      for name, fn in globals().items()
      if name.startswith("test_") and callable(fn)
  ]
  failed = 0
  for name, fn in tests:
    t0 = time.time()
    try:
      fn()
      print(f"PASS  {name} ({time.time() - t0:.2f}s)")
    except Exception:  # pylint: disable=broad-exception-caught
      failed += 1
      print(f"FAIL  {name}")
      traceback.print_exc()
  print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(_run_all())
