# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    StateDictOptions,
)

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.optimizers.disco import DiSCO
from torchtitan.optimizers.radial_helper import (
    RADIAL_METRIC_NAMES,
    calculate_aus_correction,
    resolve_aus_geometry,
)


class _DDPOnlyParallelDims:
    fsdp_enabled = False
    ep_enabled = False
    dp_replicate_enabled = False
    tp_enabled = False
    world_mesh = "unit-test"

    def get_optional_mesh(self, _name):
        return None


def _make_disco(
    params: list[torch.nn.Parameter],
    names: list[str],
    *,
    norm_factor: str,
    lr: float = 0.2,
    weight_decay: float = 0.1,
) -> DiSCO:
    param_groups = [{"params": params, "param_names": names}]
    with patch("torchtitan.optimizers.disco.dist.get_rank", return_value=0):
        optimizer = DiSCO(
            param_groups,
            is_light=False,
            weight_decay=weight_decay,
            lr=lr,
            momentum=0.0,
            nesterov=False,
            eps=1e-8,
            norm_factor=norm_factor,
            backend="identity",
            backend_steps=1,
            parallel_dims=_DDPOnlyParallelDims(),
            aus_enabled=True,
        )
    # These integration tests target scheduling/correction/application, not
    # the LMO backend itself. Identity gives a known raw update direction.
    optimizer.lmo = lambda g, **_kwargs: g
    return optimizer


class TestAUSGeometry(unittest.TestCase):
    def setUp(self):
        self.weight = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        self.update = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def test_master_identity_for_supported_geometries(self):
        aus = 0.5
        for geometry in ("rms_to_rms", "rms_to_inf", "l1_to_rms"):
            with self.subTest(geometry=geometry):
                result = calculate_aus_correction(
                    self.weight, self.update, geometry
                )
                self.assertTrue(result["valid"].item())
                eta = aus * result["correction"]
                achieved_first_order_aus = (
                    eta * result["tangent_norm"] / result["radius"]
                )
                torch.testing.assert_close(
                    achieved_first_order_aus, torch.tensor(aus)
                )

    def test_radius_and_phi_include_geometry_scaling(self):
        weight = torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        update = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
        expected = {
            "rms_to_rms": (3.0 * math.sqrt(3.0 / 2.0), math.sqrt(3.0 / 2.0)),
            "rms_to_inf": (3.0 * math.sqrt(3.0), math.sqrt(3.0)),
            "l1_to_rms": (3.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        }
        for geometry, (radius, phi_update) in expected.items():
            with self.subTest(geometry=geometry):
                result = calculate_aus_correction(weight, update, geometry)
                torch.testing.assert_close(result["radius"], torch.tensor(radius))
                torch.testing.assert_close(
                    result["phi_update"], torch.tensor(phi_update)
                )

    def test_update_scaling_preserves_applied_displacement(self):
        result = calculate_aus_correction(
            self.weight, self.update, "rms_to_inf"
        )
        scaled = calculate_aus_correction(
            self.weight, 7.0 * self.update, "rms_to_inf"
        )
        torch.testing.assert_close(
            result["correction"] * self.update,
            scaled["correction"] * (7.0 * self.update),
        )

    def test_pure_radial_update_uses_nominal_fallback(self):
        result = calculate_aus_correction(
            self.weight, 2.0 * self.weight, "rms_to_rms"
        )
        self.assertFalse(result["valid"].item())
        torch.testing.assert_close(result["correction"], torch.tensor(1.0))

    def test_large_radius_degenerate_updates_use_finite_fallback(self):
        weight = torch.diag(torch.tensor([8.0, 1.0]))
        for geometry in ("rms_to_rms", "rms_to_inf", "l1_to_rms"):
            for update in (torch.zeros_like(weight), 2.0 * weight):
                with self.subTest(geometry=geometry, update=update):
                    result = calculate_aus_correction(weight, update, geometry)
                    self.assertTrue(result["finite"].item())
                    self.assertFalse(result["valid"].item())
                    self.assertTrue(torch.isfinite(result["raw_correction"]).item())
                    torch.testing.assert_close(result["correction"], torch.tensor(1.0))

    def test_nonfinite_input_uses_nominal_fallback(self):
        update = self.update.clone()
        update[0, 0] = torch.nan
        result = calculate_aus_correction(
            self.weight, update, "rms_to_rms"
        )
        self.assertFalse(result["valid"].item())
        self.assertFalse(result["finite"].item())
        torch.testing.assert_close(result["correction"], torch.tensor(1.0))

    def test_nondifferentiable_norm_uses_nominal_fallback(self):
        result = calculate_aus_correction(
            torch.eye(2), self.update, "rms_to_rms"
        )
        self.assertFalse(result["norm_differentiable"].item())
        self.assertFalse(result["valid"].item())
        torch.testing.assert_close(result["correction"], torch.tensor(1.0))

    def test_embedding_transpose_matches_logical_matrix(self):
        stored_weight = torch.tensor(
            [[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        stored_update = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]
        )
        transposed = calculate_aus_correction(
            stored_weight,
            stored_update,
            "l1_to_rms",
            transpose=True,
        )
        logical = calculate_aus_correction(
            stored_weight.T.contiguous(),
            stored_update.T.contiguous(),
            "l1_to_rms",
        )
        for name in ("correction", "radius", "phi_update", "tangent_norm"):
            torch.testing.assert_close(transposed[name], logical[name])

    def test_norm_factor_mapping(self):
        self.assertEqual(resolve_aus_geometry("spectral"), ("rms_to_rms", False))
        self.assertEqual(
            resolve_aus_geometry("rmnp_row_norm"), ("rms_to_inf", False)
        )
        self.assertEqual(
            resolve_aus_geometry("embed_sqrt"), ("l1_to_rms", True)
        )
        with self.assertRaisesRegex(ValueError, "does not have a geometry"):
            resolve_aus_geometry("sign")


class TestAUSDiSCODDP(unittest.TestCase):
    def test_same_group_matrices_get_distinct_eta_and_correct_weight_decay(self):
        p1 = torch.nn.Parameter(
            torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        )
        p2 = torch.nn.Parameter(
            torch.tensor([[2.0, 0.0], [0.0, 0.5]])
        )
        optimizer = _make_disco(
            [p1, p2], ["p1", "p2"], norm_factor="rmnp_row_norm"
        )
        self.assertEqual(optimizer.communication_dtype, torch.float32)

        u1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u2 = torch.tensor([[0.5, 1.0], [1.0, -1.0]])
        w1 = p1.detach().clone()
        w2 = p2.detach().clone()
        p1.grad = u1.clone()
        p2.grad = u2.clone()

        c1 = calculate_aus_correction(w1, u1, "rms_to_inf")["correction"]
        c2 = calculate_aus_correction(w2, u2, "rms_to_inf")["correction"]
        self.assertFalse(torch.isclose(c1, c2).item())

        def cpu_calculate_norm(matrix, norms_to_log, transpose=False):
            matrix = matrix.detach().float()
            if transpose:
                matrix = matrix.T
            return {
                norms_to_log[0]: torch.linalg.vector_norm(matrix),
                "spectrum": torch.linalg.svdvals(matrix),
            }

        def zero_radial_metrics(weight, *_args, **_kwargs):
            return {
                name: torch.zeros((), dtype=torch.float32, device=weight.device)
                for name in RADIAL_METRIC_NAMES
            }

        optimizer.calculate_norm_at_next_step(["rms_to_inf"], gram_level=0)
        with (
            patch(
                "torchtitan.optimizers.disco.calculate_norm",
                side_effect=cpu_calculate_norm,
            ),
            patch(
                "torchtitan.optimizers.disco.calculate_radial_metrics",
                side_effect=zero_radial_metrics,
            ),
        ):
            optimizer.step()

        expected1 = w1 - 0.2 * c1 * (u1 + 0.1 * w1)
        expected2 = w2 - 0.2 * c2 * (u2 + 0.1 * w2)
        torch.testing.assert_close(p1.detach(), expected1)
        torch.testing.assert_close(p2.detach(), expected2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.2)

        metrics = optimizer.get_norms_at_current_step()
        torch.testing.assert_close(metrics["track_aus_eta/p1"], 0.2 * c1)
        torch.testing.assert_close(metrics["track_aus_eta/p2"], 0.2 * c2)
        torch.testing.assert_close(metrics["track_aus_correction/p1"], c1)
        torch.testing.assert_close(metrics["track_aus_correction/p2"], c2)
        torch.testing.assert_close(metrics["track_aus_valid/p1"], torch.tensor(1.0))
        torch.testing.assert_close(metrics["track_aus_valid/p2"], torch.tensor(1.0))

    def test_scheduler_and_correction_recompute_each_step(self):
        p = torch.nn.Parameter(
            torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        )
        optimizer = _make_disco(
            [p],
            ["p"],
            norm_factor="rmnp_row_norm",
            lr=0.01,
            weight_decay=0.0,
        )
        scheduler = LRSchedulersContainer.Config(schedule_type="aus").build(
            optimizers=[optimizer], training_steps=2
        )

        updates = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[0.5, -1.0], [1.0, 0.25]]),
        )
        corrections = []
        for update_number, update in enumerate(updates, start=1):
            weight = p.detach().clone()
            correction = calculate_aus_correction(
                weight, update, "rms_to_inf"
            )["correction"]
            corrections.append(correction)
            nominal_aus = 0.5 / math.sqrt(update_number)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], nominal_aus)

            p.grad = update.clone()
            optimizer.step()
            torch.testing.assert_close(
                p.detach(), weight - nominal_aus * correction * update
            )
            scheduler.step()

        self.assertFalse(torch.isclose(corrections[0], corrections[1]).item())

    def test_nonfinite_matrix_update_fails_before_weight_application(self):
        p = torch.nn.Parameter(
            torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        )
        optimizer = _make_disco(
            [p], ["p"], norm_factor="rmnp_row_norm", weight_decay=0.0
        )
        before = p.detach().clone()
        p.grad = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])

        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            optimizer.step()
        torch.testing.assert_close(p.detach(), before, equal_nan=True)

    def test_degenerate_update_applies_nominal_lr_and_weight_decay(self):
        for update_scale in (0.0, 2.0):
            with self.subTest(update_scale=update_scale):
                weight = torch.diag(torch.tensor([8.0, 1.0]))
                p = torch.nn.Parameter(weight.clone())
                optimizer = _make_disco([p], ["p"], norm_factor="rmnp_row_norm")
                p.grad = update_scale * weight
                optimizer.step()
                torch.testing.assert_close(
                    p.detach(), weight - 0.2 * (update_scale * weight + 0.1 * weight)
                )

    def test_nonfinite_eta_fails_before_weight_application(self):
        p = torch.nn.Parameter(torch.diag(torch.tensor([8.0, 1.0])))
        optimizer = _make_disco(
            [p], ["p"], norm_factor="rmnp_row_norm", lr=1e38, weight_decay=0.0
        )
        before = p.detach().clone()
        p.grad = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            optimizer.step()
        torch.testing.assert_close(p.detach(), before)

    def test_mixed_embedding_and_dense_validation_precedes_all_weight_updates(self):
        # Four embeddings exercise both canonical and batched-extra paths.
        for invalid in (None, "embedding", "dense"):
            with self.subTest(invalid=invalid):
                embeddings = [
                    torch.nn.Parameter(
                        torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
                    )
                    for _ in range(4)
                ]
                dense = torch.nn.Parameter(torch.diag(torch.tensor([3.0, 1.0])))
                groups = [
                    {
                        "params": embeddings,
                        "param_names": [f"embedding_{i}" for i in range(4)],
                        "norm_factor": "embed_sqrt",
                    },
                    {
                        "params": [dense],
                        "param_names": ["dense"],
                        "norm_factor": "rmnp_row_norm",
                    },
                ]
                with patch("torchtitan.optimizers.disco.dist.get_rank", return_value=0):
                    optimizer = DiSCO(
                        groups,
                        is_light=False,
                        weight_decay=0.1,
                        lr=0.2,
                        momentum=0.0,
                        nesterov=False,
                        eps=1e-8,
                        norm_factor="rmnp_row_norm",
                        backend="identity",
                        backend_steps=1,
                        parallel_dims=_DDPOnlyParallelDims(),
                        aus_enabled=True,
                    )
                optimizer.lmo = lambda g, **_kwargs: g
                for p in embeddings:
                    p.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
                dense.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                if invalid == "embedding":
                    embeddings[-1].grad[0, 0] = torch.nan
                elif invalid == "dense":
                    dense.grad[0, 0] = torch.nan
                params = [*embeddings, dense]
                before = [p.detach().clone() for p in params]
                if invalid is not None:
                    with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                        optimizer.step()
                    for p, weight in zip(params, before):
                        torch.testing.assert_close(p.detach(), weight)
                else:
                    expected = []
                    for p, weight in zip(params, before):
                        is_embedding = p is not dense
                        correction = calculate_aus_correction(
                            weight,
                            p.grad,
                            "l1_to_rms" if is_embedding else "rms_to_inf",
                            transpose=is_embedding,
                        )["correction"]
                        expected.append(
                            weight - 0.2 * correction * (p.grad + 0.1 * weight)
                        )
                    optimizer.step()
                    for p, weight in zip(params, expected):
                        torch.testing.assert_close(p.detach(), weight)

    def test_embedding_path_uses_logical_l1_to_rms_geometry(self):
        p = torch.nn.Parameter(
            torch.tensor([[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        optimizer = _make_disco(
            [p],
            ["tok_embeddings.weight"],
            norm_factor="embed_sqrt",
        )
        update = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]
        )
        weight = p.detach().clone()
        p.grad = update.clone()
        correction = calculate_aus_correction(
            weight,
            update,
            "l1_to_rms",
            transpose=True,
        )["correction"]

        optimizer.step()

        expected = weight - 0.2 * correction * (update + 0.1 * weight)
        torch.testing.assert_close(p.detach(), expected)
        self.assertEqual(len(optimizer.embed_params), 1)
        self.assertEqual(len(optimizer.ddp_params), 0)


class TestAUSCheckpointCPU(unittest.TestCase):
    """Exercise in-memory checkpoint state only; no process group or DCP I/O."""

    def _make_container(
        self, *, aus_enabled, aus_coefficient=0.5, group_coefficient=None
    ):
        model = torch.nn.Linear(2, 2, bias=False)
        split_rules = []
        if group_coefficient is not None:
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.Linear(2, 2, bias=False),
            )
            split_rules = [
                {"str_match": r"^1\.weight$", "aus_coefficient": group_coefficient}
            ]
        config = OptimizersContainer.Config(
            name="DiSCO",
            aus_enabled=aus_enabled,
            aus_coefficient=aus_coefficient,
            extra_param_group_split_rules=split_rules,
            norm_factor="rmnp_row_norm",
            zeropower_backend="identity",
            lr=0.01,
            weight_decay=0.0,
            enable_spectrum_plot=False,
            enable_spectrum_export=False,
        )
        with patch("torchtitan.optimizers.disco.dist.get_rank", return_value=0):
            container = config.build(
                model_parts=[model], parallel_dims=_DDPOnlyParallelDims()
            )
        container.optimizers[0].lmo = lambda g, **_kwargs: g
        return model, container

    def test_pre_aus_checkpoint_schema_and_optimizer_state_restore(self):
        old_model, old = self._make_container(aus_enabled=False)
        old_optimizer = old.optimizers[0]
        old_optimizer.state[old_model.weight]["momentum_buffer"].fill_(3.0)
        old_optimizer.state[old_model.weight]["radial_state"]["raw_A2"].fill_(7.0)
        # Reconstruct the exact pre-AUS schema from the unfiltered torch
        # optimizer state; strict DCP planning requires these keys to match.
        legacy = copy.deepcopy(
            get_optimizer_state_dict(
                old_model,
                old_optimizer,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
        )
        del legacy["param_groups.weight.aus_enabled"]
        del legacy["param_groups.weight.aus_coefficient"]
        for enabled in (False, True):
            with self.subTest(aus_enabled=enabled):
                model, container = self._make_container(aus_enabled=enabled)
                self.assertEqual(set(container.state_dict()), set(legacy))
                container.load_state_dict(legacy)
                optimizer = container.optimizers[0]
                self.assertEqual(optimizer.param_groups[0]["aus_enabled"], enabled)
                self.assertEqual(optimizer.param_groups[0]["aus_coefficient"], 0.5)
                torch.testing.assert_close(
                    optimizer.state[model.weight]["momentum_buffer"],
                    torch.full_like(model.weight, 3.0),
                )
                torch.testing.assert_close(
                    optimizer.state[model.weight]["radial_state"]["raw_A2"],
                    torch.tensor(7.0),
                )
        self.assertNotIn("param_groups.weight.aus_enabled", legacy)

    def test_adam_checkpoint_state_is_unchanged(self):
        model = torch.nn.Linear(2, 2, bias=False)
        config = OptimizersContainer.Config(name="Adam", implementation="for-loop")
        container = config.build(
            model_parts=[model], parallel_dims=_DDPOnlyParallelDims()
        )
        model.weight.grad = torch.ones_like(model.weight)
        container.step()
        saved = copy.deepcopy(container.state_dict())
        unfiltered = get_optimizer_state_dict(
            model,
            container.optimizers[0],
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        torch.testing.assert_close(saved, unfiltered)
        # A resume starts with a fresh model and no outstanding gradients,
        # which lets PyTorch initialize Adam's lazy state before loading.
        restored_model = torch.nn.Linear(2, 2, bias=False)
        restored = config.build(
            model_parts=[restored_model], parallel_dims=_DDPOnlyParallelDims()
        )
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.state_dict(), saved)

    def test_checkpoint_aus_settings_do_not_override_run_config(self):
        _, container = self._make_container(aus_enabled=True)
        state = copy.deepcopy(container.state_dict())
        state["param_groups.weight.aus_enabled"] = False
        state["param_groups.weight.aus_coefficient"] = 0.1
        container.load_state_dict(state)
        group = container.optimizers[0].param_groups[0]
        self.assertTrue(group["aus_enabled"])
        self.assertEqual(group["aus_coefficient"], 0.5)
        self.assertFalse(state["param_groups.weight.aus_enabled"])
        self.assertEqual(state["param_groups.weight.aus_coefficient"], 0.1)

    def test_direct_optimizer_restore_preserves_group_coefficient(self):
        _, old = self._make_container(aus_enabled=True, aus_coefficient=0.1)
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                state = copy.deepcopy(old.optimizers[0].state_dict())
                if legacy:
                    del state["param_groups"][0]["aus_coefficient"]
                _, new = self._make_container(aus_enabled=True, aus_coefficient=0.7)
                new.optimizers[0].load_state_dict(state)
                self.assertEqual(new.optimizers[0].param_groups[0]["aus_coefficient"], 0.7)

    def test_resume_uses_configured_aus_at_saved_step(self):
        for source_schedule in ("wsd", "aus"):
            for reconfigure_lrs in (False, True):
                with self.subTest(source=source_schedule, reconfigure=reconfigure_lrs):
                    _, old = self._make_container(
                        aus_enabled=source_schedule == "aus",
                        aus_coefficient=0.1,
                        group_coefficient=0.2,
                    )
                    old_scheduler = LRSchedulersContainer.Config(
                        schedule_type=source_schedule,
                        warmup_steps=0,
                    ).build(optimizers=old, training_steps=100)
                    for _ in range(3):
                        old.step()
                        old_scheduler.step()
                    optimizer_state = copy.deepcopy(old.state_dict())
                    scheduler_state = copy.deepcopy(old_scheduler.state_dict())

                    model, new = self._make_container(
                        aus_enabled=True, group_coefficient=0.8
                    )
                    scheduler = LRSchedulersContainer.Config(schedule_type="aus").build(
                        optimizers=new, training_steps=100
                    )
                    new.preserve_lrs_when_loading = reconfigure_lrs
                    scheduler.preserve_lrs_when_loading = reconfigure_lrs
                    new.load_state_dict(optimizer_state)
                    scheduler.load_state_dict(scheduler_state)
                    optimizer = new.optimizers[0]
                    self.assertEqual(scheduler.schedulers[0].last_epoch, 3)
                    self.assertEqual(scheduler.schedulers[0].base_lrs, [0.5, 0.8])
                    self.assertEqual(scheduler.schedulers[0].get_last_lr(), [0.25, 0.4])

                    # Check each group's first actual resumed displacement,
                    # before the scheduler advances again in the training loop.
                    expected_weights = []
                    for group, coefficient in zip(optimizer.param_groups, (0.5, 0.8)):
                        self.assertEqual(group["aus_coefficient"], coefficient)
                        self.assertEqual(group["initial_lr"], coefficient)
                        self.assertEqual(group["lr"], coefficient / 2)
                        weight = group["params"][0]
                        with torch.no_grad():
                            weight.copy_(torch.diag(torch.tensor([3.0, 1.0])))
                        before = weight.detach().clone()
                        weight.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                        group["momentum"] = 0.0
                        correction = calculate_aus_correction(
                            before, weight.grad, "rms_to_inf"
                        )["correction"]
                        expected_weights.append(
                            before - coefficient / 2 * correction * weight.grad
                        )
                    new.step()
                    for weight, expected in zip(model.parameters(), expected_weights):
                        torch.testing.assert_close(weight.detach(), expected)
                    scheduler.step()
                    for group, coefficient in zip(optimizer.param_groups, (0.5, 0.8)):
                        self.assertAlmostEqual(group["lr"], coefficient / math.sqrt(5))


if __name__ == "__main__":
    unittest.main()
