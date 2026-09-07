# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import unittest
from unittest.mock import MagicMock

import torch
from torch.optim import Adam

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ConfigManager


class TestLRScheduler(unittest.TestCase):
    def setUp(self):
        # Create a simple model with parameters
        self.model = torch.nn.Linear(10, 10)
        # Create an optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.1)

        # We don't actually call `optimizer.step()` which will cause a warning
        # from PyTorch. Avoid the warnings that may confuse people.
        self.optimizer._opt_called = True

        # Create an optimizer container
        self.optimizer_container = MagicMock(spec=OptimizersContainer)
        self.optimizer_container.__iter__.return_value = iter([self.optimizer])
        self.optimizer_container.__len__.return_value = 1

    def create_trainer_config(
        self,
        training_steps=10,
        warmup_steps=None,
        decay_ratio=None,
        decay_type=None,
        min_lr_factor=None,
        schedule_type=None,
    ):
        # Create a trainer config with the specified parameters
        args = [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--training.steps",
            str(training_steps),
        ]

        args += (
            ["--lr_scheduler.schedule_type", schedule_type]
            if schedule_type is not None
            else []
        )
        args += (
            ["--lr_scheduler.warmup_steps", str(warmup_steps)]
            if warmup_steps is not None
            else []
        )
        args += (
            ["--lr_scheduler.decay_ratio", str(decay_ratio)]
            if decay_ratio is not None
            else []
        )
        args += (
            ["--lr_scheduler.decay_type", decay_type] if decay_type is not None else []
        )
        args += (
            ["--lr_scheduler.min_lr_factor", str(min_lr_factor)]
            if min_lr_factor is not None
            else []
        )

        config_manager = ConfigManager()
        # Create base config with parameters passed directly
        config = config_manager.parse_args(args)

        return config

    def test_aus_inverse_sqrt(self):
        """AUS defaults to 0.5 and bypasses WSD and the optimizer LR."""
        config = self.create_trainer_config(
            training_steps=5,
            schedule_type="aus",
        )
        self.optimizer.param_groups[0]["lr"] = 0.123
        self.optimizer.param_groups[0]["aus_enabled"] = True
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        for update_number in range(1, 6):
            expected_aus = 0.5 / torch.sqrt(torch.tensor(float(update_number)))
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_aus.item(),
                places=6,
                msg=(
                    f"Update {update_number}: expected AUS {expected_aus.item()}, "
                    f"got {self.optimizer.param_groups[0]['lr']}"
                ),
            )
            lr_scheduler.step()

    def test_aus_group_coefficients_across_optimizers_and_resume(self):
        optimizers = [
            Adam(
                [
                    {"params": [self.model.weight], "aus_coefficient": 0.2},
                    {"params": [self.model.bias]},  # Default coefficient is 0.5.
                ],
                lr=0.123,
            ),
            Adam([torch.nn.Parameter(torch.ones(1))], lr=0.9),
        ]
        optimizers[1].param_groups[0]["aus_coefficient"] = 0.8
        for optimizer in optimizers:
            optimizer._opt_called = True
            for group in optimizer.param_groups:
                group["aus_enabled"] = True
                group["initial_lr"] = 0.01  # Stale scheduler base must be ignored.
        config = LRSchedulersContainer.Config(schedule_type="aus")
        scheduler = config.build(optimizers=optimizers, training_steps=5)
        coefficients = [[0.2, 0.5], [0.8]]
        for update_number in range(1, 4):
            for optimizer, group_coefficients in zip(optimizers, coefficients):
                for group, coefficient in zip(
                    optimizer.param_groups, group_coefficients
                ):
                    self.assertAlmostEqual(
                        group["lr"], coefficient / math.sqrt(update_number)
                    )
            scheduler.step()

        saved = copy.deepcopy(scheduler.state_dict())
        original_saved = copy.deepcopy(saved)
        optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        # Reconfigure one group, then simulate optimizer state being loaded first.
        optimizers[0].param_groups[0]["aus_coefficient"] = 0.6
        coefficients[0][0] = 0.6
        resumed = config.build(optimizers=optimizers, training_steps=5)
        for optimizer, state in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(state)
        resumed.load_state_dict(saved)
        self.assertEqual(saved, original_saved)
        for child, group_coefficients in zip(resumed, coefficients):
            self.assertEqual(child.last_epoch, 3)
            self.assertEqual(child.base_lrs, group_coefficients)
            self.assertEqual(child.get_last_lr(), [c / 2 for c in group_coefficients])
            for group, coefficient in zip(
                child.optimizer.param_groups, group_coefficients
            ):
                self.assertEqual(group["initial_lr"], coefficient)
                self.assertEqual(group["lr"], coefficient / 2)
        resumed.step()
        for optimizer, group_coefficients in zip(optimizers, coefficients):
            for group, coefficient in zip(optimizer.param_groups, group_coefficients):
                self.assertAlmostEqual(group["lr"], coefficient / math.sqrt(5))

    def test_aus_rejects_invalid_group_coefficients_before_changing_lrs(self):
        for coefficient in (0.0, -0.1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(coefficient=coefficient):
                optimizer = Adam(
                    [
                        {"params": [self.model.weight], "aus_coefficient": 0.2},
                        {"params": [self.model.bias], "aus_coefficient": coefficient},
                    ],
                    lr=0.123,
                )
                for group in optimizer.param_groups:
                    group["aus_enabled"] = True
                with self.assertRaisesRegex(ValueError, "aus_coefficient.*group 1"):
                    LRSchedulersContainer.Config(schedule_type="aus").build(
                        optimizers=[optimizer], training_steps=5
                    )
                self.assertEqual(
                    [group["lr"] for group in optimizer.param_groups], [0.123, 0.123]
                )

    def test_aus_requires_optimizer_correction(self):
        config = self.create_trainer_config(
            training_steps=5,
            schedule_type="aus",
        )
        with self.assertRaisesRegex(ValueError, "optimizer.aus_enabled=true"):
            config.lr_scheduler.build(
                optimizers=self.optimizer_container,
                training_steps=config.training.steps,
            )

    def test_linear_warmup_decay(self):
        """Test the linear warmup followed by linear decay schedule."""
        # Create a job config with 10 steps, 2 warmup steps, and linear decay
        config = self.create_trainer_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,  # Use default decay: start decay immediately
            decay_type=None,
            min_lr_factor=None,
        )

        # Build the lr scheduler
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.5,  # Step 0: 50% of max LR (warmup)
            1.0,  # Step 1: 100% of max LR (warmup complete)
            1.0,  # Step 2: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            7.0 / 8.0,  # Step 3: 7/8 of max LR
            6.0 / 8.0,  # Step 4: 3/4 of max LR
            5.0 / 8.0,  # Step 5: 5/8 of max LR
            4.0 / 8.0,  # Step 6: 1/2 of max LR
            3.0 / 8.0,  # Step 7: 3/8 of max LR
            2.0 / 8.0,  # Step 8: 1/4 of max LR
            1.0 / 8.0,  # Step 9: 1/8 of max LR
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            # The LambdaLR multiplies the base lr by the factor
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_warmup_stable_decay(self):
        """Test warmup followed by stable phase and then decay."""
        # Create a job config with 10 steps, 2 warmup steps, 3 stable steps, and 5 decay steps
        config = self.create_trainer_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=0.5,  # 50% of steps for decay
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.5,  # Step 0: 50% of max LR (warmup)
            1.0,  # Step 1: 100% of max LR (warmup complete)
            1.0,  # Step 2: Stable phase
            1.0,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
            1.0,  # Step 5: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            0.8,  # Step 6: Linear decay starts (80% of max LR)
            0.6,  # Step 7: 60% of max LR
            0.4,  # Step 8: 40% of max LR
            0.2,  # Step 9: 20% of max LR
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_min_lr(self):
        """Test that the learning rate doesn't go below the minimum."""
        # Create a job config with a minimum learning rate
        config = self.create_trainer_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=None,
            decay_type="linear",
            min_lr_factor=0.2,  # 20% of base LR as minimum
        )

        # Build the lr scheduler
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        # Step through all steps
        for _ in range(10):
            lr_scheduler.step()

        # After all steps, LR should be at minimum (0.1 * 0.2 = 0.02)
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 0.02, places=6)

    def test_warmup_exceeds_training(self):
        """Test when warmup steps exceed training steps."""
        # Create a job config where warmup steps > training steps
        config = self.create_trainer_config(
            training_steps=5,
            warmup_steps=10,  # More than training steps
            decay_ratio=None,
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler - should adjust warmup steps
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.2,  # Step 0: 50% of max LR (warmup)
            0.4,  # Step 1: 100% of max LR (warmup complete)
            0.6,  # Step 2: Stable phase
            0.8,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_warmup_stable_only(self):
        """Test warmup followed by stable phase only, with no decay phase."""
        # Create a job config with 10 steps, 2 warmup steps, and no decay phase
        config = self.create_trainer_config(
            training_steps=10,
            warmup_steps=2,
            decay_ratio=0.0,  # 0% of steps for decay (no decay)
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.5,  # Step 0: 50% of max LR (warmup)
            1.0,  # Step 1: 100% of max LR (warmup complete)
            1.0,  # Step 2: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            1.0,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
            1.0,  # Step 5: Stable phase
            1.0,  # Step 6: Stable phase
            1.0,  # Step 7: Stable phase
            1.0,  # Step 8: Stable phase
            1.0,  # Step 9: Stable phase
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()

    def test_warmup_plus_decay_exceeds_training(self):
        """Test when warmup + decay steps exceed training steps."""
        # Create a job config where warmup + decay steps > training steps
        # Expected behavior: warmup steps = 5, decay steps = 5
        config = self.create_trainer_config(
            training_steps=10,
            warmup_steps=5,
            decay_ratio=0.8,  # 80% of steps for decay (8 steps)
            decay_type="linear",
            min_lr_factor=0.0,
        )

        # Build the lr scheduler - should adjust warmup steps
        lr_scheduler = config.lr_scheduler.build(
            optimizers=self.optimizer_container,
            training_steps=config.training.steps,
        )

        # Expected adjustment factors for each step
        expected_factors = [
            0.2,  # Step 0: 50% of max LR (warmup)
            0.4,  # Step 1: 100% of max LR (warmup complete)
            0.6,  # Step 2: Stable phase
            0.8,  # Step 3: Stable phase
            1.0,  # Step 4: Stable phase
            1.0,  # Step 5: We maunally added step of stable phase, to prevent LR from dropping to 0 at last step
            0.8,  # Step 6: Linear decay starts (80% of max LR)
            0.6,  # Step 7: 60% of max LR
            0.4,  # Step 8: 40% of max LR
            0.2,  # Step 9: 20% of max LR
        ]

        # Check the learning rate at each step
        for i, factor in enumerate(expected_factors):
            expected_lr = 0.1 * factor
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"],
                expected_lr,
                places=6,
                msg=f"Step {i}: Expected LR {expected_lr}, got {self.optimizer.param_groups[0]['lr']}",
            )
            lr_scheduler.step()


if __name__ == "__main__":
    unittest.main()
