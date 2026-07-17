from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "revision" / "39_morphology_learnability_control.py"
PAIRED_SCRIPT = ROOT / "scripts" / "revision" / "40_paired_morphology_learnability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("morphology_learnability_control", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MorphologyLearnabilityControlTests(unittest.TestCase):
    def test_paired_bootstrap_helper_is_called_with_keyword_only_contract(self):
        tree = ast.parse(PAIRED_SCRIPT.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "paired_bootstrap_difference"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, [])
        self.assertEqual(
            {keyword.arg for keyword in calls[0].keywords},
            {"y_true", "full_prob", "comparator_prob", "spec", "n_boot", "seed"},
        )

    def test_frozen_channels_are_restored_after_weight_decay_step(self):
        module = load_module()
        bank = module.ControlledRandomConvolutionBank(
            c_in=2,
            num_kernels=8,
            kernel_length=3,
            dilations=[1, 2],
            trainable_fraction=0.5,
            ppv_temperature=0.1,
            seed=42,
        )
        initial = [conv.weight.detach().clone() for conv in bank.convs]
        optimizer = torch.optim.AdamW(bank.parameters(), lr=1e-2, weight_decay=0.1)
        x = torch.randn(4, 2, 32)
        optimizer.zero_grad(set_to_none=True)
        bank(x).sum().backward()
        optimizer.step()
        bank.restore_frozen_kernel_channels()

        trainable_changed = False
        for conv, before, mask_name in zip(bank.convs, initial, bank._trainable_mask_names):
            mask = getattr(bank, mask_name).bool().expand_as(conv.weight)
            self.assertTrue(torch.equal(conv.weight.detach()[~mask], before[~mask]))
            trainable_changed = trainable_changed or bool(
                torch.any(conv.weight.detach()[mask] != before[mask])
            )
        self.assertTrue(trainable_changed)

    def test_variants_have_matched_kernel_and_head_initialization(self):
        module = load_module()
        params = {
            "num_kernels": 8,
            "kernel_length": 3,
            "dilations": [1, 2],
            "ppv_temperature": 0.1,
            "bank_seed": 42,
            "hidden_dim": 4,
            "dropout": 0.0,
            "n_classes": 3,
        }
        torch.manual_seed(7)
        frozen = module.MorphologyLearnabilityControl(
            model_params=params, trainable_fraction=0.0
        )
        torch.manual_seed(7)
        partial = module.MorphologyLearnabilityControl(
            model_params=params, trainable_fraction=0.5
        )

        self.assertEqual(module.tensor_state_hash(frozen), module.tensor_state_hash(partial))
        self.assertEqual(frozen.bank.trainable_kernel_count, 0)
        self.assertEqual(partial.bank.trainable_kernel_count, 4)


if __name__ == "__main__":
    unittest.main()
