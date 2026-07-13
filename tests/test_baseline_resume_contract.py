import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revision"
    / "16_raw_mamba_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("raw_mamba_resume_test", SCRIPT)
assert SPEC and SPEC.loader
RAW_MAMBA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RAW_MAMBA
SPEC.loader.exec_module(RAW_MAMBA)


def load_revision_script(module_name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / "revision" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RESNET = load_revision_script("resnet_resume_test", "14_resnet1d_cnn_baseline.py")
HYBRID = load_revision_script("hybrid_resume_test", "26_hybrid_morphology_baseline.py")


class RawMambaResumeContractTests(unittest.TestCase):
    def test_checkpoint_regenerates_missing_fold_cache_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            fold_cache_dir = root / "folds"
            checkpoint_dir.mkdir()
            fold_cache_dir.mkdir()

            model = torch.nn.Linear(1, 1)
            freeze_contract = {"freeze_manifest_sha256": "freeze-sha"}
            load_info = {
                "oof_predictions_sha256": "oof-sha",
                "raw_cache_sha256": "raw-sha",
                "dataset_record_order_fingerprint": "record-order",
                "freeze_contract": freeze_contract,
            }
            args = SimpleNamespace(
                seed=42,
                force_rerun=False,
                reuse_checkpoints=True,
                checkpoint_dir=checkpoint_dir,
                ema_decay=0.999,
                batch_size=2,
                num_workers=0,
                threshold=0.5,
                epochs=20,
                lr=0.0009,
                lr_min=1e-6,
                weight_decay=0.05,
                grad_clip=1.0,
                bce_pos_weight="fold",
                asym_start_epoch=8,
                amp=False,
                amp_dtype="float16",
                allow_tf32=False,
            )
            checkpoint_path = checkpoint_dir / "fold1_raw_mamba_final_ema.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": 1,
                    "protocol": RAW_MAMBA.PROTOCOL,
                    "feature_contract": RAW_MAMBA.FEATURE_CONTRACT,
                    "weights_kind": "ema",
                    "args": vars(args),
                    "load_info": load_info,
                },
                checkpoint_path,
            )

            x = np.zeros((2, 12, int(RAW_MAMBA.CONFIG["slice_length"])), dtype=np.float32)
            y = np.zeros((2, 27), dtype=np.float32)
            y[1, 0] = 1.0
            slice_prob = np.full((1, 27), 0.1, dtype=np.float32)
            slice_prob[0, 0] = 0.9

            def fail_if_optimizer_is_built(*_args, **_kwargs):
                raise AssertionError("checkpoint resume unexpectedly entered training")

            with patch.object(RAW_MAMBA, "FOLD_PREDICTION_DIR", fold_cache_dir), patch.object(
                RAW_MAMBA, "build_model", return_value=model
            ), patch.object(
                RAW_MAMBA,
                "predict_slice_probabilities",
                return_value=(
                    slice_prob,
                    np.asarray([1], dtype=np.int64),
                    np.asarray([0], dtype=np.int32),
                ),
            ), patch.object(torch.optim, "AdamW", side_effect=fail_if_optimizer_is_built):
                result = RAW_MAMBA.train_one_fold(
                    fold=1,
                    X=x,
                    y=y,
                    tr_idx=np.asarray([0], dtype=np.int64),
                    va_idx=np.asarray([1], dtype=np.int64),
                    device=torch.device("cpu"),
                    args=args,
                    load_info=load_info,
                    model_params={"test": True},
                )

            summary = result[-1]
            self.assertTrue(summary["reused_checkpoint"])
            self.assertFalse(summary["reused_fold_predictions"])
            self.assertTrue((fold_cache_dir / "raw_mamba_fold1_predictions.npz").exists())

    def test_resnet_trusted_legacy_checkpoint_regenerates_cache_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            fold_cache_dir = root / "folds"
            checkpoint_dir.mkdir()
            fold_cache_dir.mkdir()

            model = torch.nn.Linear(1, 1)
            freeze_contract = {"freeze_manifest_sha256": "freeze-sha"}
            load_info = {
                "oof_predictions_sha256": "oof-sha",
                "raw_cache_sha256": "raw-sha",
                "dataset_record_order_fingerprint": "record-order",
                "freeze_contract": freeze_contract,
            }
            args = SimpleNamespace(
                seed=42,
                force_rerun=False,
                reuse_checkpoints=True,
                allow_legacy_checkpoint_metadata=True,
                checkpoint_dir=checkpoint_dir,
                save_checkpoints=True,
                batch_size=2,
                num_workers=0,
                threshold=0.5,
                epochs=20,
                base_channels=64,
                dropout=0.2,
                lr=0.001,
                weight_decay=0.0001,
                amp=False,
                allow_tf32=False,
            )
            checkpoint_path = checkpoint_dir / "fold1_resnet1d_cnn_final.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": 1,
                    "protocol": RESNET.PROTOCOL,
                    "args": vars(args),
                    "load_info": load_info,
                },
                checkpoint_path,
            )

            x = np.zeros((2, 12, int(RESNET.CONFIG["slice_length"])), dtype=np.float32)
            y = np.zeros((2, 27), dtype=np.float32)
            y[1, 0] = 1.0
            slice_prob = np.full((1, 27), 0.1, dtype=np.float32)
            slice_prob[0, 0] = 0.9

            def fail_if_optimizer_is_built(*_args, **_kwargs):
                raise AssertionError("legacy checkpoint resume unexpectedly entered training")

            with patch.object(RESNET, "FOLD_PREDICTION_DIR", fold_cache_dir), patch.object(
                RESNET, "build_model", return_value=model
            ), patch.object(RESNET, "build_loader", return_value=[]), patch.object(
                RESNET,
                "predict_slice_probabilities",
                return_value=(
                    slice_prob,
                    np.asarray([1], dtype=np.int64),
                    np.asarray([0], dtype=np.int32),
                ),
            ), patch.object(torch.optim, "AdamW", side_effect=fail_if_optimizer_is_built):
                result = RESNET.train_one_fold(
                    fold=1,
                    X=x,
                    y=y,
                    tr_idx=np.asarray([0], dtype=np.int64),
                    va_idx=np.asarray([1], dtype=np.int64),
                    device=torch.device("cpu"),
                    args=args,
                    load_info=load_info,
                    model_params={"architecture": RESNET.ARCHITECTURE_NAME},
                )

            summary = result[-1]
            self.assertTrue(summary["reused_checkpoint"])
            self.assertFalse(summary["reused_fold_predictions"])
            self.assertTrue((fold_cache_dir / "resnet1d_cnn_fold1_predictions.npz").exists())

    def test_legacy_transformer_checkpoint_requires_architecture_arguments(self):
        args = SimpleNamespace(
            epochs=20,
            batch_size=256,
            base_channels=96,
            dropout=0.2,
            lr=0.001,
            weight_decay=0.0001,
            transformer_embed_dim=96,
            transformer_heads=4,
            transformer_depth=3,
            transformer_patch_size=50,
            transformer_patch_stride=25,
            transformer_ff_multiplier=4,
        )
        saved_args = vars(args).copy()
        checkpoint_path = Path("fold1_transformer_ecg_final.pt")
        RESNET.validate_legacy_checkpoint_arguments(
            saved_args,
            args,
            checkpoint_path,
            architecture_name="patch_transformer_raw_ecg",
        )

        saved_args.pop("transformer_heads")
        with self.assertRaisesRegex(ValueError, "transformer_heads"):
            RESNET.validate_legacy_checkpoint_arguments(
                saved_args,
                args,
                checkpoint_path,
                architecture_name="patch_transformer_raw_ecg",
            )

    def test_resnet_fold_cache_is_bound_to_exact_checkpoint_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "fold1.npz"
            checkpoint = root / "fold1.pt"
            checkpoint.write_bytes(b"checkpoint-a")
            y = np.zeros((2, 27), dtype=np.float32)
            val_indices = np.asarray([1], dtype=np.int64)
            y_prob = np.zeros_like(y)
            slice_prob = np.zeros((1, 27), dtype=np.float32)
            slice_record_id = np.asarray([1], dtype=np.int64)
            slice_start = np.asarray([0], dtype=np.int32)
            slice_count = np.asarray([0, 1], dtype=np.int16)
            load_info = {"oof_predictions_sha256": "oof", "raw_cache_sha256": "raw"}
            args = SimpleNamespace(epochs=20, seed=42)
            model_params = {"architecture": "test"}

            RESNET.save_fold_predictions(
                cache,
                fold=1,
                val_indices=val_indices,
                y_prob=y_prob,
                slice_count=slice_count,
                slice_prob=slice_prob,
                slice_record_id=slice_record_id,
                slice_start=slice_start,
                checkpoint_sha256=RESNET.sha256_file(checkpoint),
                load_info=load_info,
                args=args,
                model_params=model_params,
            )
            self.assertIsNotNone(
                RESNET.fold_prediction_matches(
                    cache,
                    checkpoint_path=checkpoint,
                    y=y,
                    val_indices=val_indices,
                    load_info=load_info,
                    args=args,
                    model_params=model_params,
                )
            )

            checkpoint.write_bytes(b"checkpoint-b")
            self.assertIsNone(
                RESNET.fold_prediction_matches(
                    cache,
                    checkpoint_path=checkpoint,
                    y=y,
                    val_indices=val_indices,
                    load_info=load_info,
                    args=args,
                    model_params=model_params,
                )
            )

    def test_hybrid_fold_cache_is_bound_to_exact_checkpoint_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "fold1.npz"
            checkpoint = root / "fold1.pt"
            checkpoint.write_bytes(b"checkpoint-a")
            val_indices = np.asarray([1], dtype=np.int64)
            params_json = '{"hidden_dim": 8}'
            load_info = {
                "oof_predictions_sha256": "oof",
                "minirocket_cache_sha256": "rocket",
            }
            np.savez_compressed(
                cache,
                protocol=np.asarray(HYBRID.PROTOCOL),
                classifier_params_json=np.asarray(params_json),
                oof_predictions_sha256=np.asarray("oof"),
                minirocket_cache_sha256=np.asarray("rocket"),
                checkpoint_sha256=np.asarray(HYBRID.sha256_file(checkpoint)),
                val_indices=val_indices,
                y_prob=np.zeros((1, 27), dtype=np.float32),
            )
            self.assertIsNotNone(
                HYBRID.load_reusable_fold_cache(
                    cache_path=cache,
                    checkpoint_path=checkpoint,
                    val_indices=val_indices,
                    n_classes=27,
                    params_json=params_json,
                    load_info=load_info,
                )
            )
            checkpoint.write_bytes(b"checkpoint-b")
            self.assertIsNone(
                HYBRID.load_reusable_fold_cache(
                    cache_path=cache,
                    checkpoint_path=checkpoint,
                    val_indices=val_indices,
                    n_classes=27,
                    params_json=params_json,
                    load_info=load_info,
                )
            )

    def test_final_prediction_contract_requires_all_five_checkpoint_hashes(self):
        contract = {
            "status": "complete",
            "missing_folds": [],
            "checkpoints": [
                {"fold": fold, "sha256": str(fold) * 64}
                for fold in range(1, 6)
            ],
        }
        metadata = RESNET.checkpoint_contract_metadata(contract)
        np.testing.assert_array_equal(metadata["checkpoint_folds"], [1, 2, 3, 4, 5])
        self.assertEqual(metadata["checkpoint_sha256"].shape, (5,))

        incomplete = {"status": "partial", "missing_folds": [5], "checkpoints": contract["checkpoints"][:4]}
        with self.assertRaises(RuntimeError):
            RESNET.checkpoint_contract_metadata(incomplete)

    def test_raw_mamba_checkpoint_contract_records_weights_kind_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            for fold in range(1, 6):
                (checkpoint_dir / f"fold{fold}_raw_mamba_final_ema.pt").write_bytes(
                    f"fold-{fold}".encode("ascii")
                )
            contract = RAW_MAMBA.build_checkpoint_contract(
                SimpleNamespace(checkpoint_dir=checkpoint_dir, ema_decay=0.999)
            )
            self.assertEqual(contract["status"], "complete")
            self.assertEqual(contract["weights_kind"], "ema")
            self.assertEqual([row["fold"] for row in contract["checkpoints"]], [1, 2, 3, 4, 5])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in contract["checkpoints"]))


if __name__ == "__main__":
    unittest.main()
