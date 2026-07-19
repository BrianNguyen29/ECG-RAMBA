import importlib
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn


extract = importlib.import_module("scripts.revision.22_extract_representations")
probe = importlib.import_module("scripts.revision.20_representation_probe")


class RepresentationContractTests(unittest.TestCase):
    def _canonical_fixture(self, root: Path, n_records: int = 20):
        fold_id = (np.arange(n_records) % 5 + 1).astype(np.int16)
        y_true = np.zeros((n_records, 2), dtype=np.float32)
        y_true[:, 0] = (np.arange(n_records) % 2 == 0).astype(np.float32)
        y_true[:, 1] = (np.arange(n_records) % 3 == 0).astype(np.float32)
        oof_path = root / "oof.npz"
        np.savez_compressed(
            oof_path,
            y_true=y_true,
            record_id=np.arange(n_records, dtype=np.int64),
            fold_id=fold_id,
            class_names=np.asarray(["A", "B"]),
        )
        freeze_path = root / "freeze.json"
        freeze_path.write_text(
            json.dumps({"record_file_sha256": probe.sha256_file(oof_path)}),
            encoding="utf-8",
        )
        canonical = probe.load_current_canonical_contract(oof_path, freeze_path)
        return canonical, fold_id, y_true

    def test_checkpoint_local_cache_membership_and_sha_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical, fold_id, _ = self._canonical_fixture(root)
            rng = np.random.default_rng(42)
            archive_path = root / "chapman.zip"
            archive_path.write_bytes(b"canonical-archive")
            old_zip_path = extract.PATHS["zip_path"]
            old_hydra_dim = extract.CONFIG["hydra_dim"]
            extract.PATHS["zip_path"] = str(archive_path)
            extract.CONFIG["hydra_dim"] = 2
            self.addCleanup(extract.PATHS.__setitem__, "zip_path", old_zip_path)
            self.addCleanup(extract.CONFIG.__setitem__, "hydra_dim", old_hydra_dim)

            source_bundle = extract.representation_source_contract()["bundle_sha256"]
            self.assertEqual(
                source_bundle,
                probe.current_extraction_source_bundle_sha256(),
            )
            raw_input_contract = {
                "schema_version": 1,
                "source_archive_path": str(archive_path),
                "source_archive_sha256": probe.sha256_file(archive_path),
                "input_signal_sha256": "signal-sha",
                "preprocessing_extractor_source_sha256": "preprocess-sha",
                "rocket_extractor_source_sha256": "rocket-source-sha",
                "rocket_transform_config_sha256": "rocket-config-sha",
                "dataset_record_order_fingerprint": "record-order",
                "evaluation_config_hash": extract.EVALUATION_CONFIG_HASH,
            }
            raw_input_contract["contract_sha256"] = extract.canonical_json_sha256(
                raw_input_contract
            )
            split_contract = {
                "fold_assignment_sha256": probe.array_sha256(fold_id, np.int16),
                "folds_file_sha256": "folds-file-sha",
            }
            index_rows = []
            for fold in range(1, 6):
                train_ids = np.flatnonzero(fold_id != fold).astype(np.int64)
                validation_ids = np.flatnonzero(fold_id == fold).astype(np.int64)
                train_embeddings = {
                    key: rng.normal(size=(len(train_ids), 4)).astype(np.float32)
                    for key in extract.EMBEDDING_KEYS
                }
                validation_embeddings = {
                    key: rng.normal(size=(len(validation_ids), 4)).astype(np.float32)
                    for key in extract.EMBEDDING_KEYS
                }
                checkpoint_sha = f"checkpoint-{fold}"
                pca_contract = extract.gen.scoped_cache_contract(
                    raw_input_contract,
                    artifact_kind="fold_train_pca",
                    fold_num=fold,
                    tr_idx=train_ids,
                    va_idx=None,
                    source_config_hash="config",
                )
                pca_path = root / f"fold{fold}.joblib"
                pca = types.SimpleNamespace(
                    n_components_=2,
                    components_=np.ones((2, 3), dtype=np.float64),
                    explained_variance_ratio_=np.asarray([0.6, 0.3], dtype=np.float64),
                )
                extract.joblib.dump(pca, pca_path)
                extract.gen._pca_contract_path(pca_path).write_text(
                    json.dumps(pca_contract, sort_keys=True), encoding="utf-8"
                )
                verified_pca = extract.inspect_pca_artifact(
                    path=pca_path,
                    expected_contract=pca_contract,
                    expected_raw_dim=3,
                )
                self.assertIsNotNone(verified_pca)
                _, pca_identity = verified_pca
                cache_contract = extract.build_fold_embedding_cache_contract(
                    fold_num=fold,
                    checkpoint_kind="final_ema",
                    checkpoint_sha256=checkpoint_sha,
                    source_config_hash="config",
                    tr_idx=train_ids,
                    va_idx=validation_ids,
                    oof_sha256=canonical["oof_sha256"],
                    freeze_sha256=canonical["freeze_sha256"],
                    split_contract=split_contract,
                    cache_provenance=raw_input_contract,
                    dataset_record_fingerprint="record-order",
                    dataset_record_order_sha256="record-order-full-sha",
                    hrv_input_sha256="hrv-sha",
                    pca_identity=pca_identity,
                )
                cache_path = root / f"fold{fold}.npz"
                extract.save_fold_embedding_cache(
                    path=cache_path,
                    fold_num=fold,
                    tr_idx=train_ids,
                    va_idx=validation_ids,
                    train_embeddings=train_embeddings,
                    validation_embeddings=validation_embeddings,
                    train_slice_count=np.ones(len(train_ids), dtype=np.int16),
                    validation_slice_count=np.ones(len(validation_ids), dtype=np.int16),
                    cache_contract=cache_contract,
                    summary={
                        "source_config_hash": "config",
                        "coordinate_system_id": f"fold{fold}:{checkpoint_sha}",
                    },
                )
                index_rows.append(
                    {
                        "fold": fold,
                        "path": str(cache_path),
                        "sha256": probe.sha256_file(cache_path),
                        "checkpoint_sha256": checkpoint_sha,
                        "coordinate_system_id": f"fold{fold}:{checkpoint_sha}",
                        "train_index_sha256": probe.array_sha256(train_ids, np.int64),
                        "validation_index_sha256": probe.array_sha256(
                            validation_ids, np.int64
                        ),
                        "cache_contract_sha256": cache_contract["contract_sha256"],
                        "cache_contract": cache_contract,
                    }
                )

            emb = {
                "source_oof_sha256": canonical["oof_sha256"],
                "source_freeze_sha256": canonical["freeze_sha256"],
                "source_bundle_sha256": source_bundle,
                "local_fold_cache_index": index_rows,
            }
            local_folds = probe.load_checkpoint_local_folds(emb, canonical)
            self.assertEqual([row["fold_id"] for row in local_folds], [1, 2, 3, 4, 5])
            for row in local_folds:
                self.assertEqual(
                    np.intersect1d(row["train_record_id"], row["validation_record_id"]).size,
                    0,
                )
            global_views = {
                view: np.zeros((len(fold_id), 4), dtype=np.float32)
                for view in probe.VIEW_KEYS
            }
            for row in local_folds:
                validation_ids = row["validation_record_id"]
                for view in global_views:
                    global_views[view][validation_ids] = row["views"][view]["validation"]
            projection_emb = {**emb, "views": global_views}
            probe.validate_global_embedding_projection(projection_emb, local_folds)

            oof_contract = {
                "path": canonical["oof_path"],
                "sha256": canonical["oof_sha256"],
                "freeze_manifest": canonical["freeze_manifest_path"],
                "freeze_manifest_sha256": canonical["freeze_sha256"],
                "y_true": canonical["y_true"],
                "record_id": np.arange(len(fold_id), dtype=np.int64),
                "fold_id": fold_id,
                "class_names": canonical["class_names"],
            }
            final_path = root / "representation_embeddings.npz"
            final_embeddings = {
                candidates[0]: global_views[view]
                for view, candidates in probe.VIEW_KEYS.items()
            }
            extract.write_final_embedding_npz(
                path=final_path,
                oof=oof_contract,
                embeddings=final_embeddings,
                fold_id=fold_id,
                slice_count=np.ones(len(fold_id), dtype=np.int16),
                payload={
                    "checkpoint_kind": "final_ema",
                    "dataset_record_order_fingerprint": "record-order",
                    "source_contract": extract.representation_source_contract(),
                    "local_fold_cache_index": index_rows,
                    "fold_summaries": [
                        {
                            "fold": fold,
                            "checkpoint_sha256": f"checkpoint-{fold}",
                        }
                        for fold in range(1, 6)
                    ],
                },
            )
            checkpoint_contracts = {
                fold: {"sha256": f"checkpoint-{fold}"}
                for fold in range(1, 6)
            }
            audit = extract.inspect_final_embedding_reuse(
                final_path,
                oof_contract,
                "final_ema",
                checkpoint_contracts,
                split_contract,
            )
            self.assertTrue(audit["reusable"], audit["issues"])

            projection_emb["views"]["morphology"][0, 0] += 1.0
            with self.assertRaisesRegex(RuntimeError, "Global OOF projection differs"):
                probe.validate_global_embedding_projection(projection_emb, local_folds)

            stale = dict(emb)
            stale["source_oof_sha256"] = "0" * 64
            stale.update(
                {
                    "y_true": canonical["y_true"],
                    "record_id": canonical["record_id"],
                    "fold_id": canonical["fold_id"],
                    "class_names": canonical["class_names"],
                }
            )
            with self.assertRaisesRegex(RuntimeError, "oof_predictions_sha256"):
                probe.validate_embedding_against_canonical(stale, canonical)

            stale_source = {
                **stale,
                "source_oof_sha256": canonical["oof_sha256"],
                "source_bundle_sha256": "1" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "source_bundle_sha256"):
                probe.validate_embedding_against_canonical(stale_source, canonical)

            bad_index = [dict(row) for row in index_rows]
            bad_index[0]["sha256"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "cache SHA mismatch"):
                probe.load_checkpoint_local_folds(
                    {**emb, "local_fold_cache_index": bad_index}, canonical
                )

    def test_probe_fits_and_evaluates_within_each_checkpoint_coordinate_system(self):
        rng = np.random.default_rng(7)
        n_records = 250
        latent = rng.normal(size=(n_records, 5))
        y = (latent[:, [0]] + 0.15 * latent[:, [1]] > 0).astype(np.float32)
        fold_id = (np.arange(n_records) % 5 + 1).astype(np.int16)
        local_folds = []
        for fold in range(1, 6):
            train_ids = np.flatnonzero(fold_id != fold)
            validation_ids = np.flatnonzero(fold_id == fold)
            rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
            local = (latent @ rotation).astype(np.float32)
            local_folds.append(
                {
                    "fold_id": fold,
                    "coordinate_system_id": f"fold{fold}:checkpoint",
                    "checkpoint_sha256": f"checkpoint-{fold}",
                    "train_record_id": train_ids,
                    "validation_record_id": validation_ids,
                    "views": {
                        "morphology": {
                            "train": local[train_ids],
                            "validation": local[validation_ids],
                        }
                    },
                }
            )

        result = probe.probe_one_view(
            local_folds,
            "morphology",
            y,
            threshold=0.5,
            seed=42,
            max_iter=2000,
        )
        fold_rows = result.pop("_fold_rows")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["coordinate_design"], probe.LOCAL_COORDINATE_PROTOCOL)
        self.assertEqual(len(fold_rows), 5)
        self.assertGreater(result["roc_auc_macro"], 0.95)
        self.assertGreater(result["pr_auc_macro"], 0.95)
        self.assertTrue(all(row["n_train_records"] == 200 for row in fold_rows))

    def test_fold_cache_rejects_legacy_raw_input_and_pca_contract_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tr_idx = np.asarray([0, 1, 2], dtype=np.int64)
            va_idx = np.asarray([3, 4], dtype=np.int64)
            raw_contract = {
                "source_archive_sha256": "archive",
                "input_signal_sha256": "signal",
                "dataset_record_order_fingerprint": "records",
            }
            raw_contract["contract_sha256"] = extract.canonical_json_sha256(raw_contract)
            pca_contract = extract.gen.scoped_cache_contract(
                raw_contract,
                artifact_kind="fold_train_pca",
                fold_num=1,
                tr_idx=tr_idx,
                va_idx=None,
                source_config_hash="config",
            )
            contract = extract.build_fold_embedding_cache_contract(
                fold_num=1,
                checkpoint_kind="final_ema",
                checkpoint_sha256="checkpoint",
                source_config_hash="config",
                tr_idx=tr_idx,
                va_idx=va_idx,
                oof_sha256="oof",
                freeze_sha256="freeze",
                split_contract={
                    "fold_assignment_sha256": "fold-assignment",
                    "folds_file_sha256": "folds-file",
                },
                cache_provenance=raw_contract,
                dataset_record_fingerprint="records",
                dataset_record_order_sha256="records-full",
                hrv_input_sha256="hrv",
                pca_identity={
                    "pca_model_path": "pca.joblib",
                    "pca_model_sha256": "pca-model",
                    "pca_model_size_bytes": 10,
                    "pca_contract_path": "pca.contract.json",
                    "pca_contract_file_sha256": "pca-contract-file",
                    "pca_contract_sha256": pca_contract["contract_sha256"],
                    "pca_contract": pca_contract,
                    "pca_n_components": 2,
                    "pca_raw_dim": 3,
                },
            )
            path = root / "fold.npz"
            train_embeddings = {
                key: np.ones((len(tr_idx), 2), dtype=np.float32)
                for key in extract.EMBEDDING_KEYS
            }
            validation_embeddings = {
                key: np.ones((len(va_idx), 2), dtype=np.float32)
                for key in extract.EMBEDDING_KEYS
            }
            extract.save_fold_embedding_cache(
                path=path,
                fold_num=1,
                tr_idx=tr_idx,
                va_idx=va_idx,
                train_embeddings=train_embeddings,
                validation_embeddings=validation_embeddings,
                train_slice_count=np.ones(len(tr_idx), dtype=np.int16),
                validation_slice_count=np.ones(len(va_idx), dtype=np.int16),
                cache_contract=contract,
                summary={"source_config_hash": "config"},
            )
            self.assertIsNotNone(
                extract.load_fold_embedding_cache(
                    path=path,
                    fold_num=1,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    expected_contract=contract,
                )
            )

            changed_raw = deepcopy(contract)
            changed_raw["raw_input_contract"]["input_signal_sha256"] = "different-signal"
            raw_without_sha = {
                key: value
                for key, value in changed_raw["raw_input_contract"].items()
                if key != "contract_sha256"
            }
            changed_raw["raw_input_contract"]["contract_sha256"] = (
                extract.canonical_json_sha256(raw_without_sha)
            )
            changed_raw["raw_input_contract_sha256"] = changed_raw["raw_input_contract"][
                "contract_sha256"
            ]
            changed_raw["contract_sha256"] = extract.canonical_json_sha256(
                {key: value for key, value in changed_raw.items() if key != "contract_sha256"}
            )
            self.assertNotEqual(
                extract.fold_embedding_cache_path(1, "final_ema", root, contract),
                extract.fold_embedding_cache_path(1, "final_ema", root, changed_raw),
            )
            self.assertIsNone(
                extract.load_fold_embedding_cache(
                    path=path,
                    fold_num=1,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    expected_contract=changed_raw,
                )
            )

            changed_pca = deepcopy(contract)
            changed_pca["pca_model_sha256"] = "different-pca"
            changed_pca["contract_sha256"] = extract.canonical_json_sha256(
                {key: value for key, value in changed_pca.items() if key != "contract_sha256"}
            )
            self.assertIsNone(
                extract.load_fold_embedding_cache(
                    path=path,
                    fold_num=1,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    expected_contract=changed_pca,
                )
            )

            cache_bytes = path.read_bytes()
            path.write_bytes(cache_bytes[: len(cache_bytes) // 2])
            self.assertIsNone(
                extract.load_fold_embedding_cache(
                    path=path,
                    fold_num=1,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    expected_contract=contract,
                )
            )

            legacy_path = root / "legacy.npz"
            np.savez_compressed(legacy_path, train_record_id=tr_idx)
            self.assertIsNone(
                extract.load_fold_embedding_cache(
                    path=legacy_path,
                    fold_num=1,
                    tr_idx=tr_idx,
                    va_idx=va_idx,
                    expected_contract=contract,
                )
            )

    def test_atomic_representation_write_does_not_expose_failed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            with mock.patch.object(
                extract,
                "validate_npz_payload",
                side_effect=RuntimeError("forced validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced validation failure"):
                    extract.atomic_savez_compressed(
                        path,
                        {"array": np.arange(5, dtype=np.float32)},
                    )
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(f".{path.name}.write.lock").exists())
            self.assertEqual(list(path.parent.glob(f"{path.name}.partial.*")), [])


class ForwardEmbeddingParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "mamba_ssm" not in sys.modules:
            module = types.ModuleType("mamba_ssm")

            class IdentityMamba(nn.Module):
                def __init__(self, d_model=None, *args, **kwargs):
                    super().__init__()

                def forward(self, values):
                    return values

            module.Mamba2 = IdentityMamba
            module.Mamba = IdentityMamba
            sys.modules["mamba_ssm"] = module
        cls.model_module = importlib.import_module("src.model")

    def test_forward_with_embeddings_logits_match_forward_for_supported_ablations(self):
        cfg = dict(self.model_module.CONFIG)
        cfg.update(
            {
                "d_model": 24,
                "hydra_dim": 12,
                "hrv_dim": 6,
                "n_latents": 4,
                "n_layers": 0,
                "drop_path_rate": 0.0,
                "use_spatial_attention": False,
                "use_cross_attention_fusion": True,
                "use_final_perceiver": True,
                "fusion_heads": 8,
            }
        )
        torch.manual_seed(11)
        x = torch.randn(2, 12, 128)
        xh = torch.randn(2, cfg["hydra_dim"])
        xhr = torch.randn(2, cfg["hrv_dim"])
        for variant, ablation in self.model_module.STRUCTURED_ABLATION_SPECS.items():
            with self.subTest(variant=variant):
                model = self.model_module.ECGRambaV7Advanced(cfg=cfg, ablation=ablation)
                model.eval()
                with torch.no_grad():
                    expected = model(x, xh, xhr)
                    observed, embeddings = model.forward_with_embeddings(x, xh, xhr)
                    wrapped, wrapped_embeddings = extract.forward_with_embeddings(
                        model, x, xh, xhr
                    )
                torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
                torch.testing.assert_close(wrapped, expected, rtol=0.0, atol=0.0)
                self.assertEqual(set(embeddings), set(extract.EMBEDDING_KEYS))
                self.assertEqual(set(wrapped_embeddings), set(extract.EMBEDDING_KEYS))
                for key in extract.EMBEDDING_KEYS:
                    self.assertEqual(tuple(embeddings[key].shape), (2, cfg["d_model"]))


if __name__ == "__main__":
    unittest.main()
