import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import joblib
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "revision" / "08_build_fold_pca.py"
SPEC = importlib.util.spec_from_file_location("fold_pca_builder_contract", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FoldPcaCacheContractTests(unittest.TestCase):
    def _artifact(self, root: Path):
        path = root / "fold1.joblib"
        contract = {
            "artifact_kind": "fold_train_pca",
            "contract_sha256": "c" * 64,
        }
        pca = types.SimpleNamespace(
            n_components_=2,
            components_=np.ones((2, 3), dtype=np.float64),
            explained_variance_ratio_=np.asarray([0.6, 0.3], dtype=np.float64),
        )
        joblib.dump(pca, path)
        path.with_suffix(path.suffix + ".contract.json").write_text(
            json.dumps(contract, sort_keys=True), encoding="utf-8"
        )
        integrity = MODULE.bind_pca_artifact(
            path=path,
            cache_contract=contract,
            expected_raw_dim=3,
            expected_components=2,
        )
        return path, contract, pca, integrity

    def test_reuse_requires_bound_artifact_sha_and_decodable_pca(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, contract, _, _ = self._artifact(Path(tmp))
            verified = MODULE.load_verified_pca_artifact(
                path=path,
                cache_contract=contract,
                expected_raw_dim=3,
                expected_components=2,
            )
            self.assertIsNotNone(verified)

            with path.open("ab") as handle:
                handle.write(b"mutation")
            self.assertIsNone(
                MODULE.load_verified_pca_artifact(
                    path=path,
                    cache_contract=contract,
                    expected_raw_dim=3,
                    expected_components=2,
                )
            )

    def test_manifest_recovery_is_not_key_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, contract, pca, integrity = self._artifact(root)
            train_indices = np.asarray([0, 2, 4], dtype=np.int64)
            row = MODULE.pca_manifest_row(
                fold_num=1,
                destination=path,
                train_indices=train_indices,
                source_config_hash="config",
                checkpoint_kind="final_ema",
                cache_contract=contract,
                pca=pca,
                integrity=integrity,
            )
            recovered = MODULE.recover_verified_pca_manifest_row(
                fold_num=1,
                destination=path,
                train_indices=train_indices,
                source_config_hash="config",
                checkpoint_kind="final_ema",
                cache_contract=contract,
                prior_row=row,
                expected_raw_dim=3,
                expected_components=2,
            )
            self.assertIsNotNone(recovered)

            key_only = {"fold": 1, "path": str(path)}
            self.assertIsNone(
                MODULE.recover_verified_pca_manifest_row(
                    fold_num=1,
                    destination=path,
                    train_indices=train_indices,
                    source_config_hash="config",
                    checkpoint_kind="final_ema",
                    cache_contract=contract,
                    prior_row=key_only,
                    expected_raw_dim=3,
                    expected_components=2,
                )
            )

            path.unlink()
            self.assertIsNone(
                MODULE.recover_verified_pca_manifest_row(
                    fold_num=1,
                    destination=path,
                    train_indices=train_indices,
                    source_config_hash="config",
                    checkpoint_kind="final_ema",
                    cache_contract=contract,
                    prior_row=row,
                    expected_raw_dim=3,
                    expected_components=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
