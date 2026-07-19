import types
import unittest
from unittest import mock

from src import features
from src.provenance import canonical_json_sha256


class HrvCacheRuntimeContractTests(unittest.TestCase):
    def test_runtime_contract_records_detector_and_library_versions(self):
        contract = features.hrv_extractor_runtime_contract()
        self.assertEqual(
            contract["cache_schema_version"], features.HRV36_CACHE_SCHEMA_VERSION
        )
        self.assertIn("detector_policy", contract)
        self.assertIn("numpy_version", contract)
        self.assertIn("scipy_version", contract)

    def test_neurokit_availability_changes_cache_identity(self):
        with mock.patch.object(features, "HAS_NEUROKIT", False), mock.patch.object(
            features, "nk", None
        ):
            scipy_only = canonical_json_sha256(
                features.hrv_extractor_runtime_contract()
            )
        with mock.patch.object(features, "HAS_NEUROKIT", True), mock.patch.object(
            features, "nk", types.SimpleNamespace(__version__="test-version")
        ):
            neurokit = canonical_json_sha256(
                features.hrv_extractor_runtime_contract()
            )
        self.assertNotEqual(scipy_only, neurokit)


if __name__ == "__main__":
    unittest.main()
