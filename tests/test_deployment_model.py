import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pioupiou.inference.deployment import ensure_deployment_model


class DeploymentModelTests(unittest.TestCase):
    def test_stale_model_is_replaced_by_sha_verified_release_asset(self):
        downloaded = b"new deployment model"
        expected_sha256 = hashlib.sha256(downloaded).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "artifacts" / "traverse_model.joblib"
            model.parent.mkdir()
            model.write_bytes(b"stale model")
            manifest = root / "deployment_model.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "release": "test-model",
                        "url": "https://example.test/traverse_model.joblib",
                        "sha256": expected_sha256,
                    }
                )
            )
            with patch(
                "pioupiou.inference.deployment.urllib.request.urlopen",
                return_value=BytesIO(downloaded),
            ):
                result = ensure_deployment_model(model, manifest)

            self.assertEqual(result, expected_sha256)
            self.assertEqual(model.read_bytes(), downloaded)


if __name__ == "__main__":
    unittest.main()
