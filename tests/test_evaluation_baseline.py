import json
import shutil
import tempfile
import unittest
from pathlib import Path
from evaluation.baseline import BASELINE, CONFIG, file_hash, verify


class BaselineTests(unittest.TestCase):
    def test_baseline_reproduces_published_probabilities(self):
        self.assertEqual(verify()["drivers"], 22)

    def test_tampered_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "baseline"
            shutil.copytree(BASELINE, target)
            (target / "singapore_predictions.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                verify(target)

    def test_json_hash_is_independent_of_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "config.json"
            target.write_text(json.dumps(json.loads(CONFIG.read_text()), indent=4))
            self.assertEqual(file_hash(target), file_hash(CONFIG))
