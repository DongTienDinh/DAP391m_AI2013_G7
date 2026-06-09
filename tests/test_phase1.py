import unittest
from pathlib import Path

import numpy as np

from src.olist_pipeline.core.config import AppConfig, get_config
from src.olist_pipeline.core.exceptions import ConfigurationError, DataValidationError
from src.olist_pipeline.utils.config_loader import Config as BackwardConfig
from src.olist_pipeline.utils.math_utils import softclip_positive


class TestPhase1(unittest.TestCase):
    def test_config_loading(self):
        """Test that config loads and validates correctly."""
        config = get_config()
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.paths.data.raw_olist, Path)
        self.assertEqual(config.training.random_state, 42)

    def test_config_backward_compatibility(self):
        """Test that the old Config utility still works."""
        self.assertIsInstance(BackwardConfig.paths, dict)
        self.assertIsInstance(BackwardConfig.training, dict)
        self.assertIsInstance(BackwardConfig.get_path("data", "raw_olist"), Path)
        self.assertTrue(BackwardConfig.get_path("data", "raw_olist").is_absolute())

    def test_softclip_positive(self):
        """Test softclip_positive function and its error handling."""
        x = np.array([-1.0, 0.0, 1.0])

        # Test valid case
        result = softclip_positive(x, k=3.0)
        self.assertEqual(len(result), 3)
        self.assertGreater(result[1], 0)  # softclip(0) should be > 0

        # Test error case
        with self.assertRaisesRegex(DataValidationError, "Parameter k must be positive"):
            softclip_positive(x, k=-1.0)

    def test_config_missing_file(self):
        """Test ConfigurationError when config file is missing."""
        import src.olist_pipeline.core.config as config_mod
        original_root_fn = config_mod.get_project_root

        # Mock project root to a temp directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            (tmp_root / "configs").mkdir()

            # Monkeypatch
            config_mod.get_project_root = lambda: tmp_root

            try:
                with self.assertRaises(ConfigurationError):
                    AppConfig.load()
            finally:
                # Restore
                config_mod.get_project_root = original_root_fn

if __name__ == "__main__":
    unittest.main()
