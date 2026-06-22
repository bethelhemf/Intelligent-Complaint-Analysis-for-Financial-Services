$testCode = @"
import unittest
import os
import faiss
from src import config

class TestCrediTrustPipeline(unittest.TestCase):
    def test_directory_structure(self):
        self.assertTrue(os.path.exists('src'), "src folder missing")
        self.assertTrue(os.path.exists('notebooks'), "notebooks folder missing")

    def test_vector_store_integrity(self):
        # Only run if index exists
        if os.path.exists(config.INDEX_PATH):
            index = faiss.read_index(config.INDEX_PATH)
            self.assertGreater(index.ntotal, 0, "FAISS index exists but is empty")
        else:
            print("Skipping index test: Index not built yet.")

if __name__ == '__main__':
    unittest.main()
"@
$testCode | Out-File -FilePath "tests/test_pipeline.py" -Encoding utf8