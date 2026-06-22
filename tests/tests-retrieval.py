# Create the tests directory if it doesn't exist
if (!(Test-Path "tests")) { New-Item -ItemType Directory -Path "tests" }

$testCode = @"
import unittest
import os
import faiss
import pickle

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.index_path = 'vector_store/complaints_index.faiss'
        self.meta_path = 'vector_store/metadata.pkl'

    def test_files_exist(self):
        self.assertTrue(os.path.exists(self.index_path), "FAISS index file missing.")
        self.assertTrue(os.path.exists(self.meta_path), "Metadata pickle file missing.")

    def test_index_loading(self):
        index = faiss.read_index(self.index_path)
        self.assertGreater(index.ntotal, 0, "FAISS index is empty.")

    def test_metadata_consistency(self):
        with open(self.meta_path, 'rb') as f:
            data = pickle.load(f)
        self.assertIn('chunks', data)
        self.assertIn('metadata', data)
        self.assertEqual(len(data['chunks']), len(data['metadata']), "Chunks and Metadata counts mismatch.")

if __name__ == '__main__':
    unittest.main()
"@
$testCode | Out-File -FilePath "tests/test_retrieval.py" -Encoding utf8