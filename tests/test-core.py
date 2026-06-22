$testCode = @"
import unittest
import os
import sys
# Add root to path so we can import src
sys.path.append(os.getcwd())

from src.rag_engine import CrediTrustRAG
from src import config

class TestCrediTrustSystem(unittest.TestCase):
    def test_config_paths(self):
        """Verify all critical system paths are defined."""
        self.assertIsNotNone(config.LLM_MODEL)
        self.assertTrue(str(config.INDEX_PATH).endswith('.faiss'))

    def test_rag_initialization(self):
        """Verify the RAG engine can load the FAISS index."""
        try:
            rag = CrediTrustRAG()
            self.assertIsNotNone(rag.index)
            # Check if index has data
            self.assertGreater(rag.index.ntotal, 0)
        except Exception as e:
            self.fail(f"RAG Engine failed to load: {e}")

    def test_retrieval_output(self):
        """Verify the retriever returns structured metadata."""
        rag = CrediTrustRAG()
        results = rag.retrieve("test query", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn('complaint_id', results[0])
        self.assertIn('product', results[0])

if __name__ == '__main__':
    unittest.main()