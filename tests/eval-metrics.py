import unittest
import numpy as np
from src.rag_engine import CrediTrustRAG

class TestRAGQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = CrediTrustRAG()

    def test_retrieval_relevance(self):
        """Proof of Accuracy: Testing if 'Credit' queries find 'Credit' products."""
        query = "issue with my credit card billing"
        results = self.rag.retrieve(query, k=5)
        
        # Calculate Precision: how many results match the intent?
        matches = [r for r in results if "credit" in r['product'].lower()]
        precision = len(matches) / 5
        
        print(f"📈 Precision Score: {precision * 100}%")
        self.assertGreaterEqual(precision, 0.6, "Quality Gap: Retrieval precision is too low.")

if __name__ == "__main__":
    unittest.main()