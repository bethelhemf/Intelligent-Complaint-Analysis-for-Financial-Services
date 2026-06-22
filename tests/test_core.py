import unittest
from src.rag_engine import CrediTrustRAG
class TestRAGQuality(unittest.TestCase):
    def test_relevance(self):
        rag = CrediTrustRAG()
        results = rag.retrieve("credit card", k=1)
        self.assertGreater(len(results), 0)
        print(f"Verified Retrieval Score: {results[0]['relevance_score']}")
if __name__ == '__main__':
    unittest.main()
