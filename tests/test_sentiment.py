import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import hash_password, predict_sentiment, verify_password


class SentimentTests(unittest.TestCase):
    def test_positive_review(self):
        result = predict_sentiment("this movie is amazing and excellent")
        self.assertGreater(result, 0.5)

    def test_negative_review(self):
        result = predict_sentiment("this movie is boring and terrible")
        self.assertLess(result, 0.5)

    def test_password_hash_and_verify(self):
        hashed_password = hash_password("secret123")
        self.assertTrue(verify_password(hashed_password, "secret123"))
        self.assertFalse(verify_password(hashed_password, "wrongpassword"))


if __name__ == "__main__":
    unittest.main()
