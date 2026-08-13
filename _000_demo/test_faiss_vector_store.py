import unittest

from common.faiss_vector_store import FaissEntityStore, entity_text


class FaissEntityStoreTest(unittest.TestCase):
    def test_entity_text_prefers_type_and_name(self):
        self.assertEqual(entity_text({"type": "Herb", "name": "黄芪"}), "Herb: 黄芪")
        self.assertEqual(entity_text({"name": "黄芪"}), "黄芪")

    def test_empty_question_returns_no_match(self):
        store = FaissEntityStore("unused", "unused.index")
        self.assertEqual(store.search(""), [])


if __name__ == "__main__":
    unittest.main()
