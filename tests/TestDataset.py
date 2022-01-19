import unittest
from StackOverflow16k import StackOverflow16k


class TestDataset(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_get_item(self):
        dataset = StackOverflow16k(root="../stack_oveflow_16k_train")
        print(dataset.item_lookup.iloc[1, 0])


if __name__ == '__main__':
    unittest.main()
