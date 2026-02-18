import sys
import os
import unittest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class DescriptionTest(unittest.TestCase):
    def test_import_nodes(self):
        """Test if nodes.py can be imported and class exists."""
        try:
            from nodes import MossTTSDNode
            self.assertTrue(hasattr(MossTTSDNode, "INPUT_TYPES"))
            self.assertTrue(hasattr(MossTTSDNode, "RETURN_TYPES"))
            self.assertTrue(hasattr(MossTTSDNode, "FUNCTION"))
            print("Import successful.")
        except ImportError as e:
            self.fail(f"Failed to import MossTTSDNode: {e}")
        except Exception as e:
            self.fail(f"An error occurred: {e}")

    def test_instantiation(self):
        """Test if the node can be instantiated (without loading model)."""
        from nodes import MossTTSDNode
        try:
            node = MossTTSDNode()
            self.assertIsNotNone(node)
            print("Instantiation successful.")
        except Exception as e:
            self.fail(f"Failed to instantiate MossTTSDNode: {e}")

if __name__ == "__main__":
    unittest.main()
