
import sys
import os

# Add parent directory to sys.path to import nodes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nodes_voice_generator import MossSoundEffectLoadModel, MossSoundEffectGenerate
    print("SUCCESS: Imported MossSoundEffectLoadModel and MossSoundEffectGenerate")
except ImportError as e:
    print(f"FAILURE: Could not import nodes: {e}")
    sys.exit(1)

def test_instantiation():
    try:
        loader = MossSoundEffectLoadModel()
        generator = MossSoundEffectGenerate()
        print("SUCCESS: Instantiated nodes")
        
        # Check INPUT_TYPES
        loader_inputs = loader.INPUT_TYPES()
        generator_inputs = generator.INPUT_TYPES()
        
        print(f"Loader Inputs: {loader_inputs['required'].keys()}")
        print(f"Generator Inputs: {generator_inputs['required'].keys()}")
        
        if "moss_se_model" in generator_inputs["required"]:
             print("SUCCESS: Generator correctly expects 'moss_se_model'")
        else:
             print("FAILURE: Generator missing 'moss_se_model' input")

    except Exception as e:
        print(f"FAILURE: Instantiation or inspection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_instantiation()
