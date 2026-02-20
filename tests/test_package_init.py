
import sys
import os
import importlib

# Add the parent directory of the project root to sys.path
# Structure: /.../ComfyUI/custom_nodes/ComfyUI-kaola-moss-ttsd/tests/test_package_init.py
# We want to add /.../ComfyUI/custom_nodes/ to sys.path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # ComfyUI-kaola-moss-ttsd
custom_nodes_dir = os.path.dirname(project_root) # parent of project

sys.path.insert(0, custom_nodes_dir)

package_name = os.path.basename(project_root) # ComfyUI-kaola-moss-ttsd

print(f"Attempting to import package: {package_name}")

try:
    module = importlib.import_module(package_name)
    print(f"Successfully imported {package_name}")
    
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    
    print(f"Found {len(mappings)} nodes in NODE_CLASS_MAPPINGS")
    
    expected_new_nodes = [
        "MossSoundEffectLoadModel",
        "MossSoundEffectGenerate"
    ]
    
    for node in expected_new_nodes:
        if node in mappings:
            print(f"SUCCESS: Found {node}")
        else:
            print(f"FAILURE: Missing {node}")
            
except ImportError as e:
    print(f"FAILURE: Initial import failed: {e}")
except Exception as e:
    print(f"FAILURE: Runtime error during import: {e}")
    import traceback
    traceback.print_exc()
