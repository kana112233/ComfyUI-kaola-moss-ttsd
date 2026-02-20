
import unittest
import sys
import os
import torch
from unittest.mock import MagicMock

# Dynamically load the class to bypass relative import issues
def load_node_class():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, "nodes_moss_tts.py")
    
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Strip relative imports
    lines = code.split("\n")
    new_lines = []
    
    skip_import_block = False
    for line in lines:
        if "from .utils import" in line:
            new_lines.append("pass # stripped relative import")
            skip_import_block = True
        elif skip_import_block and line.strip().endswith(")"):
            skip_import_block = False
        elif skip_import_block:
            continue
        else:
            new_lines.append(line)
            
    modified_code = "\n".join(new_lines)
    
    # Setup mock environment
    module_scope = {
        "os": os,
        "torch": torch,
        "torchaudio": MagicMock(),
        "AutoModel": MagicMock(),
        "AutoProcessor": MagicMock(),
        "folder_paths": None,
        # Mock utils
        "get_model_path": MagicMock(),
        "auto_download": MagicMock(),
        "_patch_tqdm_for_comfyui": MagicMock(),
        "normalize_text": MagicMock(side_effect=lambda x: x), # simple passthrough
        "__file__": file_path
    }
    
    exec(modified_code, module_scope)
    
    return module_scope["MossTTSGenerate"]

MossTTSGenerate = load_node_class()

class TestMossTTSGenerate(unittest.TestCase):
    def setUp(self):
        self.node = MossTTSGenerate()
        self.device = "cpu"
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_processor.model_config.sampling_rate = 24000
        
        self.moss_tts_model = {
            "model": self.mock_model,
            "processor": self.mock_processor,
            "device": self.device,
            "dtype": torch.float32
        }

    def test_generate_basic(self):
        """Test basic text generation without reference audio."""
        text = "Hello world"
        
        # Mock processor behavior
        self.mock_processor.build_user_message.return_value = "mock_msg"
        self.mock_processor.return_value = {
            "input_ids": torch.zeros((1, 10)),
            "attention_mask": torch.zeros((1, 10))
        }
        
        # Mock decode
        mock_wav = torch.randn(1, 24000)
        mock_message = MagicMock()
        mock_message.audio_codes_list = [mock_wav]
        self.mock_processor.decode.return_value = [mock_message]

        self.node.generate(
            moss_tts_model=self.moss_tts_model,
            text=text,
            audio_temperature=1.0,
            audio_top_p=0.8,
            audio_top_k=50,
            audio_repetition_penalty=1.0,
            max_new_tokens=100,
            text_normalize=True
        )
        
        self.mock_processor.build_user_message.assert_called_with(text="Hello world")

    def test_generate_with_reference(self):
        """Test generation with reference audio (zero-shot cloning)."""
        text = "Cloning test"
        mock_audio_input = {
            "waveform": torch.randn(1, 1, 24000), # [batch, channels, samples]
            "sample_rate": 24000
        }
        
        # Mock encoding
        self.mock_processor.encode_audios_from_wav.return_value = ["mock_codes"]
        
        # Mock processor behavior
        self.mock_processor.return_value = {
            "input_ids": torch.zeros((1, 10)),
            "attention_mask": torch.zeros((1, 10))
        }
        mock_message = MagicMock()
        mock_message.audio_codes_list = [torch.randn(1, 24000)]
        self.mock_processor.decode.return_value = [mock_message]

        self.node.generate(
            moss_tts_model=self.moss_tts_model,
            text=text,
            audio_temperature=1.0,
            audio_top_p=0.8,
            audio_top_k=50,
            audio_repetition_penalty=1.0,
            max_new_tokens=100,
            text_normalize=True,
            reference_audio=mock_audio_input
        )
        
        # Check if reference audio was encoded
        self.mock_processor.encode_audios_from_wav.assert_called()
        # Check if reference codes were passed to prompt builder
        self.mock_processor.build_user_message.assert_called_with(text=text, reference="mock_codes")

if __name__ == '__main__':
    unittest.main()
