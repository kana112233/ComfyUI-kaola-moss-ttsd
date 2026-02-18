# ComfyUI-kaola-moss-ttsd

A ComfyUI custom node for [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD), a high-quality text-to-spoken dialogue generation model.

## Features
- **Text-to-Speech Generation**: Generate high-fidelity speech from text.
- **Multi-Speaker Support**: Supports standard `[S1]`, `[S2]` tags for dialogue generation.
- **Voice Cloning**: Clone voices using short reference audio clips.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes` directory:
    ```bash
    cd custom_nodes
    git clone https://github.com/Startgame/ComfyUI-kaola-moss-ttsd.git
    ```

2.  Install dependencies:
    ```bash
    cd ComfyUI-kaola-moss-ttsd
    pip install -r requirements.txt
    ```

    *Note: For GPU acceleration, ensure you have a CUDA-compatible PyTorch installed.*

## Usage

1.  From the node menu, select `Kaola` -> `MOSS-TTSD` -> `MOSS-TTSD Generation`.
2.  Input your dialogue text. Use tags like `[S1]` and `[S2]` to switch speakers.
    - Example: `[S1] Hello there! [S2] Hi! How are you?`
3.  (Optional) Connect `reference_audio_s1` and/or `reference_audio_s2` to provide voice reference clips for specific speakers.
4.  Connect the `AUDIO` output to a `PreviewAudio` or `SaveAudio` node.

## Models
The node will automatically download the [MOSS-TTSD-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) model from Hugging Face on the first run. You can configure a local path or different model ID in the node settings.

## License
Apache 2.0 (Inherited from MOSS-TTSD)
