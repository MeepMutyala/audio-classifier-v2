import torch
import sys
from pathlib import Path
import pandas as pd
from audio_utils import ESC50Dataset, ESC50Preprocessor, create_esc50_splits

model_name = sys.argv[1].lower()
if model_name == "liquids4":
    from liquidS4_audio import LiquidS4AudioClassifier as ModelClass
elif model_name == "mamba":
    from mamba_audio import MambaAudioClassifier as ModelClass
elif model_name == "vjepa2":
    from vjepa2_audio import VJEPA2AudioClassifier as ModelClass
else:
    raise ValueError("Unknown model: %s" % model_name)

MODEL_CLASSES = {
    "liquids4": LiquidS4AudioClassifier,
    "mamba": MambaAudioClassifier,
    "vjepa2": VJEPA2AudioClassifier
}

def load_model(model_name, num_classes=50, device="cpu"):
    model_cls = MODEL_CLASSES[model_name.lower()]
    # Adjust kwargs per model signature if needed
    if model_name.lower() == "liquids4":
        model = model_cls(numclasses=num_classes)
    elif model_name.lower() == "mamba":
        model = model_cls(numclasses=num_classes)
    elif model_name.lower() == "vjepa2":
        model = model_cls(numclasses=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
    return model.to(device)

def get_item_from_esc50(esc50_path, index=0, modeltype="sequence"):
    traindf, valdf, testdf = create_esc50_splits(esc50_path)
    testset = ESC50Dataset(testdf, esc50_path, modeltype=modeltype, augment=False)
    data, label = testset[index]
    return data.unsqueeze(0), label  # Batch size 1

def run_inference(model, input_tensor, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        pred = logits.argmax(dim=-1).item()
        return pred

def main(model_name, esc50_path, index=0, device="cpu", checkpoint_path=None):
    modeltype = "sequence" if model_name != "vjepa2" else "tubelet"
    model = load_model(model_name, checkpoint_path, device=device)
    x, y = get_item_from_esc50(esc50_path, index, modeltype=modeltype)
    pred = run_inference(model, x, device=device)
    correct = (pred == y)
    print(f"Model: {model_name}, True Label: {y.item()}, Predicted: {pred}, Correct: {correct}")

if __name__ == "__main__":
    # Example usage: python inference.py liquids4 /path/to/ESC-50 0 cuda /path/to/model.pth
    model_name = sys.argv[1]
    esc50_path = sys.argv[2]
    index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    checkpoint_path = sys.argv[5] if len(sys.argv) > 5 else None
    main(model_name, esc50_path, index, device, checkpoint_path)
