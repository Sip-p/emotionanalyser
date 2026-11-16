
import h5py
import json

model_path = "Artifacts/BiGRU_model.h5"

with h5py.File(model_path, "r") as f:

    print("Keras version used during training:")
    print(f.attrs.get("keras_version"))

    print("\nBackend:")
    print(f.attrs.get("backend"))

    model_config = f.attrs.get("model_config")

    if isinstance(model_config, bytes):
        model_config = model_config.decode("utf-8")

    config = json.loads(model_config)

    print("\nInspecting model layers...\n")

    layers = config["config"]["layers"]

    for layer in layers:
        if layer["class_name"] in ["InputLayer", "Embedding"]:
            print("Layer:", layer["class_name"])
            print("Configuration:")
            print(layer["config"])
            print("-" * 60)
