# Turn a PyTorch LoRA folder into the MLX adapter folder this app can load.
# The source adapter must have been trained on gemma-4-12B-it.

import argparse
import json
import os

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


# Read the rank and alpha out of a PEFT adapter config.
def read_peft_config(source_dir):
    path = os.path.join(source_dir, "adapter_config.json")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rank = int(raw["r"])
    alpha = float(raw["lora_alpha"])
    return rank, alpha


# The part of a weight name that names one layer piece, such as self_attn.q_proj.
def module_name(key):
    marker = ".layers."
    place = key.find(marker)
    if place == -1:
        return ""
    rest = key[place + len(marker):]
    index = 0
    while index < len(rest) and rest[index].isdigit():
        index = index + 1
    if index < len(rest) and rest[index] == ".":
        index = index + 1
    module = rest[index:]
    if module.endswith(".lora_a"):
        module = module[: -len(".lora_a")]
    elif module.endswith(".lora_b"):
        module = module[: -len(".lora_b")]
    return module


# Find the layer number inside a weight name. Names without a layer count as layer 0.
def layer_number(key):
    marker = ".layers."
    place = key.find(marker)
    if place == -1:
        return 0
    rest = key[place + len(marker):]
    digits = ""
    for char in rest:
        if char.isdigit():
            digits = digits + char
        else:
            break
    if digits == "":
        return 0
    return int(digits)


# Rename and flip one LoRA matrix so MLX can read it.
def convert_key(key, array):
    name = key
    prefix = "base_model.model."
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(".lora_A.weight"):
        name = name[: -len(".lora_A.weight")] + ".lora_a"
        return name, np.ascontiguousarray(np.transpose(array))
    if name.endswith(".lora_B.weight"):
        name = name[: -len(".lora_B.weight")] + ".lora_b"
        return name, np.ascontiguousarray(np.transpose(array))
    return "", array


# Read a PEFT folder and write adapters.safetensors plus adapter_config.json.
def convert_folder(source_dir, dest_dir):
    rank, alpha = read_peft_config(source_dir)
    weights_path = os.path.join(source_dir, "adapter_model.safetensors")
    converted = {}
    module_names = []
    highest_layer = 0
    with safe_open(weights_path, framework="numpy") as handle:
        for key in handle.keys():
            array = handle.get_tensor(key)
            new_name, new_array = convert_key(key, array)
            if new_name == "":
                continue
            converted[new_name] = new_array
            number = layer_number(new_name)
            if number > highest_layer:
                highest_layer = number
            piece = module_name(new_name)
            if piece != "" and piece not in module_names:
                module_names.append(piece)
    if len(converted) == 0:
        raise SystemExit("No LoRA weights were found in that folder.")
    if os.path.isdir(dest_dir) == False:
        os.makedirs(dest_dir)
    save_file(converted, os.path.join(dest_dir, "adapters.safetensors"))
    scale = 0.0
    if rank != 0:
        scale = alpha / float(rank)
    config_body = {}
    config_body["fine_tune_type"] = "lora"
    config_body["num_layers"] = highest_layer + 1
    lora_parameters = {}
    lora_parameters["rank"] = rank
    lora_parameters["scale"] = scale
    lora_parameters["dropout"] = 0.0
    # Only the pieces that were actually trained. Other pieces stay as the base model.
    if len(module_names) > 0:
        lora_parameters["keys"] = module_names
    config_body["lora_parameters"] = lora_parameters
    config_path = os.path.join(dest_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config_body, handle, indent=2)
        handle.write("\n")


# Command line: python convert_lora.py --source <peft folder> --dest adapters/brief
def main():
    parser = argparse.ArgumentParser(description="Convert a Gemma 4 12B PEFT LoRA into MLX format.")
    parser.add_argument("--source", required=True, help="Folder with adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--dest", required=True, help="Output folder, for example adapters/brief")
    args = parser.parse_args()
    convert_folder(args.source, args.dest)
    print("Wrote MLX adapter to " + args.dest)


if __name__ == "__main__":
    main()
