import argparse
import os
import shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def merge_r3(r3_path: str, safetensors_path: str, output_path: str) -> None:
    print(f"Loading R3 from: {r3_path}")
    r3: torch.Tensor = torch.load(r3_path, map_location="cpu")
    print(f"  R3 shape: {r3.shape}, dtype: {r3.dtype}")

    # safetensors does not support float64 — cast to float32
    # nn.Linear stores weight as [out, in]; the loading code does copy_(r3_weight.T),
    # so the safetensors entry must be R3.T
    r3_f32 = r3.T.to(torch.float32).contiguous()

    print(f"Loading safetensors from: {safetensors_path}")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    print(f"  Loaded {len(tensors)} tensors")

    layer_nums = sorted(
        set(
            int(k.split(".layers.")[1].split(".")[0])
            for k in tensors
            if ".layers." in k
        )
    )
    print(f"  Detected {len(layer_nums)} layers: {layer_nums[0]}..{layer_nums[-1]}")

    added = []
    for layer_idx in layer_nums:
        for proj in ("q", "k"):
            key = f"model.layers.{layer_idx}.self_attn.{proj}_R3.weight"
            tensors[key] = r3_f32.clone()
            added.append(key)

    print(f"\nAdded {len(added)} R3 weight tensors")

    if os.path.abspath(output_path) == os.path.abspath(safetensors_path):
        backup = safetensors_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(safetensors_path, backup)
            print(f"Backed up original to: {backup}")
        else:
            print(f"Backup already exists: {backup}")

    print(f"Saving merged safetensors to: {output_path}")
    save_file(tensors, output_path)
    print(f"Done. Total tensors in output: {len(tensors)}")

    print("Verifying output...")
    with safe_open(output_path, framework="pt", device="cpu") as f:
        out_keys = list(f.keys())
        r3_keys_found = [k for k in out_keys if "R3" in k]
    print(f"  Total keys: {len(out_keys)}")
    print(f"  R3 keys found: {len(r3_keys_found)}")
    assert len(r3_keys_found) == len(layer_nums) * 2, (
        f"Expected {len(layer_nums) * 2} R3 keys, got {len(r3_keys_found)}"
    )
    print("  Verification passed ✓")


def main() -> None:
    # 1. Duplicate directory
    src_dir = "output/qwen3_1.7b/no_had_model_random_H"
    dst_dir = "output/qwen3_1.7b/no_had_model_random_H_ssd"
    
    if os.path.exists(dst_dir):
        print(f"Destination directory {dst_dir} already exists. Removing it...")
        shutil.rmtree(dst_dir)
    
    print(f"Duplicating {src_dir} to {dst_dir}...")
    shutil.copytree(src_dir, dst_dir)
    print("Duplication complete.")

    parser = argparse.ArgumentParser(description="Merge R3 rotation into safetensors")
    parser.add_argument(
        "--r3_path",
        default="/prj/qct/aicechina_scratch/weihuan/eval/qwen3-llm/llm_notebook_ce/Ali/qwen3-1.7b/example1/dataset/R3.pt",
    )
    # Use the new duplicated directory for the default paths
    parser.add_argument(
        "--safetensors_path",
        default=os.path.join(dst_dir, "model.safetensors"),
    )
    parser.add_argument(
        "--output_path",
        default=os.path.join(dst_dir, "model.safetensors"),
    )
    args = parser.parse_args()
    
    merge_r3(args.r3_path, args.safetensors_path, args.output_path)


if __name__ == "__main__":
    main()
