"""
SpinQuant W4A16 quantization on Qwen3-0.6B, reusing the OFFICIAL SpinQuant modules.

This driver imports SpinQuant's own code (under ./SpinQuant):
  - utils.fuse_norm_utils.fuse_layer_norms     : fold RMSNorm scale into linears
  - utils.hadamard_utils.random_hadamard_matrix: SpinQuant R1 init (randomized Hadamard)
  - eval_utils.rotation_utils.*                : exact weight-absorbed R1 / R2 rotations
  - eval_utils.gptq_utils.GPTQ                 : error-compensated GPTQ (Cholesky)
  - utils.quant_utils.WeightQuantizer          : group-wise weight quantizer

Scheme: W4A16 (4-bit weights, fp16 activations).
Rotation: R1 (residual stream) + R2 (o/v within head) -- both exactly absorbed into
weights, so the fp32 forward is unchanged. R4(down) and QK-Hadamard are skipped because
they require an ONLINE Hadamard on activations (only meaningful for activation quant).

Then evaluate on the 221 Chinese keyword-detection samples and compare to the
fp16 baseline (89.14%).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SPINQUANT_DIR = Path(__file__).parent.parent / "SpinQuant"
sys.path.insert(0, str(SPINQUANT_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- official SpinQuant modules ----
from utils import fuse_norm_utils  # noqa: E402
from utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear  # noqa: E402
from eval_utils import rotation_utils  # noqa: E402
from eval_utils.gptq_utils import GPTQ  # noqa: E402
from utils.quant_utils import WeightQuantizer  # noqa: E402

# reuse our keyword eval harness
import keyword_detector as kd  # noqa: E402


@torch.inference_mode()
def rotate_weight_only(model, R1):
    """Apply R1 (residual) + R2 (o/v) using SpinQuant's exact weight-absorbed routines.
    Skips R4(down)/QK which need online activation Hadamard."""
    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)

    rotation_utils.rotate_embeddings(model, R1)
    rotation_utils.rotate_head(model, R1)
    for layer in model.model.layers:
        rotation_utils.rotate_attention_inputs(layer, R1)
        rotation_utils.rotate_attention_output(layer, R1)
        rotation_utils.rotate_mlp_input(layer, R1)
        # mlp output: only R1 absorption (NO R4 online hadamard) -> exact
        W = layer.mlp.down_proj
        dt = W.weight.data.dtype
        W.weight.data = torch.matmul(
            R1.T, W.weight.data.to("cuda", torch.float64)
        ).to("cpu", dt)
        # R2: exact head-wise Hadamard absorbed between v_proj and o_proj
        rotation_utils.rotate_ov_proj(layer, num_heads, head_dim, R2=None)


def get_calib_data(tokenizer, data_path, n_samples, seqlen, device):
    """Build calibration batches from the Chinese keyword dataset texts."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [s["text"] for s in data["data"] if s.get("lang") == "zh"]
    # render with chat template (matches inference distribution)
    rendered = []
    for t in texts:
        rendered.append(kd.build_prompt(tokenizer, t))
    batches = []
    for i in range(min(n_samples, len(rendered))):
        enc = tokenizer(rendered[i], return_tensors="pt", truncation=True,
                        max_length=seqlen)
        batches.append(enc.input_ids.to(device))
    return batches


@torch.no_grad()
def gptq_quantize(model, calib_batches, w_bits, groupsize, device, percdamp=0.01):
    """Sequential per-layer GPTQ, robust to transformers version (stores full kwargs)."""
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    dtype = next(model.parameters()).dtype
    inps, kw_cache = [], {}

    class Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kwargs):
            inps.append(hidden_states.detach())
            # capture everything else (attention_mask, position_ids, position_embeddings...)
            kw_cache.update({k: v for k, v in kwargs.items()})
            raise ValueError

    layers[0] = Catcher(layers[0]).to(device)
    for b in calib_batches:
        try:
            model(b)
        except ValueError:
            pass
    layers[0] = layers[0].mod
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    fwd_kwargs = {k: v for k, v in kw_cache.items()
                  if k in ("attention_mask", "position_ids", "position_embeddings")}

    outs = [None] * len(inps)
    seq = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
           ["self_attn.o_proj"],
           ["mlp.gate_proj", "mlp.up_proj"],
           ["mlp.down_proj"]]
    rel_errs = []
    for i in range(len(layers)):
        layer = layers[i].to(device)
        named = dict(layer.named_modules())
        for group in seq:
            gptq = {}
            for name in group:
                lin = named[name]
                g = GPTQ(lin)
                g.quantizer = WeightQuantizer()
                g.quantizer.configure(w_bits, perchannel=True, sym=False,
                                      mse=False, weight_groupsize=groupsize)
                gptq[name] = g
            handles = []
            for name in group:
                def mk(nm):
                    def h(_, inp, out):
                        gptq[nm].add_batch(inp[0].data, out.data)
                    return h
                handles.append(named[name].register_forward_hook(mk(name)))
            for j in range(len(inps)):
                layer(inps[j], **fwd_kwargs)
            for h in handles:
                h.remove()
            for name in group:
                w_before = named[name].weight.data.clone().float()
                gptq[name].fasterquant(percdamp=percdamp, groupsize=groupsize,
                                       actorder=True, static_groups=False)
                w_after = named[name].weight.data.float()
                rel = (torch.norm(w_before - w_after) /
                       torch.norm(w_before).clamp(min=1e-8)).item()
                rel_errs.append(rel)
                gptq[name].free()
        for j in range(len(inps)):
            outs[j] = layer(inps[j], **fwd_kwargs)[0]
        layers[i] = layer.cpu() if False else layer  # keep on GPU (0.6B fits)
        inps = outs
        outs = [None] * len(inps)
        print(f"  layer {i} done", flush=True)
    print(f"  GPTQ mean rel-err={sum(rel_errs)/len(rel_errs):.4f} max={max(rel_errs):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./qwen3-0.6b")
    ap.add_argument("--data_path", default="./dataset/local_keywords_samples.json")
    ap.add_argument("--output_path", default="./keyword_eval_spinquant_w4a16.json")
    ap.add_argument("--per_keyword", type=int, default=100)
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--no_quant", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} rotate={args.rotate} quant={not args.no_quant} "
          f"w_bits={args.w_bits} groupsize={args.groupsize}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(device)
    model.eval()

    # untie lm_head (SpinQuant incompatible with tied embeddings)
    if model.config.tie_word_embeddings or (
        model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()):
        print("untying lm_head")
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.data.clone())
        model.config.tie_word_embeddings = False

    if args.rotate:
        print("fuse_layer_norms (official SpinQuant)...")
        fuse_norm_utils.fuse_layer_norms(model)
        print("R1 randomized-Hadamard rotation + R2 (official SpinQuant)...")
        R1 = random_hadamard_matrix(model.config.hidden_size, "cuda")
        rotate_weight_only(model, R1.to(torch.float64))
        del R1
        torch.cuda.empty_cache()
        model = model.to(device)

    if not args.no_quant:
        print(f"GPTQ W{args.w_bits}A16 quantization (group={args.groupsize}, "
              f"{args.nsamples} calib samples)...")
        calib = get_calib_data(tok, args.data_path, args.nsamples, args.seqlen, device)
        t0 = time.time()
        gptq_quantize(model, calib, args.w_bits, args.groupsize, device)
        print(f"  GPTQ took {time.time()-t0:.1f}s")
        model = model.to(device)

    # ---- evaluate ----
    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)["data"]
    per_kw = args.per_keyword if args.per_keyword > 0 else None
    print(f"\nevaluating (zh, per_keyword={per_kw})...")
    t0 = time.time()
    res = kd.evaluate(model, tok, samples, device, sample_per_keyword=per_kw, verbose=False)
    print(f"eval wall-time {time.time()-t0:.1f}s")

    print("\n========== summary ==========")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    print("\n========== per-keyword ==========")
    for kw, st in res["per_keyword_accuracy"].items():
        print(f"  {kw}: {st['correct']}/{st['total']} = {st['accuracy']*100:.1f}%")

    out = {"model": "Qwen3-0.6B",
           "method": "SpinQuant(R1+R2)+GPTQ" if args.rotate else "GPTQ-only",
           "config": {"rotate": args.rotate, "quant": not args.no_quant,
                      "w_bits": args.w_bits, "groupsize": args.groupsize,
                      "scheme": "W4A16"},
           **res}
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.output_path}")


if __name__ == "__main__":
    main()
