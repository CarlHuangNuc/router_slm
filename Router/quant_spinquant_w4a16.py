"""
SpinQuant-style W4A16 quantization for Qwen3-0.6B + keyword-detection accuracy eval.

Pipeline (weight-only W4A16, activations kept fp16):
  1. Untie lm_head from embeddings (tied model) so output path is independent.
  2. Fuse each RMSNorm learnable scale into the following Linear weights, set norm=1.
     (Required so an orthogonal rotation of the residual stream is a true no-op.)
  3. Apply an orthogonal R1 rotation (normalized Hadamard, size=hidden=1024) to the
     residual stream, absorbed into weights  -- this is SpinQuant's rotation init.
       read-projections (q,k,v,gate,up,lm_head): W' = W @ Q
       write-projections (o,down):               W' = Q^T @ W
       input embedding:                          E' = E @ Q
     With Q orthogonal the fp32 forward output is mathematically unchanged.
  4. Group-wise 4-bit RTN fake-quant of all decoder Linear weights (group_size=128,
     asymmetric). Weights stored dequantized to fp16 -> simulates W4A16 inference.

Then reuse keyword_detector.evaluate to compare accuracy vs the fp16 baseline.
"""
import argparse
import json
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import keyword_detector as kd


# ---------------- Hadamard rotation ----------------
def hadamard_matrix(n: int, device, dtype=torch.float64) -> torch.Tensor:
    assert (n & (n - 1)) == 0, f"n={n} must be a power of 2 for Hadamard"
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / (n ** 0.5)  # normalized -> orthonormal


# ---------------- RMSNorm fusion ----------------
def fuse_norm_into_linears(norm_module, linears):
    """Fold RMSNorm weight vector into following linear weights, reset norm to 1."""
    w = norm_module.weight.data.to(torch.float64)
    for lin in linears:
        # lin.weight: [out, in]; scale each input column j by w[j]
        lin.weight.data = (lin.weight.data.to(torch.float64) * w.unsqueeze(0)).to(lin.weight.dtype)
    norm_module.weight.data = torch.ones_like(norm_module.weight.data)


# ---------------- 4-bit group-wise RTN fake quant ----------------
@torch.no_grad()
def fake_quant_4bit(weight: torch.Tensor, group_size: int = 128, bits: int = 4) -> torch.Tensor:
    """Asymmetric per-group RTN. weight [out, in], groups along in-dim."""
    orig_dtype = weight.dtype
    W = weight.data.to(torch.float32)
    out_f, in_f = W.shape
    assert in_f % group_size == 0, f"in_dim {in_f} not divisible by group {group_size}"
    qmax = (1 << bits) - 1
    Wg = W.reshape(out_f, in_f // group_size, group_size)
    wmin = Wg.min(dim=-1, keepdim=True).values
    wmax = Wg.max(dim=-1, keepdim=True).values
    scale = (wmax - wmin).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round((Wg - wmin) / scale), 0, qmax)
    deq = (q * scale + wmin).reshape(out_f, in_f)
    return deq.to(orig_dtype)


def quant_error(orig: torch.Tensor, deq: torch.Tensor) -> float:
    o = orig.to(torch.float32)
    return (torch.norm(o - deq.to(torch.float32)) / torch.norm(o).clamp(min=1e-8)).item()


def get_linears(model):
    """Return decoder linears split into (read_projs, write_projs) plus lm_head."""
    layers = model.model.layers
    read, write = [], []
    for lyr in layers:
        a = lyr.self_attn
        read += [a.q_proj, a.k_proj, a.v_proj]
        write += [a.o_proj]
        m = lyr.mlp
        read += [m.gate_proj, m.up_proj]
        write += [m.down_proj]
    return read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='./qwen3-0.6b')
    ap.add_argument('--data_path', default='./dataset/local_keywords_samples.json')
    ap.add_argument('--output_path', default='./keyword_eval_w4a16_result.json')
    ap.add_argument('--per_keyword', type=int, default=100)
    ap.add_argument('--group_size', type=int, default=128)
    ap.add_argument('--rotate', action='store_true', help='apply Hadamard R1 rotation')
    ap.add_argument('--no-quant', dest='quant', action='store_false')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device} rotate={args.rotate} quant={args.quant} group_size={args.group_size}')

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map=device)
    model.eval()
    hidden = model.config.hidden_size

    # 1. Untie lm_head if tied
    if model.config.tie_word_embeddings or (model.lm_head.weight.data_ptr() ==
                                            model.model.embed_tokens.weight.data_ptr()):
        print('untying lm_head from embeddings')
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.data.clone())
        model.config.tie_word_embeddings = False

    if args.rotate:
        print('fusing RMSNorm scales into following linears...')
        for lyr in model.model.layers:
            fuse_norm_into_linears(lyr.input_layernorm,
                                   [lyr.self_attn.q_proj, lyr.self_attn.k_proj, lyr.self_attn.v_proj])
            fuse_norm_into_linears(lyr.post_attention_layernorm,
                                   [lyr.mlp.gate_proj, lyr.mlp.up_proj])
        fuse_norm_into_linears(model.model.norm, [model.lm_head])

        print(f'applying Hadamard R1 rotation (size={hidden})...')
        Q = hadamard_matrix(hidden, device=device)  # float64, orthonormal
        read, write = get_linears(model)
        # read-projections: W' = W @ Q
        for lin in read + [model.lm_head]:
            lin.weight.data = (lin.weight.data.to(torch.float64) @ Q).to(lin.weight.dtype)
        # write-projections: W' = Q^T @ W
        for lin in write:
            lin.weight.data = (Q.t() @ lin.weight.data.to(torch.float64)).to(lin.weight.dtype)
        # input embedding: E' = E @ Q
        E = model.model.embed_tokens.weight
        E.data = (E.data.to(torch.float64) @ Q).to(E.dtype)
        del Q
        torch.cuda.empty_cache()

    if args.quant:
        print(f'fake-quantizing decoder linears to 4-bit (group={args.group_size})...')
        read, write = get_linears(model)
        targets = read + write  # quantize attn+mlp linears (activations stay fp16)
        errs = []
        for lin in targets:
            orig = lin.weight.data.clone()
            deq = fake_quant_4bit(lin.weight.data, group_size=args.group_size, bits=4)
            errs.append(quant_error(orig, deq))
            lin.weight.data = deq
        print(f'  quantized {len(targets)} linears | mean rel-err={sum(errs)/len(errs):.4f} '
              f'| max={max(errs):.4f}')

    # ---- evaluate on Chinese keyword samples ----
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data['data']
    per_kw = args.per_keyword if args.per_keyword > 0 else None
    print(f'\nevaluating ({len(samples)} total, per_keyword={per_kw}, zh only)...\n')
    t0 = time.time()
    res = kd.evaluate(model, tok, samples, device, sample_per_keyword=per_kw, verbose=False)
    print(f'eval wall-time {time.time()-t0:.1f}s')

    print('\n========== summary ==========')
    print(json.dumps(res['summary'], ensure_ascii=False, indent=2))
    print('\n========== per-keyword ==========')
    for kw, st in res['per_keyword_accuracy'].items():
        print(f'  {kw}: {st["correct"]}/{st["total"]} = {st["accuracy"]*100:.1f}%')

    out = {'model': 'Qwen3-0.6B',
           'config': {'rotate': args.rotate, 'quant': args.quant,
                      'bits': 4, 'group_size': args.group_size, 'scheme': 'W4A16'},
           **res}
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nsaved -> {args.output_path}')


if __name__ == '__main__':
    main()
