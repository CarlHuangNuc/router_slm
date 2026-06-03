# coding=utf-8

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import tqdm
from transformers import PreTrainedModel

from utils import quant_utils, utils

class GPTQ:
    """GPTQ quantization algorithm implementation.
    
    This class implements the GPTQ (Gradient-based Post-Training Quantization) algorithm
    for quantizing neural network weights.
    
    Args:
        layer: The layer to be quantized (typically nn.Linear)
    """
    
    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer: Optional[quant_utils.WeightQuantizer] = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Add a batch of data to compute the Hessian matrix.
        
        Args:
            inp: Input tensor
            out: Output tensor (not used in current implementation)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        static_groups: bool = False,
        export_to_et: bool = False,
    ) -> None:
        """Perform fast quantization using the GPTQ algorithm.
        
        Args:
            blocksize: Block size for processing columns
            percdamp: Percentage of dampening to apply to the Hessian
            groupsize: Group size for quantization (-1 for per-channel)
            actorder: Whether to use activation order
            static_groups: Whether to use static groups
            export_to_et: Whether to export for ExecuTorch
        """
        if self.quantizer is None:
            raise ValueError("Quantizer must be set before calling fasterquant")
        W = self.layer.weight.data.clone().float()
        Scale = self.layer.weight.data.clone().float()
        W_int = self.layer.weight.data.clone().float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        groups = []
        if static_groups:
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        perm = None
        invperm = None
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Scale1 = torch.zeros_like(W1).to(Scale.dtype)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q, int_weight, scale = self.quantizer.fake_quantize(w.unsqueeze(1))
                q_flat = q.flatten()
                Q1[:, i] = q_flat
                W_int1[:, i] = int_weight.flatten()
                Scale1[:, i] = scale.flatten()

                Losses1[:, i] = (w - q_flat) ** 2 / d**2

                err1 = (w - q_flat) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Scale[:, i1:i2] = Scale1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", W_int.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.error(
                f"NaN detected in weights after quantization. "
                f"Quantizer bits: {self.quantizer.bits}, "
                f"scale shape: {self.quantizer.scale.shape}, "
                f"zero shape: {self.quantizer.zero.shape}"
            )
            raise ValueError("NaN detected in weights after quantization")

    def free(self) -> None:
        """Free memory by deleting cached tensors."""
        self.H = None
        if hasattr(self, 'Losses'):
            self.Losses = None
        if hasattr(self, 'Trace'):
            self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)

@torch.no_grad()
def gptq_fwrd(
    model: PreTrainedModel,
    dataloader: torch.utils.data.DataLoader,
    dev: torch.device,
    args,
) -> Dict[str, quant_utils.WeightQuantizer]:
    """Perform GPTQ quantization on the model.
    
    This function applies GPTQ quantization to all layers in the model using
    calibration data from the dataloader.
    
    Args:
        model: The pretrained model to quantize
        dataloader: DataLoader containing calibration data
        dev: Device to run quantization on
        args: Arguments containing quantization configuration
        
    Returns:
        Dictionary mapping layer names to their quantizers
    """
    logging.info("-----GPTQ Quantization-----")

    # Qwen2_5OmniThinkerConfig might not have use_cache directly
    use_cache = False
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        use_cache = model.config.text_config.use_cache
        model.config.text_config.use_cache = False
        
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # Use model's max_position_embeddings if available, otherwise default to 4096
    if hasattr(model, 'seqlen'):
        max_seq_len = model.seqlen
    else:
        max_seq_len = getattr(model.config, 'max_position_embeddings', 4096)
    
    hidden_size = 0
    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
        print(f"DEBUG: Found hidden_size in model.config: {hidden_size}")
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_size = model.config.text_config.hidden_size
        print(f"DEBUG: Found hidden_size in model.config.text_config: {hidden_size}")
    
    if hidden_size == 0:
        raise AttributeError("Could not find hidden_size in config")
        
    inps = torch.zeros(
        (args.nsamples, max_seq_len, hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None, "inps_list": []}

    class Catcher(nn.Module):
        """Helper module to capture intermediate activations."""
        
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module
            if hasattr(module, "layer_type"):
                self.layer_type = module.layer_type

        def forward(self, inp: torch.Tensor, **kwargs) -> torch.Tensor:
            # Store inputs in a list to handle variable sequence lengths
            cache["inps_list"].append(inp)
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError("Catcher forward pass - expected behavior")

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    # Reconstruct inps from the captured list with actual sequence lengths
    inps_list = cache["inps_list"]
    if len(inps_list) > 0:
        # Get the actual sequence length from the first captured input
        actual_seq_len = inps_list[0].shape[1]
        inps = torch.zeros(
            (args.nsamples, actual_seq_len, hidden_size), dtype=dtype, device=dev
        )
        for i, inp in enumerate(inps_list[:args.nsamples]):
            inps[i] = inp
    
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quantizers: Dict[str, quant_utils.WeightQuantizer] = {}
    
    # Define the sequential order for quantizing layers (standard model)
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        [
            "mlp.up_proj.module",
            "mlp.gate_proj.module",
            "shared_mlp.input_linear.module",
        ],
        [
            "mlp.down_proj.module",
            "shared_mlp.output_linear.module",
        ],
    ]

    # Define sequential order for LoRA-adapted models
    sequential_lora = [
        [
            "self_attn.k_proj.base_layer.module",
            "self_attn.v_proj.base_layer.module",
            "self_attn.q_proj.base_layer.module",
            "self_attn.k_proj.lora_A.default.module",
            "self_attn.k_proj.lora_B.default.module",
            "self_attn.v_proj.lora_A.default.module",
            "self_attn.v_proj.lora_B.default.module",
            "self_attn.q_proj.lora_A.default.module",
            "self_attn.q_proj.lora_B.default.module",
        ],
        [
            "self_attn.o_proj.base_layer.module",
            "self_attn.o_proj.lora_A.default.module",
            "self_attn.o_proj.lora_B.default.module",
        ],
        [
            "mlp.up_proj.base_layer.module",
            "mlp.gate_proj.base_layer.module",
            "mlp.up_proj.lora_A.default.module",
            "mlp.up_proj.lora_B.default.module",
            "mlp.gate_proj.lora_A.default.module",
            "mlp.gate_proj.lora_B.default.module",
        ],
        [
            "mlp.down_proj.base_layer.module",
            "mlp.down_proj.lora_A.default.module",
            "mlp.down_proj.lora_B.default.module",
        ],
    ]

    # Determine if LoRA is being used by checking args or model structure
    use_lora = hasattr(args, 'lora_path') and args.lora_path is not None and args.lora_path != ""
    
    if use_lora:
        logging.info("Using LoRA sequential order for quantization")
        sequential = sequential_lora
    else:
        logging.info("Using standard sequential order for quantization")
    
    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear, quant_utils.QuantizeLinear])
        
        # Initialize gptq dict at layer level to avoid UnboundLocalError
        gptq: Dict[str, GPTQ] = {}
        
        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

            if not subset:
                continue
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                
                # Determine quantization bits for this layer
                layer_weight_bits = args.w_bits
                layer_weight_sym = not args.w_asym
                
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                    
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

            def add_batch(name: str):
                """Create a hook function to add batch data to GPTQ."""
                def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers[f"model.layers.{i}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.cpu()
        del layer
        if gptq:  # Only delete if gptq was populated
            del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        model.config.text_config.use_cache = use_cache
        
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(
    model: PreTrainedModel,
    dev: torch.device,
    args,
    custom_layers: Optional[List[nn.Module]] = None,
) -> Dict[str, quant_utils.WeightQuantizer]:
    """Perform Round-to-Nearest (RTN) quantization on the model.
    
    RTN is a simpler quantization method that rounds weights to the nearest
    quantized value without using calibration data.
    
    Args:
        model: The pretrained model to quantize
        dev: Device to run quantization on
        args: Arguments containing quantization configuration
        custom_layers: Optional list of custom layers to quantize
        
    Returns:
        Dictionary mapping layer names to their quantizers
    """
    if custom_layers:
        layers = custom_layers
    else:
        layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers: Dict[str, quant_utils.WeightQuantizer] = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RTN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(
            layer, layers=[torch.nn.Linear, torch.nn.Embedding]
        )

        for name in subset:
            # Determine quantization configuration for this layer
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
            if args.export_to_et:
                # All per-channel 8 bits for ExecuTorch export
                layer_weight_bits = 8
                w_groupsize = -1
                
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not args.w_asym,
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
            )
            
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
            
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)
                
            quantizers[f"model.layers.{i}.{name}"] = quantizer.cpu()
            
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
