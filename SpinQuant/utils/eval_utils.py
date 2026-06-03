# coding=utf-8

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os

import torch
from tqdm import tqdm

from utils import model_utils


@torch.no_grad()
def evaluator(model, testenc, dev, args):
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = (
        input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
    )  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    
    # Handle nested config structure (e.g., Qwen2_5OmniThinkerConfig has text_config)
    if hasattr(model.config, 'text_config'):
        hidden_size = model.config.text_config.hidden_size
    else:
        hidden_size = model.config.hidden_size
    
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, hidden_size),
        dtype=dtype,
        device=dev,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": [], "position_ids": [], "position_embeddings": []}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "layer_type"):
                self.layer_type = module.layer_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"].append(kwargs.get("attention_mask"))
            cache["position_ids"].append(kwargs.get("position_ids"))
            cache["position_embeddings"].append(kwargs.get("position_embeddings"))
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        # Dump the layer input and output
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f"Dumped layer input and output to: {save_path}")

        for j in range(nbatches):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask[j],
                #  defined.
                position_ids=position_ids[j],
                position_embeddings=position_embeddings[j],
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)

    model.lm_head = model.lm_head.to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        
        if hasattr(model.config, "logits_scaling") and model.config.logits_scaling:
            lm_logits = lm_logits / model.config.logits_scaling
            
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")
    return ppl.item()

from torch.nn import CrossEntropyLoss

def llm_compute_loss_from_logits(outputs, labels, model_config=None):
    '''
    This function computes the loss from the logits and the labels passed
    '''
    # Get logits from outputs - handle both tuple and object formats
    if hasattr(outputs, 'logits'):
        lm_logits = outputs.logits
    elif isinstance(outputs, tuple):
        # Handle nested tuples - keep extracting first element until we get a tensor
        lm_logits = outputs[0]
        while isinstance(lm_logits, tuple):
            lm_logits = lm_logits[0]
    else:
        lm_logits = outputs
    
    # If lm_logits is still not a tensor (e.g., ModelOutput object), try to extract logits
    if not isinstance(lm_logits, torch.Tensor):
        if hasattr(lm_logits, 'logits'):
            lm_logits = lm_logits.logits
        elif hasattr(lm_logits, 'to_tuple'):
            # ModelOutput objects have to_tuple() method
            lm_logits = lm_logits.to_tuple()[0]
        else:
            # Last resort: try to convert to tuple and get first element
            try:
                lm_logits = tuple(lm_logits)[0]
            except:
                raise TypeError(f"Cannot extract logits from output of type {type(lm_logits)}")
    
    # Apply logits scaling if configured
    if model_config is not None and hasattr(model_config, "logits_scaling") and model_config.logits_scaling:
        lm_logits = lm_logits / model_config.logits_scaling
    
    # Keep on same device, convert to float32 for numerical stability
    shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    
    # Get vocab size from logits
    vocab_size = shift_logits.size(-1)
    
    # Clamp labels to valid range [0, vocab_size-1] or set to -100 if out of bounds
    # This prevents IndexError when labels contain token IDs outside vocabulary
    shift_labels_clamped = shift_labels.clone()
    out_of_bounds_mask = (shift_labels >= vocab_size) | (shift_labels < 0)
    shift_labels_clamped[out_of_bounds_mask] = -100

    # Compute the loss with ignore_index=-100 to skip invalid tokens
    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels_clamped.view(-1),
    )
    # Reshape and compute mean per sequence
    loss = loss.view(shift_labels.size(0), -1).mean(dim=1)
    neg_log_likelihood = loss.mean()
    return neg_log_likelihood

def llm_evaluate_ppl_with_dataloader(model, dataloader, num_batches=None, model_forward_kwargs={}):
    '''
    This function takes in a dada loader and a model and computes ppl score
    params:
    model: the model to evaluate
    dataloader: dataset loader
    num_batches: number of batches to run evaluation on
    '''
    from aimet_torch.utils import change_tensor_device_placement

    num_batches = num_batches if num_batches else len(dataloader)
    nlls=[]
    device=model.device
    dtype = model.dtype
    model_forward_kwargs = change_tensor_device_placement(model_forward_kwargs, device)

    for batch_id, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating")):
        if batch_id >= num_batches:
            break
        if "inputs_embeds" in batch:
            batch["input_ids"] = batch["labels"]
            batch["inputs_embeds"] = batch["inputs_embeds"].to(device=device,dtype=dtype)
            outputs = model(inputs_embeds=batch["inputs_embeds"], **model_forward_kwargs)
        else:
            batch["input_ids"] = batch["input_ids"].to(device)
            outputs = model(input_ids=batch["input_ids"], **model_forward_kwargs)

        nlls.append(llm_compute_loss_from_logits(outputs, batch["input_ids"], model.config))
        del outputs
    ppl = torch.exp(torch.stack(nlls).mean())
    return float(ppl)
