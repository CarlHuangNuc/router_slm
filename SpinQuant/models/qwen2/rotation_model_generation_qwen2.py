# coding=utf-8

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import Qwen2TokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
import datasets
from utils.data_utils import CustomJsonDataset

log: Logger = utils.get_logger("spinquant")


from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils

def rotation_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    # Rotate the weights
    if args.rotate:
        log.info("Rotate the weights from {}".format(args.optimized_rotation_path))
        fuse_norm_utils.fuse_layer_norms(model)

        rotation_utils.rotate_model_no_had(model, args)
        # utils.cleanup_memory(verbos=True)
        
    # quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
    # qlayers = quant_utils.find_qlayers(model)
    # for name in qlayers:
    #     if "down_proj" in name:
    #         had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    #         qlayers[name].online_full_had = True
    #         qlayers[name].had_K = had_K
    #         qlayers[name].K = K
    #         qlayers[name].fp32_had = args.fp32_had

    return model
        

def model_rotation() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = Qwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    model = rotation_model(ptq_args, model, model_args)

    # Solve the issue that " You are trying to save a non contiguous tensor: `model.layers.0.self_attn.v_proj.weight` which is not allowed"
    for param in model.parameters(): 
        param.data = param.data.contiguous()

    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Start to load tokenizer...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    
    with torch.no_grad():
        prompt = "Human:请写下你对过去一年的总结，并提出自己在未来一年的目标和计划。\n\nAssistant:"
        inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to("cuda")
        model.to("cuda")
        generate_ids_inf = model.generate(inputs.input_ids, do_sample=False, max_length=training_args.model_max_length)
        print("="*100)
        print(tokenizer.batch_decode(generate_ids_inf, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        print("="*100)

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))

    ## Save ratation model.
    if ptq_args.save_no_had_model_path is not None:
        model.save_pretrained(ptq_args.save_no_had_model_path)
        tokenizer.save_pretrained(ptq_args.save_no_had_model_path)

    dist.barrier()


if __name__ == "__main__":
    model_rotation()
