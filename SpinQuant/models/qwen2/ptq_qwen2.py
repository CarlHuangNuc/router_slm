# coding=utf-8

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def train() -> None:
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

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    
    # Try to load tokenizer from original model path if input_model is a converted model
    tokenizer_path = model_args.input_model
    if "no_had_model" in model_args.input_model:
        # Use the original model path for tokenizer
        original_model_path = '/prj/qct/aicechina_scratch/ruzhongl/llm/qwen2_5_omini/llm_notebook_ce/Ali/qwen2.5-3b-omni/helper_scripts/llm/sub_model/llm'
        log.info(f"Loading tokenizer from original model path: {original_model_path}")
        tokenizer_path = original_model_path
    
    # Use slow tokenizer to avoid tokenizer.json parsing issues
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
        trust_remote_code=True,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    dist.barrier()


if __name__ == "__main__":
    train()
