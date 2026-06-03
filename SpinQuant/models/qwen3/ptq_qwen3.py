# coding=utf-8

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import datetime
from logging import Logger
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_qwen3 import Qwen3ForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from eval_utils import gptq_utils, rotation_utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)
from peft import PeftModel

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
    model = Qwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    # Load LoRA adapter if specified
    if hasattr(ptq_args, 'lora_path') and ptq_args.lora_path:
        log.info(f"Loading LoRA adapter from: {ptq_args.lora_path}")
        peftmodel = PeftModel.from_pretrained(
            model,
            ptq_args.lora_path,
            torch_dtype=torch.bfloat16
        ).eval()
        model = peftmodel.base_model.model

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )

    model = ptq_model(ptq_args, model, model_args, tokenizer)
    for param in model.parameters(): 
        param.data = param.data.contiguous()
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")

    model.config.use_cache = False

    testloader = None
    try:
        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
    except Exception as e:
        log.warning(f"Skipping wikitext2 loading (offline?): {e}")

    # Optional: Test generation with custom prompt
    if hasattr(ptq_args, 'test_prompt') and ptq_args.test_prompt:
        with torch.no_grad():
            log.info("Running test generation with custom prompt")
            inputs = tokenizer(ptq_args.test_prompt, padding=True, truncation=True, return_tensors="pt").to("cuda")
            model.to("cuda")
            generate_ids = model.generate(inputs.input_ids, do_sample=False, max_length=training_args.model_max_length)
            output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            log.info(f"Generated output:\n{output_text}")

    if testloader is not None:
        try:
            dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
        except Exception as e:
            log.warning(f"Skipping wiki2 ppl eval: {e}")

    # ---- Keyword-detection function-call accuracy on the quantized model ----
    if local_rank == 0:
        try:
            import sys, json as _json
            router_dir = "/dfs/data/Qoder/router_slm/Router"
            if router_dir not in sys.path:
                sys.path.insert(0, router_dir)
            import keyword_detector as kd
            model.config.use_cache = True
            model.cuda().eval()
            data_path = router_dir + "/dataset/local_keywords_samples.json"
            with open(data_path, "r", encoding="utf-8") as f:
                samples = _json.load(f)["data"]
            log.info("Running keyword-detection eval on quantized SpinQuant model...")
            res = kd.evaluate(model, tokenizer, samples, "cuda",
                              sample_per_keyword=100, verbose=False)
            log.info("SpinQuant W4F16 summary: {}".format(
                _json.dumps(res["summary"], ensure_ascii=False)))
            for kw, st in res["per_keyword_accuracy"].items():
                log.info("  {}: {}/{} = {:.1f}%".format(
                    kw, st["correct"], st["total"], st["accuracy"] * 100))
            out = {"model": "Qwen3-0.6B-SpinQuant-W4F16(trained-rotation)",
                   "config": {"w_bits": 4, "a_bits": 16, "scheme": "W4F16",
                              "rotation": "learned (Cayley SGD, 100 steps)"},
                   **res}
            out_path = "/dfs/data/Qoder/router_slm/Router/keyword_eval_spinquant_w4f16_trained.json"
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(out, f, ensure_ascii=False, indent=2)
            log.info("saved -> {}".format(out_path))
        except Exception as e:
            import traceback
            log.warning("keyword eval failed: {}\n{}".format(e, traceback.format_exc()))

    dist.barrier()


if __name__ == "__main__":
    train()
