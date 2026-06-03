"""
Evaluate Qwen3-4B-SpinQuant-NVFP4 (compressed-tensors) on the 221 Chinese
keyword-detection samples via function-call style prompting.

Notes / environment shims:
- compressed-tensors 0.13.0 imports `transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS`,
  which doesn't exist in transformers 4.51. Since this NVFP4 checkpoint only quantizes
  Linear layers (kv_cache_scheme=null), the attention-masking registry is never exercised,
  so we inject a minimal compatible shim before importing compressed_tensors.
- The checkpoint's tokenizer_config stores `extra_special_tokens` as a list (incompatible
  with transformers 4.51), so we load the tokenizer from the base Qwen3-4B (identical vocab).
"""
import os, sys, types, json, time
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

# ---- shim: transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS ----
import transformers
if 'transformers.masking_utils' not in sys.modules:
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS  # noqa
    except Exception:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        _mod = types.ModuleType('transformers.masking_utils')
        # Reuse the same registry class so .register/__getitem__ behave consistently.
        _registry = type(ALL_ATTENTION_FUNCTIONS)()
        # seed common impls so __getitem__ won't KeyError if ever touched
        for k in ('eager', 'sdpa', 'flash_attention_2', 'flash_attention_3'):
            try:
                _registry.register(k, lambda *a, **k_: None)
            except Exception:
                pass
        _mod.ALL_MASK_ATTENTION_FUNCTIONS = _registry
        sys.modules['transformers.masking_utils'] = _mod
        transformers.masking_utils = _mod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keyword_detector_awq as kd


def main():
    mp = '/dfs/data/model_hub/models/Qwen3-4B-SpinQuant-NVFP4'
    base = '/dfs/data/model_hub/models/Qwen3-4B'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    print(f'模型路径 (SpinQuant NVFP4): {mp}')

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base)  # identical vocab, avoids broken tok config
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f'模型加载耗时: {time.time()-t0:.2f}s')

    data = json.load(open('./dataset/local_keywords_samples.json'))['data']
    zh = [s for s in data if s['lang'] == 'zh']
    print(f'中文样本数: {len(zh)}')

    res = kd.evaluate(model, tok, zh, model.device if hasattr(model, 'device') else device,
                      sample_per_keyword=None, verbose=True)

    print('\n========== 评估汇总 ==========')
    print(json.dumps(res['summary'], ensure_ascii=False, indent=2))
    print('\n========== 各关键词准确率 ==========')
    for k, v in res['per_keyword_accuracy'].items():
        print(f'  {k}: {v["correct"]}/{v["total"]} = {v["accuracy"]*100:.1f}%')

    out = {
        'model': 'Qwen3-4B-SpinQuant-NVFP4',
        'model_path': mp,
        'loader': 'compressed-tensors(nvfp4-pack-quantized)',
        'device': str(device),
        'eval_scope': 'all_zh_221',
        'per_keyword_sample_size': None,
        **res,
    }
    json.dump(out, open('./keyword_eval_spinquant_nvfp4_zh_full.json', 'w'),
              ensure_ascii=False, indent=2)
    print('\nsaved -> ./keyword_eval_spinquant_nvfp4_zh_full.json')


if __name__ == '__main__':
    main()
