"""
基于 Llama-3.2-1B-Instruct 的本地关键词检测 (Function Call)
8个本地关键词:
  耳机拍照, 手机拍照, 上一曲, 下一曲, 音量增大, 音量减小, 暂停播放, 继续播放
"""
import json
import re
import time
import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

KEYWORD_TO_FN = {
    '耳机拍照': 'earphone_photo',
    '手机拍照': 'phone_photo',
    '上一曲': 'previous_track',
    '下一曲': 'next_track',
    '音量增大': 'volume_up',
    '音量减小': 'volume_down',
    '暂停播放': 'pause_playback',
    '继续播放': 'resume_playback',
}
FN_TO_KEYWORD = {v: k for k, v in KEYWORD_TO_FN.items()}

SYSTEM_PROMPT = """你是一个本地语音助手,负责识别用户语音指令并调用对应的function。

可用的function如下:
1. earphone_photo - 耳机按键触发拍照
2. phone_photo - 手机相机拍照(注意:"帮我拍照"等无设备说明的请求默认为手机拍照)
3. previous_track - 上一曲/上一首
4. next_track - 下一曲/切歌/跳过这首歌
5. volume_up - 音量增大/调大声音
6. volume_down - 音量减小/调小声音
7. pause_playback - 暂停播放/暂停音乐
8. resume_playback - 继续播放/开始播放音乐
9. unknown - 不属于以上任何一种

请严格按以下JSON格式输出,不要输出任何其他内容:
{"function": "<function_name>"}"""


def build_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def parse_function_call(output_text: str) -> str:
    text = output_text.strip()
    m = re.search(r'\{[^{}]*"function"\s*:\s*"([^"]+)"[^{}]*\}', text)
    if m:
        return m.group(1)
    for fn in FN_TO_KEYWORD.keys():
        if fn in text:
            return fn
    if 'unknown' in text.lower():
        return 'unknown'
    return 'parse_error'


def run_inference(model, tokenizer, text: str, device, max_new_tokens: int = 32) -> tuple:
    prompt = build_prompt(tokenizer, text)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - t0

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_function_call(output_text), output_text.strip(), elapsed


def evaluate(model, tokenizer, samples, device, sample_per_keyword=None, verbose=True):
    if sample_per_keyword:
        bucket = defaultdict(list)
        for s in samples:
            bucket[s['app']].append(s)
        selected = []
        for kw, items in bucket.items():
            zh = [i for i in items if i['lang'] == 'zh'][:sample_per_keyword]
            selected.extend(zh)
        samples = selected

    results = []
    correct_count = 0
    total_time = 0.0

    for i, sample in enumerate(samples):
        text = sample['text']
        expected_kw = sample['app']
        expected_fn = KEYWORD_TO_FN[expected_kw]

        predicted_fn, raw_output, elapsed = run_inference(model, tokenizer, text, device)
        is_correct = predicted_fn == expected_fn
        if is_correct:
            correct_count += 1
        total_time += elapsed

        result = {
            'id': sample['id'],
            'text': text,
            'lang': sample['lang'],
            'expected_keyword': expected_kw,
            'expected_function': expected_fn,
            'predicted_function': predicted_fn,
            'raw_output': raw_output,
            'correct': is_correct,
            'elapsed_sec': round(elapsed, 3)
        }
        results.append(result)

        if verbose:
            mark = 'OK ' if is_correct else 'XX '
            print(f'[{i+1}/{len(samples)}] {mark}{elapsed:.2f}s | text="{text}" | pred={predicted_fn} | expected={expected_fn}')

    accuracy = correct_count / len(samples) if samples else 0
    avg_time = total_time / len(samples) if samples else 0

    summary = {
        'total': len(samples),
        'correct': correct_count,
        'accuracy': round(accuracy, 4),
        'avg_inference_time_sec': round(avg_time, 3),
        'total_time_sec': round(total_time, 2)
    }

    per_kw = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        per_kw[r['expected_keyword']]['total'] += 1
        if r['correct']:
            per_kw[r['expected_keyword']]['correct'] += 1
    per_kw_summary = {
        kw: {
            'total': v['total'],
            'correct': v['correct'],
            'accuracy': round(v['correct']/v['total'], 4) if v['total'] else 0
        } for kw, v in per_kw.items()
    }

    return {
        'summary': summary,
        'per_keyword_accuracy': per_kw_summary,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/dfs/data/gemma/Llama-3.2-1B-Instruct')
    parser.add_argument('--data_path', default='./dataset/local_keywords_samples.json')
    parser.add_argument('--output_path', default='./keyword_eval_llama1b.json')
    parser.add_argument('--per_keyword', type=int, default=5,
                        help='每个关键词测试的样本数(0=全量)')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'设备: {device}')
    print(f'模型路径: {args.model_path}')
    print('加载 tokenizer 和 model...')

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print(f'模型加载耗时: {time.time()-t0:.2f}s')

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data['data']
    print(f'数据集样本数: {len(samples)}')

    per_kw = args.per_keyword if args.per_keyword > 0 else None
    print(f'\n开始评估 Llama-3.2-1B-Instruct (每关键词{per_kw if per_kw else "全部"}条)...\n')

    eval_result = evaluate(model, tokenizer, samples, device, sample_per_keyword=per_kw)

    print('\n========== 评估汇总 ==========')
    print(json.dumps(eval_result['summary'], ensure_ascii=False, indent=2))
    print('\n========== 各关键词准确率 ==========')
    for kw, stats in eval_result['per_keyword_accuracy'].items():
        print(f'  {kw}: {stats["correct"]}/{stats["total"]} = {stats["accuracy"]*100:.1f}%')

    output = {
        'model': 'Llama-3.2-1B-Instruct',
        'model_path': args.model_path,
        'device': device,
        'per_keyword_sample_size': per_kw,
        **eval_result
    }
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n详细结果已保存至: {args.output_path}')


if __name__ == '__main__':
    main()
