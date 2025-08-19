import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import re
from tqdm import tqdm
import argparse
from EvalPromptFiles import sys_prompt_of_judger, judge_prompt


client = OpenAI(
    base_url="<YOUR_API_URL_FOR_JUDGER>",
    api_key="<YOUR_API_KEY_FOR_JUDGER>",
)


parser = argparse.ArgumentParser(description="MMESCI_JUDGER")
parser.add_argument("--JUDGE_MODEL", type=str, default="xxx")
parser.add_argument("--test_model_name", type=str, default="xxx")
parser.add_argument("--test_data_name", type=str, default="xxx")
parser.add_argument("--num_threads", type=int, default=16)
parser.add_argument("--max_retries", type=int, default=5)
parser.add_argument("--timeout", type=int, default=240)
parser.add_argument("--max_token_len", type=int, default=256)
parser.add_argument("--eval_mode", type=str, default='xxx')
parser.add_argument("--w_extraction", action='store_true', default=False)

args = parser.parse_args()

JUDGE_MODEL = args.JUDGE_MODEL
test_model_name = args.test_model_name
test_data_name = args.test_data_name
num_threads = args.num_threads
max_retries = args.max_retries
TIMEOUT = args.timeout
MAX_TOKEN_LEN = args.max_token_len
w_extraction = args.w_extraction
eval_mode = args.eval_mode


input_file = f'../eval_res-{eval_mode}/{test_data_name}/{test_model_name}.jsonl'
output_file = f'../eval_res-{eval_mode}/{test_data_name}/{test_model_name}_judged_by_{JUDGE_MODEL}.jsonl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
image_base_path = "../meta_data/image"

write_lock = threading.Lock()
processed_ids = set()
stop_processing_event = threading.Event()

def process_line(line):
    if stop_processing_event.is_set():
        return
    try:
        data = json.loads(line)
        entry_id = data['id']
        if entry_id in processed_ids:
            print(f"Entry {entry_id} already processed, skipping.")
            return

        raw_question_and_options = data['raw_question_and_options']
        task_gt = data['task_gt']
        model_answer = data['model_answer']
        if w_extraction:
            if '_zh' or '_en' in test_data_name:
                pattern = r"\\boxed\{(.*)\}|最终答案\s*(.*)|Final Answer\s*(.*)"
            elif '_fr' in test_data_name:
                pattern = r"\\boxed\{(.*)\}|Réponse finale\s*(.*)|Final Answer\s*(.*)"
            elif '_es' in test_data_name:
                pattern = r"\\boxed\{(.*)\}|Respuesta final\s*(.*)|Final Answer\s*(.*)"
            elif '_ja' in test_data_name:
                pattern = r"\\boxed\{(.*)\}|最終答え\s*(.*)|Final Answer\s*(.*)"
            else:
                pattern = r"\\boxed\{(.*)\}|最终答案\s*(.*)|Final Answer\s*(.*)"

            match = re.search(pattern, model_answer, re.DOTALL)
            if match:
                model_answer_extracted = next((group for group in [match.group(1), match.group(2), match.group(3)] if group), None)
                if model_answer_extracted:
                    model_answer_extracted = model_answer_extracted.strip()
                    data['model_answer_extracted'] = model_answer_extracted
                else:
                    model_answer_extracted = model_answer
            else:
                model_answer_extracted = model_answer
        else:
            model_answer_extracted = model_answer


        judge_content = [
            {
                "type": "text",
                "text": judge_prompt.format(
                    question=raw_question_and_options,
                    standard_answer=task_gt,
                    ai_respond=model_answer_extracted
                )
            }
        ]

        retry_attempts = 0
        
        while retry_attempts < max_retries:

            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt_of_judger},
                        {'role': 'user', 'content': judge_content}
                    ],
                    temperature=0,
                    timeout=TIMEOUT,
                    max_tokens=MAX_TOKEN_LEN,
                )
                judge_answer = response.choices[0].message.content.strip().lower()
                data['judge_answer'] = 0 if 'incorrect' in judge_answer else 1
                
                with write_lock:
                    with open(output_file, 'a', encoding='utf-8') as outfile:
                        json.dump(data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                    processed_ids.add(entry_id)
                break

            except Exception as e:
                retry_attempts += 1
                print(f"An error occurred while processing entry {entry_id} for {retry_attempts}/{max_retries}: {e}")

    except json.JSONDecodeError:
        print("Invalid JSON format, skipping line.")
        return


if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as outfile:
        for line in outfile:
            try:
                existing_data = json.loads(line)
                processed_ids.add(existing_data['id'])
            except json.JSONDecodeError:
                continue

filtered_lines = []
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        try:
            data = json.loads(line)
            entry_id = data['id']
            if entry_id in processed_ids:
                continue 
            filtered_lines.append(line)
        except json.JSONDecodeError:
            print("Invalid JSON format, skipping line.")
            continue

print(f"BEGIN TEST: {test_data_name}, MODEL: {test_model_name}, JUDGER: {JUDGE_MODEL}, MODE: {eval_mode}, Total lines to process after filtering: {len(filtered_lines)}")

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_line, line) for line in filtered_lines]
    with tqdm(total=len(filtered_lines), desc="Processing lines", unit="line") as pbar:
        for future in as_completed(futures):
            future.result() 
            pbar.update(1)

print("Processing complete.")
