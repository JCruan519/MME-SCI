import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import get_message, get_raw_question_and_options
import argparse
import traceback
from openai import OpenAI
from datasets import load_dataset


client = OpenAI(
    base_url="<YOUR_API_URL>",
    api_key="<YOUR_API_KEY>",
)

parser = argparse.ArgumentParser(description="MMESCI_VLLM_API")
parser.add_argument("--VLM_MODEL", type=str, default="xxx")
parser.add_argument("--test_file_name", type=str, default="xxx")
parser.add_argument("--num_threads", type=int, default=6)
parser.add_argument("--max_retries", type=int, default=1)
parser.add_argument("--timeout", type=int, default=240)
parser.add_argument("--max_token_len", type=int, default=256)
parser.add_argument("--eval_mode", type=str, default='xxx')
parser.add_argument("--w_eval_prompt", action='store_true', default=False)

args = parser.parse_args()

VLM_MODEL = args.VLM_MODEL
test_file_name = args.test_file_name
num_threads = args.num_threads
max_retries = args.max_retries
TIMEOUT = args.timeout
MAX_TOKEN_LEN = args.max_token_len
eval_mode = args.eval_mode
w_eval_prompt = args.w_eval_prompt


if w_eval_prompt:
    if eval_mode == 'image_text_zh':
        from EvalPromptFiles import give_answer_after_final_answer_prompt_zh as eval_prompt
    elif eval_mode == 'image_text_en':
        from EvalPromptFiles import give_answer_after_final_answer_prompt_en as eval_prompt
    elif eval_mode == 'image_text_fr':
        from EvalPromptFiles import give_answer_after_final_answer_prompt_fr as eval_prompt
    elif eval_mode == 'image_text_ja':
        from EvalPromptFiles import give_answer_after_final_answer_prompt_ja as eval_prompt
    elif eval_mode == 'image_text_es':
        from EvalPromptFiles import give_answer_after_final_answer_prompt_es as eval_prompt
    elif eval_mode == 'image':
        from EvalPromptFiles import give_answer_after_final_answer_prompt_visiononly_zh as eval_prompt


input_file = f'../meta_data/{test_file_name}.parquet'
output_file = f'../eval_res-{eval_mode}/{test_file_name}/{VLM_MODEL}.jsonl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

write_lock = threading.Lock()
processed_ids = set()
stop_processing_event = threading.Event()


def process_entry(entry):
    if stop_processing_event.is_set():
        return
    try:
        entry_id = entry['id']
        if entry_id in processed_ids:
            print(f"Entry {entry_id} already processed, skipping.")
            return
        
        try:
            all_contents = get_message(entry, eval_mode)

            if not all_contents: 
                return

            if w_eval_prompt:
                all_contents.append({"type": "text", "text": eval_prompt})

            messages = [{"role": "user", "content": all_contents}]

        except Exception as e:
            print(f"An error occurred on id of {entry_id}")
            traceback.print_exc()

        retry_attempts = 0

        res_data = {}
        res_data['id'] = entry['id']
        res_data['course'] = entry['course']
        res_data['knowledge-source'] = entry.get('knowledge-source', 'NONE')
        res_data['raw_question_and_options'] = get_raw_question_and_options(entry)
        res_data['task_gt'] = entry['answer']
        
        while retry_attempts < max_retries:
            try:
                response = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    temperature=0,
                    timeout=TIMEOUT,
                    max_tokens=MAX_TOKEN_LEN,
                )
                model_answer = response.choices[0].message.content
                res_data['model_answer'] = model_answer
                with write_lock:
                    with open(output_file, 'a', encoding='utf-8') as outfile:
                        json.dump(res_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                    processed_ids.add(entry_id)
                break

            except Exception as e:
                retry_attempts += 1
                print(f"An error occurred while processing entry {entry_id} for {retry_attempts}/{max_retries}: {e}")

    except Exception as e:
        print(f"Error processing entry: {e}")
        traceback.print_exc()
        return


if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as outfile:
        for line in outfile:
            try:
                existing_data = json.loads(line)
                processed_ids.add(existing_data['id'])
            except json.JSONDecodeError:
                continue

try:
    dataset = load_dataset("parquet", data_files=input_file)
    split = dataset[next(iter(dataset))]
    filtered_entries = [entry for entry in split if entry['id'] not in processed_ids]
    
    print(f"BEGIN TEST: {test_file_name}, MODEL: {VLM_MODEL}, MODE: {eval_mode}, Total entries to process after filtering: {len(filtered_entries)}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_entry, entry) for entry in filtered_entries]
        with tqdm(total=len(filtered_entries), desc="Processing entries", unit="entry") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    print("Processing complete.")

except Exception as e:
    print(f"Error reading Parquet file: {e}")
    traceback.print_exc()
    stop_processing_event.set()
    