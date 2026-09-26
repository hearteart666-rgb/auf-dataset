#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import List, Dict
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image

PROMPT = """Your prompt here"""

# Model configuration
MODEL_PATH = "/path/to/your/model"
IMAGES_DIR = "/path/to/your/images"

def compress_image_if_needed(image_path: str) -> str:
    """Compress image if dimensions exceed 1024x1024, maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            max_size = 1024

            if width <= max_size and height <= max_size:
                return image_path

            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ext = os.path.splitext(image_path)[1] or '.jpg'
            temp_path = os.path.join(os.path.dirname(image_path), f"{base_name}_compressed_{max_size}{ext}")

            resized_img.save(temp_path, quality=95, optimize=True)
            return temp_path

    except Exception as e:
        print(f"Warning: Image compression failed {image_path}: {e}")
        return image_path

def initialize_vllm_model(model_path: str) -> tuple:
    """Initialize vLLM model and processor"""

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        max_model_len=16384,
        dtype="bfloat16",
        enforce_eager=True,
        limit_mm_per_prompt={"image": 2, "video": 1},
    )

    processor = AutoProcessor.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=8192,
        stop_token_ids=[],
    )

    return llm, processor, sampling_params

def read_output_json(json_file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_batch_vllm_true_batch(llm, processor, sampling_params, data: List[Dict], batch_size: int = 8) -> List[Dict]:
    results = []
    total = len(data)

    for start in range(0, total, batch_size):
        batch = data[start:start+batch_size]
        print(f"Processing batch {start+1}-{start+len(batch)} / {total}")

        llm_inputs_batch = []
        metas = []

        for data_item in batch:
            user_title = data_item.get('user_title', '')
            user_question = data_item.get('user_question', '')
            images = data_item.get('images', [])
            formatted_prompt = PROMPT.format(user_title=user_title, user_question=user_question)

            message_content = []
            if images:
                for image_item in images:
                    image_filename = image_item if isinstance(image_item, str) else image_item.get('filename', '')
                    if not image_filename:
                        continue
                    image_path = os.path.join(IMAGES_DIR, image_filename)
                    if os.path.exists(image_path):
                        compressed_image_path = compress_image_if_needed(image_path)
                        message_content.append({"type": "image", "image": "file:///" + compressed_image_path})
                        break
                    else:
                        print(f"Warning: Image not found: {image_path}")

            message_content.append({"type": "text", "text": formatted_prompt})
            messages = [{"role": "user", "content": message_content}]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            llm_inputs_batch.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
            })
            metas.append(data_item)

        try:
            outputs = llm.generate(llm_inputs_batch, sampling_params=sampling_params)
        except Exception as e:
            print(f"Batch inference failed: {e}")
            for data_item in metas:
                results.append({
                    'id': data_item.get('id', -1),
                    'user_title': data_item.get('user_title', ''),
                    'issue_type': data_item.get('issue_type', ''),
                    'response': None,
                    'error': f'Batch inference error: {str(e)}'
                })
            continue

        for out, data_item in zip(outputs, metas):
            try:
                gen_text = out.outputs[0].text
                results.append({
                    'id': data_item.get('id', -1),
                    'user_title': data_item.get('user_title', ''),
                    'issue_type': data_item.get('issue_type', ''),
                    'response': gen_text
                })
            except Exception as e:
                results.append({
                    'id': data_item.get('id', -1),
                    'user_title': data_item.get('user_title', ''),
                    'issue_type': data_item.get('issue_type', ''),
                    'response': None,
                    'error': f'Failed to parse output: {str(e)}'
                })

        successful_count = len([r for r in results if r.get('response')])
        print(f"Progress: {len(results)}/{total} processed, {successful_count} successful")

    return results

def main():
    """Main function"""
    output_json_path = "test.json"
    results_json_path = "classification_results.json"

    print(f"\nLoading data from {output_json_path}...")
    try:
        data = read_output_json(output_json_path)
    except FileNotFoundError:
        print(f"Error: File not found {output_json_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total items to process: {len(data)}")

    llm, processor, sampling_params = initialize_vllm_model(MODEL_PATH)
    print("vLLM model loaded successfully")

    print(f"\nStarting batch processing...")
    start_time = time.time()

    try:
        results = process_batch_vllm_true_batch(
            llm, processor, sampling_params,
            data=data,
            batch_size=8
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nSaving results to {results_json_path}...")
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        successful_count = len([r for r in results if r.get('response')])
        failed_count = len(results) - successful_count

        print("=" * 60)
        print("Processing completed!")
        print(f"Time elapsed: {duration:.1f}s")
        print(f"Successful: {successful_count}/{len(results)}")
        print(f"Failed: {failed_count}")
        if len(data) > 0:
            print(f"Speed: {len(data)/duration:.2f} items/s")
        print(f"Results saved to: {results_json_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print(f"\n\nProcess interrupted by user!")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()