"""
Multi-Modal Classification Script using OpenAI API

This script processes user feedback data with images and classifies them
into predefined categories using OpenAI's vision-capable language models.

Configuration:
  - Set API_KEY via environment variable:
    export OPENAI_API_KEY="your-key"
  - Or modify API_CONFIG directly in this file (line 58-68)
  - Images directory: ./data/images (relative path)
  - Input data: input_data.json
  - Output results: classification_results.json

Features:
  - Multi-modal processing (text + images)
  - Checkpoint-based resumption
  - Error handling and retry logic
  - OpenAI API integration
"""

import os
import json
import base64
import requests
import time
import mimetypes
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Modifiable PROMPT constant - Maintains original JSON output format
PROMPT = """Your prompt here"""

# Images directory path
IMAGES_DIR = ""

# Global variable: store image error information
IMAGE_ERRORS = []

# Load environment variables from .env file
load_dotenv()

# API Configuration - Modify according to your API provider
API_CONFIG = {
    # OpenAI API
    "openai": {
        "api_key": os.getenv('API_KEY'),
        "base_url": os.getenv('API_BASE_URL'),
        "model": os.getenv('MODEL'),
        "max_tokens": int(os.getenv('MAX_TOKENS')),
        "temperature": float(os.getenv('TEMPERATURE'))
    }
}

def save_image_errors(errors: List[Dict], error_file: str):
    """Save image error information to JSON file"""
    try:
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save image error file: {e}")

def encode_image_to_base64(image_path: str, data_item: Dict = None) -> str:
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode image {image_path}: {e}")
        # Record image encoding error
        error_record = {
            'error_type': 'encoding_failed',
            'image_path': image_path,
            'error_message': str(e),
            'timestamp': time.time()
        }
        if data_item:
            error_record['data_id'] = data_item.get('id', -1)
            error_record['data_id_str'] = str(data_item.get('id', ''))
            error_record['user_title'] = data_item.get('user_title', '')
        IMAGE_ERRORS.append(error_record)
        return None

def call_openai_api(prompt: str, image_paths: List[str], config: Dict, data_item: Dict = None) -> Optional[str]:
    """Call OpenAI API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }

    # Build message content - add images first, then text
    content = []

    # Add images
    for image_path in image_paths:
        base64_image = encode_image_to_base64(image_path, data_item)
        if base64_image:
            # Dynamically detect image type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = "image/jpeg"  # Default to jpeg

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": "high"
                }
            })

    # Add text prompt (after images)
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": config["temperature"]
    }

    try:
        response = requests.post(
            f"{config['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        return None

def call_api(prompt: str, image_paths: List[str], api_type: str = "openai", data_item: Dict = None) -> Optional[str]:
    """Call OpenAI API"""
    config = API_CONFIG.get(api_type)
    if not config:
        print(f"Unsupported API type: {api_type}")
        return None

    return call_openai_api(prompt, image_paths, config, data_item)

def process_data_item_with_api(data_item: Dict, api_type: str = "openai", retry_count: int = 3) -> Optional[str]:
    """Process single data item using API - Enhanced error handling"""
    try:
        # Extract data fields
        user_title = data_item.get('user_title', '')
        user_question = data_item.get('user_question', '')
        images = data_item.get('images', [])

        # Format PROMPT
        formatted_prompt = PROMPT.format(
            user_title=user_title,
            user_question=user_question
        )

        # Prepare image path list
        image_paths = []
        if images:
            for image_item in images:
                if isinstance(image_item, str):
                    # If string, use directly as filename
                    image_filename = image_item
                elif isinstance(image_item, dict):
                    # If dict, extract filename field
                    image_filename = image_item.get('filename', '')
                else:
                    # Skip other cases
                    continue

                if image_filename:
                    image_path = os.path.join(IMAGES_DIR, image_filename)
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                    else:
                        print(f"Warning: Image file not found: {image_path}")
                        # Record image not found error
                        error_record = {
                            'error_type': 'file_not_found',
                            'image_path': image_path,
                            'image_filename': image_filename,
                            'error_message': 'Image file does not exist',
                            'data_id': data_item.get('id', -1),
                            'user_title': data_item.get('user_title', ''),
                            'timestamp': time.time()
                        }
                        IMAGE_ERRORS.append(error_record)

        # Limit to maximum 2 images
        if len(image_paths) > 2:
            image_paths = image_paths[:2]

        # Call API with retry support and different error type handling
        last_error = None
        for attempt in range(retry_count):
            try:
                response = call_api(formatted_prompt, image_paths, api_type, data_item)

                if response:
                    return response
                else:
                    last_error = "API returned empty response"

                if attempt < retry_count - 1:
                    wait_time = (2 ** attempt) * 3 + 5
                    time.sleep(wait_time)

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                if attempt < retry_count - 1:
                    wait_time = 10 + (2 ** attempt) * 5
                    time.sleep(wait_time)

            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                if attempt < retry_count - 1:
                    wait_time = 8 + (2 ** attempt) * 4
                    time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e}"
                if e.response.status_code == 429:
                    if attempt < retry_count - 1:
                        wait_time = 20 + (3 ** attempt) * 5
                        time.sleep(wait_time)
                elif e.response.status_code >= 500:
                    if attempt < retry_count - 1:
                        wait_time = 10 + (2 ** attempt) * 3
                        time.sleep(wait_time)
                else:
                    break

            except Exception as e:
                last_error = f"Unknown error: {str(e)}"
                if attempt < retry_count - 1:
                    wait_time = (2 ** attempt) * 3
                    time.sleep(wait_time)

        print(f"  All retries failed, last error: {last_error}")
        return None

    except Exception as e:
        print(f"Error processing data item: {e}")
        return None

def read_output_json(json_file_path: str) -> List[Dict]:
    """Read output.json file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """Load checkpoint file for resumption"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            return checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint file: {e}")
            return None
    return None

def save_checkpoint(checkpoint_file: str, results: List[Dict], current_index: int, total_count: int):
    """Save checkpoint information for resumption"""
    checkpoint = {
        'results': results,
        'current_index': current_index,
        'total_count': total_count,
        'timestamp': time.time(),
        'total_processed': len(results)
    }
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save checkpoint file: {e}")

def get_processed_ids(results: List[Dict]) -> set:
    """Get set of processed IDs from results"""
    processed_ids = set()
    for result in results:
        record_id = result.get('id', '')
        if record_id:
            processed_ids.add(record_id)
    return processed_ids

def filter_unprocessed_data(data: List[Dict], processed_ids: set) -> List[Dict]:
    """Filter out unprocessed data"""
    unprocessed_data = []
    for item in data:
        record_id = item.get('id', '')
        if record_id not in processed_ids:
            unprocessed_data.append(item)
    return unprocessed_data

def process_batch_with_api(data: List[Dict], api_type: str = "openai", batch_delay: float = 1.0,
                          checkpoint_file: str = None, existing_results: List[Dict] = None,
                          image_error_file: str = None) -> List[Dict]:
    """Batch process data items (via API) - Supports checkpoint resumption, returns raw_response directly"""
    results = existing_results.copy() if existing_results else []
    total = len(data)

    for i, data_item in enumerate(data, 1):
        try:
            # Process data item
            response = process_data_item_with_api(data_item, api_type)

            if response:
                # Save raw response directly without tag parsing
                result_item = {
                    'id': data_item.get('id', -1),
                    'url': data_item.get('url', ''),
                    'user_title': data_item.get('user_title', ''),
                    'issue_type': data_item.get('issue_type', ''),
                    'raw_response': response
                }
                results.append(result_item)
            else:
                result_item = {
                    'id': data_item.get('id', -1),
                    'url': data_item.get('url', ''),
                    'user_title': data_item.get('user_title', ''),
                    'issue_type': data_item.get('issue_type', ''),
                    'raw_response': None,
                    'error': 'API call failed'
                }
                results.append(result_item)

            # Save checkpoint after each item (API calls are slow, need frequent saves)
            if checkpoint_file:
                save_checkpoint(checkpoint_file, results, i, total)

            # Save image error information
            if image_error_file and IMAGE_ERRORS:
                save_image_errors(IMAGE_ERRORS, image_error_file)

            # Show progress every 10 items
            if i % 10 == 0 or i == total:
                successful_count = len([r for r in results if r.get('raw_response')])
                print(f"Progress: {i}/{total} items processed, {successful_count} successful")

            # Delay between batches to avoid API rate limits
            if i < total:
                time.sleep(batch_delay)

        except KeyboardInterrupt:
            print(f"\nProgram interrupted by user! Processed {i-1}/{total} items")
            if checkpoint_file:
                save_checkpoint(checkpoint_file, results, i-1, total)
            # Save image error information
            if image_error_file and IMAGE_ERRORS:
                save_image_errors(IMAGE_ERRORS, image_error_file)
            raise
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            # Save error record
            error_item = {
                'id': data_item.get('id', -1),
                'url': data_item.get('url', ''),
                'user_title': data_item.get('user_title', ''),
                'issue_type': data_item.get('issue_type', ''),
                'raw_response': None,
                'error': f'Processing exception: {str(e)}'
            }
            results.append(error_item)

            if checkpoint_file:
                save_checkpoint(checkpoint_file, results, i, total)

            # Save image error information
            if image_error_file and IMAGE_ERRORS:
                save_image_errors(IMAGE_ERRORS, image_error_file)

            # Continue with next item
            continue

    return results

def main():
    """Main function - Supports checkpoint resumption"""
    # Configuration parameters
    output_json_path = "input_data.json"
    results_json_path = "classification_results.json"
    checkpoint_file = "classification_checkpoint.json"
    image_error_file = "image_errors.json"

    api_type = "openai"

    # API call interval (seconds) to avoid rate limiting
    batch_delay = 1.0

    print("=" * 60)
    print("API Classification Inference (with checkpoint resumption)")
    print(f"API type: {api_type}")
    print(f"Call interval: {batch_delay}s")
    print("=" * 60)

    # Check for checkpoint file
    checkpoint = load_checkpoint(checkpoint_file)
    resume_from_checkpoint = False
    existing_results = []
    processed_urls = set()

    if checkpoint:
        print(f"\nCheckpoint file found!")
        checkpoint_time = time.ctime(checkpoint['timestamp'])
        print(f"Checkpoint time: {checkpoint_time}")
        print(f"Processed: {checkpoint['total_processed']} items")

        # Ask whether to resume from checkpoint
        response = input("Resume from checkpoint? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            resume_from_checkpoint = True
            existing_results = checkpoint.get('results', [])
            processed_ids = get_processed_ids(existing_results)
            print(f"Resuming from checkpoint, {len(existing_results)} items already processed")
        else:
            print("Starting from scratch")
            # Backup old checkpoint file
            backup_file = f"classification_checkpoint_backup_{int(time.time())}.json"
            if os.path.exists(checkpoint_file):
                os.rename(checkpoint_file, backup_file)
                print(f"Old checkpoint backed up as: {backup_file}")
    else:
        print("No checkpoint file found, starting from scratch")

    # Check API configuration
    if api_type not in API_CONFIG:
        print(f"Unsupported API type: {api_type}")
        return

    if "your-" in API_CONFIG[api_type]["api_key"]:
        print(f"Warning: Please configure your {api_type} API key in API_CONFIG first!")
        return

    print(f"\nReading data...")
    try:
        all_data = read_output_json(output_json_path)
    except FileNotFoundError:
        print(f"Error: File not found: {output_json_path}")
        return
    except Exception as e:
        print(f"Error: Failed to read file: {e}")
        return

    # Filter data based on resumption
    if resume_from_checkpoint:
        unprocessed_data = filter_unprocessed_data(all_data, processed_ids)
        print(f"Total data: {len(all_data)}, Processed: {len(existing_results)}, Remaining: {len(unprocessed_data)}")
        data_to_process = unprocessed_data
    else:
        data_to_process = all_data
        print(f"Total items to process: {len(data_to_process)}")

    if not data_to_process:
        print("All data already processed!")
        return

    # Batch process data
    print(f"\nStarting API batch processing... (interval: {batch_delay}s)")
    start_time = time.time()

    try:
        results = process_batch_with_api(
            data_to_process,
            api_type,
            batch_delay,
            checkpoint_file,
            existing_results,
            image_error_file
        )

        end_time = time.time()
        duration = end_time - start_time

        # Save final results to JSON file
        print(f"\nSaving final results to {results_json_path}...")
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save image error information (final version)
        if IMAGE_ERRORS:
            save_image_errors(IMAGE_ERRORS, image_error_file)

        # Statistics
        successful_count = len([r for r in results if r.get('raw_response')])
        failed_count = len(results) - successful_count

        print("=" * 60)
        print("Processing complete!")
        print(f"Processing time: {duration:.1f}s")
        print(f"Successfully processed: {successful_count}/{len(results)} items")
        print(f"Failed: {failed_count} items")
        if len(data_to_process) > 0:
            print(f"Average speed: {len(data_to_process)/duration:.2f} items/s")
        print(f"Results saved to: {results_json_path}")
        print("=" * 60)

        # Delete checkpoint file after completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Checkpoint file deleted")

        # Show failed items
        if failed_count > 0:
            print("\nFailed items:")
            failed_items = [r for r in results if not r.get('raw_response')]
            for i, item in enumerate(failed_items[:5], 1):
                title = item.get('user_title', 'Unknown')[:50]
                error = item.get('error', 'Unknown error')
                print(f"  {i}. {title}... - {error}")
            if len(failed_items) > 5:
                print(f"  ... and {len(failed_items)-5} more failed items")

    except KeyboardInterrupt:
        print(f"\n\nProgram interrupted by user!")
        print("Checkpoint information saved, you can resume from checkpoint next time")
        print(f"Current progress: {len(existing_results) + len([r for r in locals().get('results', []) if r])}/{len(all_data)} items processed")
        # Save image error information
        if IMAGE_ERRORS:
            save_image_errors(IMAGE_ERRORS, image_error_file)

    except Exception as e:
        print(f"\nProgram error: {e}")
        print("Checkpoint information saved, you can resume from checkpoint next time")
        # Save image error information
        if IMAGE_ERRORS:
            save_image_errors(IMAGE_ERRORS, image_error_file)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()