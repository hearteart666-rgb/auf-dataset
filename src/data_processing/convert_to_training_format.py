import json
import os
import sys

PROMPT_TEMPLATE = """Your prompt here"""


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_to_training_format(filtered_data, original_data):
    """
    Convert distilled data to training format

    Args:
        filtered_data: Filtered distillation results
        original_data: Original data with complete information

    Returns:
        List of training format data
    """
    # Create index mapping by ID
    original_data_map = {}
    for item in original_data:
        item_id = item.get('id', '')
        if item_id:
            original_data_map[item_id] = item

    training_data = []

    for filtered_item in filtered_data:
        item_id = filtered_item.get('id', '')
        raw_response = filtered_item.get('raw_response', '')

        if not raw_response:
            print(f"Warning: Skipping item without raw_response: id={item_id}")
            continue

        original_item = original_data_map.get(item_id)
        if not original_item:
            print(f"ERROR: ID not found in original data: id={item_id}")
            print("Aborting conversion.")
            sys.exit(1)

        user_title = filtered_item.get('user_title', '')
        user_question = original_item.get('user_question', '')
        images_data = original_item.get('images', [])

        # Process images
        images_list = []
        for img_item in images_data:
            if isinstance(img_item, dict):
                filename = img_item.get('filename', '')
            elif isinstance(img_item, str):
                filename = img_item
            else:
                continue

            if filename:
                image_path = f"images/{filename}"

                if not os.path.exists(image_path):
                    print(f"ERROR: Image file not found: {image_path}")
                    print(f"Item ID: {item_id}")
                    sys.exit(1)

                images_list.append(image_path)

        image_tags = " ".join(["<image>"] * len(images_list))

        user_content = f"{image_tags} {PROMPT_TEMPLATE}".format(
            user_title=user_title,
            user_question=user_question
        ).strip()

        training_item = {
            "messages": [
                {
                    "content": user_content,
                    "role": "user"
                },
                {
                    "content": raw_response,
                    "role": "assistant"
                }
            ],
            "images": images_list
        }

        training_data.append(training_item)

    return training_data


def main():
    """Main function"""
    filtered_file = "classification_results.json"
    original_file = "data/train.json"
    output_file = "training_data.json"

    if not os.path.exists(filtered_file):
        print(f"Error: File not found {filtered_file}")
        return

    if not os.path.exists(original_file):
        print(f"Error: File not found {original_file}")
        return

    try:
        filtered_data = load_json(filtered_file)
        original_data = load_json(original_file)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    try:
        training_data = convert_to_training_format(filtered_data, original_data)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        save_json(training_data, output_file)
        print(f"Converted {len(training_data)} items -> {output_file}")
    except Exception as e:
        print(f"Failed to save: {e}")
        return


if __name__ == "__main__":
    main()
