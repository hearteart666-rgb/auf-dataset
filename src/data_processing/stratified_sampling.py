"""
Stratified Random Sampling for Test Set Extraction

This script extracts a balanced test set from a categorized dataset using stratified random sampling.
It maintains the original category distribution while ensuring minimum representation for each category.
"""

import json
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed()

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Group by category
category_groups = defaultdict(list)
for item in data:
    category = item.get('category', 'N/A')
    category_groups[category].append(item)

# Display original distribution
print("\nOriginal data distribution:")
print("=" * 70)
print(f"{'Category':<30} {'Count':>10} {'Percentage':>10}")
print("-" * 70)
for category in sorted(category_groups.keys(), key=lambda x: len(category_groups[x]), reverse=True):
    count = len(category_groups[category])
    percentage = (count / len(data)) * 100
    print(f"{category:<30} {count:>10} {percentage:>9.2f}%")
print("=" * 70)

# Configuration
test_size = 400
min_samples_per_category = 2

# Calculate proportional distribution for test set
test_distribution = {}
total_allocated = 0

for category, items in category_groups.items():
    original_count = len(items)
    proportional_count = int(original_count / len(data) * test_size)
    allocated_count = max(min_samples_per_category, proportional_count) if original_count >= min_samples_per_category else min(original_count, proportional_count)
    test_distribution[category] = allocated_count
    total_allocated += allocated_count

# Adjust to ensure exact test_size
if total_allocated > test_size:
    difference = total_allocated - test_size
    sorted_categories = sorted(category_groups.keys(), key=lambda x: len(category_groups[x]), reverse=True)
    for category in sorted_categories:
        if difference <= 0:
            break
        if test_distribution[category] > min_samples_per_category:
            reduction = min(difference, test_distribution[category] - min_samples_per_category)
            test_distribution[category] -= reduction
            difference -= reduction

elif total_allocated < test_size:
    difference = test_size - total_allocated
    sorted_categories = sorted(category_groups.keys(), key=lambda x: len(category_groups[x]), reverse=True)
    for category in sorted_categories:
        if difference <= 0:
            break
        max_additional = len(category_groups[category]) - test_distribution[category]
        if max_additional > 0:
            addition = min(difference, max_additional)
            test_distribution[category] += addition
            difference -= addition

# Sample from each category
print("\nTest set allocation:")
print("=" * 90)
print(f"{'Category':<30} {'Original':>10} {'Test Set':>12} {'Test %':>10} {'Sample %':>10}")
print("-" * 90)

test_set = []
for category in sorted(test_distribution.keys(), key=lambda x: len(category_groups[x]), reverse=True):
    original_count = len(category_groups[category])
    test_count = test_distribution[category]
    test_percentage = (test_count / test_size) * 100
    within_category_percentage = (test_count / original_count) * 100

    print(f"{category:<30} {original_count:>10} {test_count:>12} {test_percentage:>9.2f}% {within_category_percentage:>9.2f}%")

    sampled = random.sample(category_groups[category], test_count)
    test_set.extend(sampled)

print("-" * 90)
print(f"{'Total':<30} {len(data):>10} {len(test_set):>12} {'100.00%':>10}")
print("=" * 90)

output_file = 'test.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

print(f"\nTest set saved: {output_file} ({len(test_set)} samples)")
