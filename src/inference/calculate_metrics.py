#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate classification metrics (Accuracy, Precision, Recall, F1-Score)
by comparing predicted categories with ground truth categories
"""

import json
import re
from typing import Dict, List


def normalize_category(category: str) -> str:
    """
    Normalize category name to lowercase letters only
    Example: "Accessibility/Camera" -> "accessibilitycamera"
    """
    if not category:
        return ""
    return re.sub(r'[^a-zA-Z]', '', category).lower()


def extract_final_category(response: str) -> str:
    """
    Extract category from <FinalCategory> tag in response
    """
    if not response:
        return ""

    pattern = r'<FinalCategory>\s*(.*?)\s*</FinalCategory>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        category = match.group(1).strip()
        return normalize_category(category)

    return ""


def load_ground_truth(file_path: str) -> Dict[str, str]:
    """
    Load ground truth data
    Returns: dict mapping id -> normalized category
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ground_truth = {}
    for item in data:
        item_id = item.get('id', '')
        category = item.get('category', '')
        if item_id and category:
            ground_truth[str(item_id)] = normalize_category(category)

    return ground_truth


def load_predictions(file_path: str) -> Dict[str, str]:
    """
    Load predictions
    Returns: dict mapping id -> extracted normalized category
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = {}
    for item in data:
        item_id = item.get('id', '')
        response = item.get('response', '')
        if item_id:
            predictions[str(item_id)] = extract_final_category(response)

    return predictions


def calculate_metrics(ground_truth: Dict[str, str],
                     predictions: Dict[str, str]) -> Dict:
    """
    Calculate classification metrics
    """
    matched_pairs = []
    unmatched_ids = []

    for item_id in ground_truth:
        if item_id in predictions:
            true_label = ground_truth[item_id]
            pred_label = predictions[item_id]
            if pred_label:
                matched_pairs.append((true_label, pred_label))
            else:
                unmatched_ids.append(item_id)
        else:
            unmatched_ids.append(item_id)

    if not matched_pairs:
        return {
            "error": "No matched pairs found",
            "total_ground_truth": len(ground_truth),
            "total_predictions": len(predictions),
            "matched": 0
        }

    # Calculate accuracy
    correct = sum(1 for true, pred in matched_pairs if true == pred)
    total = len(matched_pairs)
    accuracy = correct / total if total > 0 else 0

    # Get all unique labels
    all_labels = set()
    for true_label, pred_label in matched_pairs:
        all_labels.add(true_label)
        all_labels.add(pred_label)

    # Calculate per-class metrics
    class_metrics = {}
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for label in sorted(all_labels):
        tp = sum(1 for true, pred in matched_pairs if true == label and pred == label)
        fp = sum(1 for true, pred in matched_pairs if true != label and pred == label)
        fn = sum(1 for true, pred in matched_pairs if true == label and pred != label)
        tn = sum(1 for true, pred in matched_pairs if true != label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        support = tp + fn

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }

        tp_total += tp
        fp_total += fp
        fn_total += fn

    # Calculate macro-averaged metrics
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(class_metrics)
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(class_metrics)
    macro_f1 = sum(m["f1_score"] for m in class_metrics.values()) / len(class_metrics)

    # Calculate weighted-averaged metrics
    total_support = sum(m["support"] for m in class_metrics.values())
    weighted_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_f1 = sum(m["f1_score"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0

    # Calculate micro-averaged metrics (same as accuracy for multi-class)
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1_score": micro_f1
        },
        "per_class_metrics": class_metrics,
        "num_classes": len(all_labels),
        "unmatched_ids": len(unmatched_ids)
    }


def save_metrics_report(metrics: Dict, output_file: str):
    """Save metrics to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {output_file}")


def main():
    """Main function"""
    ground_truth_file = r"test.json"
    predictions_file = r"classification_results.json"
    output_file = r"evaluation_metrics.json"

    print("Loading ground truth data...")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"Loaded {len(ground_truth)} ground truth samples")

    print("Loading predictions...")
    predictions = load_predictions(predictions_file)
    print(f"Loaded {len(predictions)} predictions")

    print("\nCalculating metrics...")
    metrics = calculate_metrics(ground_truth, predictions)

    print(f"\nAccuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1: {metrics.get('macro_avg', {}).get('f1_score', 0):.4f}")
    print(f"Weighted F1: {metrics.get('weighted_avg', {}).get('f1_score', 0):.4f}")

    save_metrics_report(metrics, output_file)


if __name__ == "__main__":
    main()
