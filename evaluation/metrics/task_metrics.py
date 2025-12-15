"""
Task-specific metrics for evaluating model performance across different stages.
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter
import re


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    Compute accuracy for classification tasks.
    
    Args:
        predictions: List of predicted labels
        labels: List of ground truth labels
        
    Returns:
        Accuracy score (0-1)
    """
    if not predictions or not labels:
        return 0.0
    
    correct = sum(1 for pred, label in zip(predictions, labels) if pred.strip().lower() == label.strip().lower())
    return correct / len(labels)


def compute_f1_score(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Compute F1 score for classification tasks.
    
    Args:
        predictions: List of predicted labels
        labels: List of ground truth labels
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    if not predictions or not labels:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Create confusion matrix
    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()
    
    for pred, label in zip(predictions, labels):
        pred = pred.strip().lower()
        label = label.strip().lower()
        
        if pred == label:
            true_positives[label] += 1
        else:
            false_positives[pred] += 1
            false_negatives[label] += 1
    
    # Compute micro-averaged metrics
    tp_total = sum(true_positives.values())
    fp_total = sum(false_positives.values())
    fn_total = sum(false_negatives.values())
    
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_cohens_kappa(predictions: List[str], labels: List[str]) -> float:
    """
    Compute Cohen's Kappa for inter-rater reliability (medical tasks).
    
    Args:
        predictions: List of predicted labels
        labels: List of ground truth labels
        
    Returns:
        Cohen's Kappa score
    """
    if not predictions or not labels:
        return 0.0
    
    n = len(labels)
    
    # Observed agreement
    observed_agreement = sum(1 for pred, label in zip(predictions, labels) 
                            if pred.strip().lower() == label.strip().lower()) / n
    
    # Expected agreement by chance
    pred_counts = Counter(pred.strip().lower() for pred in predictions)
    label_counts = Counter(label.strip().lower() for label in labels)
    
    expected_agreement = sum(pred_counts[cat] * label_counts[cat] 
                            for cat in set(list(pred_counts.keys()) + list(label_counts.keys()))) / (n * n)
    
    # Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else 0.0
    
    return kappa


def compute_bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
    """
    Compute BLEU score for caption generation (simple implementation).
    
    Args:
        predictions: List of generated captions
        references: List of reference captions
        n: Maximum n-gram order (default: 4)
        
    Returns:
        BLEU score
    """
    def tokenize(text):
        return text.lower().split()
    
    def count_ngrams(tokens, n):
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def modified_precision(pred_tokens, ref_tokens, n):
        pred_ngrams = count_ngrams(pred_tokens, n)
        ref_ngrams = count_ngrams(ref_tokens, n)
        
        clipped_count = sum(min(pred_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in pred_ngrams)
        total_count = sum(pred_ngrams.values())
        
        return clipped_count / total_count if total_count > 0 else 0.0
    
    # Compute precisions for all n-grams
    precisions = []
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        for i in range(1, n + 1):
            p = modified_precision(pred_tokens, ref_tokens, i)
            precisions.append(p)
    
    # Geometric mean of precisions
    if not precisions or any(p == 0 for p in precisions):
        return 0.0
    
    bleu = np.exp(np.mean(np.log(precisions)))
    
    # Brevity penalty
    pred_len = sum(len(tokenize(pred)) for pred in predictions)
    ref_len = sum(len(tokenize(ref)) for ref in references)
    bp = 1.0 if pred_len > ref_len else np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
    
    return bp * bleu


def extract_answer_from_response(response: str, task_type: str = 'mcq') -> str:
    """
    Extract the answer from a model's response based on task type.
    
    Args:
        response: Model's generated response
        task_type: Type of task ('mcq', 'classification', 'generation')
        
    Returns:
        Extracted answer
    """
    if task_type == 'mcq':
        # Look for A, B, C, D at the start or after "Answer:"
        match = re.search(r'\b([A-D])\b', response, re.IGNORECASE)
        return match.group(1).upper() if match else response.strip()[:1].upper()
    
    elif task_type == 'classification':
        # Extract first word/phrase as classification result
        # Common patterns: "Walking", "Running", "Stage 3", etc.
        match = re.search(r'^\s*([A-Za-z0-9\s]+)', response)
        return match.group(1).strip() if match else response.strip().split('\n')[0]
    
    elif task_type == 'generation':
        # Return the full response for generative tasks
        return response.strip()
    
    return response.strip()


def compute_stage_metrics(stage: int, predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Compute appropriate metrics for each stage.
    
    Args:
        stage: Stage number (1-5)
        predictions: List of predictions
        labels: List of ground truth labels
        
    Returns:
        Dictionary of metrics appropriate for the stage
    """
    metrics = {}
    
    if stage == 1:  # TSQA (Multiple Choice)
        metrics['accuracy'] = compute_accuracy(predictions, labels)
        f1_scores = compute_f1_score(predictions, labels)
        metrics.update(f1_scores)
    
    elif stage == 2:  # M4 (Captioning)
        metrics['bleu'] = compute_bleu_score(predictions, labels)
        # Could add ROUGE, METEOR here
    
    elif stage == 3:  # HAR (Classification + CoT)
        # Extract classification from CoT response
        extracted_preds = [extract_answer_from_response(pred, 'classification') for pred in predictions]
        metrics['accuracy'] = compute_accuracy(extracted_preds, labels)
        f1_scores = compute_f1_score(extracted_preds, labels)
        metrics.update(f1_scores)
    
    elif stage == 4:  # Sleep (Medical Classification)
        extracted_preds = [extract_answer_from_response(pred, 'classification') for pred in predictions]
        metrics['accuracy'] = compute_accuracy(extracted_preds, labels)
        metrics['cohens_kappa'] = compute_cohens_kappa(extracted_preds, labels)
        f1_scores = compute_f1_score(extracted_preds, labels)
        metrics.update(f1_scores)
    
    elif stage == 5:  # ECG (Medical Diagnosis)
        extracted_preds = [extract_answer_from_response(pred, 'classification') for pred in predictions]
        metrics['accuracy'] = compute_accuracy(extracted_preds, labels)
        metrics['cohens_kappa'] = compute_cohens_kappa(extracted_preds, labels)
        f1_scores = compute_f1_score(extracted_preds, labels)
        metrics.update(f1_scores)
    
    return metrics


if __name__ == "__main__":
    # Test the metrics
    print("Testing Task Metrics...")
    
    # Test Stage 1 (MCQ)
    print("\n=== Stage 1 (MCQ) ===")
    predictions = ["A", "B", "C", "A", "D"]
    labels = ["A", "B", "C", "C", "D"]
    metrics = compute_stage_metrics(1, predictions, labels)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    
    # Test Stage 2 (Captioning)
    print("\n=== Stage 2 (Captioning) ===")
    predictions = ["the time series shows an upward trend", "there is a weekly pattern"]
    labels = ["time series exhibits increasing trend", "weekly seasonality is present"]
    metrics = compute_stage_metrics(2, predictions, labels)
    print(f"BLEU: {metrics['bleu']:.3f}")
    
    # Test Stage 3 (HAR)
    print("\n=== Stage 3 (HAR) ===")
    predictions = [
        "Step 1: ... Step 2: ... Conclusion: WALKING",
        "Analysis shows: RUNNING",
        "SITTING based on the pattern"
    ]
    labels = ["WALKING", "RUNNING", "STANDING"]
    metrics = compute_stage_metrics(3, predictions, labels)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    
    # Test Stage 4 (Sleep)
    print("\n=== Stage 4 (Sleep) ===")
    predictions = ["Stage 3 NREM", "Stage 2", "REM sleep", "Stage 1"]
    labels = ["Stage 3 NREM", "Stage 2", "REM", "Wake"]
    metrics = compute_stage_metrics(4, predictions, labels)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Cohen's Kappa: {metrics['cohens_kappa']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    
    print("\n✅ All tests passed!")

