def compute_accuracy(ground_truth, result_list):
    if len(result_list) == 0:
        return 0
    else:
        matched_results = sum(1 for value in result_list if value == ground_truth)
        accuracy = matched_results / len(result_list) if len(result_list) > 0 else 0.0
        return accuracy
