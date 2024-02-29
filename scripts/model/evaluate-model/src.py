import pandas as pd
import numpy as np
import re
import argparse
import os

title_patterns = [
    r"\bLt\.\b",
    r"\bLes\.\b",
    r"\bDet\.\b",
    r"\bSat\.\b",
    r"\bCet\.\b",
    r"\bDetective\b",
    r"\bDetectives\b",
    r"\bOfficer\b",
    r"\bOfficers\b",
    r"\bSgt\.\b",
    r"\bLieutenant\b",
    r"\bSergeant\b",
    r"\bCaptain\b",
    r"\bCorporal\b",
    r"\bDeputy\b",
    r"\bOfc\.\b",
    r"\b\(?Technician\)?\b",
    r"\b\(?Criminalist\)?\b",
    r"\b\(?Photographer\)?\b",
    r"\bPolice Officer\b",
    r"\bCrime Lab Technician\b",
     r"\bLt\.\b",
    r"\bLes\.\b",
    r"\bDet\.\b",
    r"\bSat\.\b",
    r"\bCet\.\b",
    r"\bDetective\b",
    r"\bDetectives\b",
    r"\bOfficer\b",
    r"\bSgt\.\b",
    r"\bLieutenant\b",
    r"\bSergeant\b",
    r"\bCaptain\b",
    r"\bCorporal\b",
    r"\bDeputy\b",
    r"\bOfc\.\b",
    r"\b\(?Technician\)?\b",
    r"\b\(?Criminalist\)?\b",
    r"\b\(?Photographer\)?\b",
    r"\bPolice Officer\b",
    r"\bUNIT \d+\b",  
    r"\bMOUNTED OFFICERS\b",
    r"\bCrime Lab Technician\b",
    r"Officer Context:",
    r"P/O",
    r"Police"
]


def split_names(row):
    split_patterns = [r" and ", r" ?, ?", r" ?: ?", r" ?& ?"]
    for pattern in split_patterns:
        if re.search(pattern, row):
            return re.split(pattern, row)
    return [row]


def extract_last_name(name):
    words = name.split()
    return words[-1] if words else ""


def adjust_last_name(name):
    split_name = re.split(r"(?<=[a-z])(?=[A-Z])", name)
    return split_name[-1] if split_name else name


def refine_adjusted_last_name(name):
    name = adjust_last_name(name)
    split_name = re.split(r"(?<=[a-z])(?=[A-Z])", name)
    return split_name[-1] if split_name else name


def levenshtein_distance(s1, s2):
    dp = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[len(s1)][len(s2)]


def levenshtein_ratio(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - distance / max_len


def find_best_match(name, potential_matches):
    sorted_matches = sorted(
        potential_matches, key=lambda x: -x[1]
    )  # Sort by similarity score descending
    top_matches = sorted_matches[:3]  # Consider the top 3 matches

    # Check if any of the top matches is an exact match
    for match in top_matches:
        if match[0] == name:
            return match[0]

    # If no exact match, return the name with the highest similarity score
    return sorted_matches[0][0]


def split_consolidated_names(name):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)


def clean_officer_names(df, title_patterns):
    for pattern in title_patterns:
        df["Officer Name"] = (
            df["Officer Name"]
            .str.replace(pattern, "", regex=True)
            .str.strip()
            .apply(split_consolidated_names)
            .str.lower()
        )
        df["Officer Names to Match"] = (
            df["Officer Names to Match"]
            .str.replace(pattern, "", regex=True)
            .str.strip()
            .apply(split_consolidated_names)
            .str.lower()
        )
    return df


def preprocess_data(dfa_path, dfb_path, title_patterns):
    dfa = pd.read_csv(dfa_path)[["Officer Name", "iteration"]]
    dfb = pd.read_csv(dfb_path)
    df = pd.concat([dfa, dfb], axis=1).fillna("")
    df = clean_officer_names(df, title_patterns)
    return df


def compute_levenshtein_metrics(df, all_ground_truth_names, levenshtein_threshold=0.5):
    
    """
    Compute similarity metrics between extracted officer names and ground truth names using 
    Levenshtein distance across multiple iterations.

    Parameters:
    - df (DataFrame): The DataFrame containing the officer names and other relevant columns.
    - all_ground_truth_names (set of str): The set of all ground truth names against which the extracted names will be matched.
    - levenshtein_threshold (float, default=0.7): The threshold of similarity ratio above which names are considered a match.

    Returns:
    - results_df (DataFrame): A DataFrame containing various similarity metrics for each iteration such as precision, 
                              recall, f1_score, and others. It also includes cumulative metrics and metrics after 6 iterations.

    Note:
    The function first identifies potential matches based on the Levenshtein similarity ratio. 
    For each extracted name, it finds the ground truth name with the highest similarity and checks 
    if this similarity is above the given threshold. If so, the names are considered a match. The function 
    then computes precision, recall, and F1 score metrics for each iteration, as well as cumulative metrics.
    """
    
    results_levenshtein = []
    all_matched_across_iterations = set()
    all_extracted_across_iterations = set()
    cumulative_matched_names = set()

    for iteration in range(1, 7):
        beta = 2
        extracted_last_names = df[df["iteration"] == iteration][
            "Officer Refined Adjusted Last Name"
        ].unique()
        matched_last_names = {}

        for name in extracted_last_names:
            similarities = [
                (gt_name, levenshtein_ratio(name, gt_name))
                for gt_name in all_ground_truth_names
            ]
            best_match = find_best_match(name, similarities)

            max_similarity = max([similarity[1] for similarity in similarities])
            if max_similarity > levenshtein_threshold:
                matched_last_names[name] = best_match

        current_TP = len(
            set(matched_last_names.values()).intersection(all_ground_truth_names)
        )
        current_FP = len(matched_last_names) - current_TP
        current_precision = (
            current_TP / (current_TP + current_FP) if current_TP + current_FP > 0 else 0
        )
        current_recall = (
            current_TP / len(all_ground_truth_names)
            if len(all_ground_truth_names) > 0
            else 0
        )
        current_f1_score = (
            2
            * (current_precision * current_recall)
            / (current_precision + current_recall)
            if current_precision + current_recall > 0
            else 0
        )

        cumulative_matched_names.update(matched_last_names.values())

        cumulative_TP = len(
            cumulative_matched_names.intersection(all_ground_truth_names)
        )
        cumulative_FP = len(cumulative_matched_names) - cumulative_TP
        cumulative_precision = (
            cumulative_TP / (cumulative_TP + cumulative_FP)
            if cumulative_TP + cumulative_FP > 0
            else 0
        )
        cumulative_recall = (
            cumulative_TP / len(all_ground_truth_names)
            if len(all_ground_truth_names) > 0
            else 0
        )
        cumulative_f1_score = (
            2
            * (cumulative_precision * cumulative_recall)
            / (cumulative_precision + cumulative_recall)
            if cumulative_precision + cumulative_recall > 0
            else 0
        )

        cumulative_f_beta_score = (
            (1 + beta**2)
            * (cumulative_precision * cumulative_recall)
            / ((beta**2 * cumulative_precision) + cumulative_recall)
            if (cumulative_precision + cumulative_recall) > 0
            else 0
        )

        results_levenshtein.append(
            {
                "iteration": iteration,
                "matched_count": len(matched_last_names),
                "total_ground_truth": len(all_ground_truth_names),
                "percentage_matched": len(matched_last_names)
                / len(all_ground_truth_names)
                * 100,
                "matched_names": matched_last_names,
                "true_positives": current_TP,
                "false_positives": current_FP,
                "precision": current_precision,
                "recall": current_recall,
                "f1_score": current_f1_score,
                "cumulative_true_positives": cumulative_TP,
                "cumulative_false_positives": cumulative_FP,
                "cumulative_precision": cumulative_precision,
                "cumulative_recall": cumulative_recall,
                "cumulative_f1_score": cumulative_f1_score,
                "cumulative_f_beta_score": cumulative_f_beta_score,

            }
        )

        all_matched_across_iterations.update(matched_last_names.values())
        all_extracted_across_iterations.update(extracted_last_names)

    # # Metrics after 6 iterations
    # total_TP = len(all_matched_across_iterations.intersection(all_ground_truth_names))
    # total_FP = len(all_extracted_across_iterations) - total_TP
    # total_precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    # total_recall = (
    #     total_TP / len(all_ground_truth_names) if len(all_ground_truth_names) > 0 else 0
    # )
    # total_f1_score = (
    #     2 * (total_precision * total_recall) / (total_precision + total_recall)
    #     if total_precision + total_recall > 0
    #     else 0
    # )

    # # Calculate F-beta score
    # total_f_beta_score = (
    #     (1 + beta**2)
    #     * (total_precision * total_recall)
    #     / ((beta**2 * total_precision) + total_recall)
    #     if (total_precision + total_recall) > 0
    #     else 0
    # )

    unmatched_ground_truth_names_after_6_iterations = (
        set(all_ground_truth_names) - all_matched_across_iterations
    )

    results_df = pd.DataFrame(results_levenshtein)
    # results_df.loc[5, "Total True Positives After 6 Iterations"] = total_TP
    # results_df.loc[5, "Total False Positives After 6 Iterations"] = total_FP
    # results_df.loc[5, "Total Precision After 6 Iterations"] = total_precision
    # results_df.loc[5, "Total Recall After 6 Iterations"] = total_recall
    # results_df.loc[5, "Total F1 Score After 6 Iterations"] = total_f1_score
    # results_df.loc[5, "Total F-beta Score After 6 Iterations"] = total_f_beta_score
    results_df.loc[5, "Unmatched Ground Truth Names After 6 Iterations"] = str(
        unmatched_ground_truth_names_after_6_iterations
    )

    return results_df


def aggregate_results(df, results_df, num_of_queries):

    file_type = args.file_type

    overall_output_file = (
        "../../../data/wrongful-convictions/json/evaluate/output-4-og-params/overall-output.csv"
    )

    results_df["File Type"] = file_type
    results_df["Num of Queries"] = num_of_queries

    if os.path.exists(overall_output_file):
        with open(overall_output_file, "a") as f:
            results_df.to_csv(f, header=False, index=False)
    else:
        with open(overall_output_file, "w") as f:
            results_df.to_csv(f, header=True, index=False)
    return results_df




def main(dfa_path, dfb_path, output_dir, file_type):
    df = preprocess_data(dfa_path, dfb_path, title_patterns)
    df["Split Officer Names"] = df["Officer Name"].apply(split_names).str[0]
    df["Officer Last Name"] = df["Split Officer Names"].apply(extract_last_name)
    df["Officer Refined Adjusted Last Name"] = df["Officer Last Name"].apply(
        refine_adjusted_last_name
    )
    df["Split Officer Names to Match"] = (
        df["Officer Names to Match"].apply(split_names).str[0]
    )
    df["Officer Match Last Name"] = df["Split Officer Names to Match"].apply(
        extract_last_name
    )
    df["Officer Match Refined Adjusted Last Name"] = df[
        "Officer Match Last Name"
    ].apply(refine_adjusted_last_name)
    all_ground_truth_names = set(
        [
            name
            for name in df["Officer Match Refined Adjusted Last Name"].unique()
            if name.strip() != ""
        ]
    )

    if "_six_queries.csv" in dfa_path:
        num_of_queries = 6
    elif "_one_query.csv" in dfa_path:
        num_of_queries = 1
    else:
        num_of_queries = None  

    results_df = compute_levenshtein_metrics(df, all_ground_truth_names)
    output_file_name = (
        dfa_path.split("/")[-1]
        .replace("_six_queries.csv", "_six_queries-results.csv")
        .replace("_one_query.csv", "_one_query-results.csv")
    )
    output_name = os.path.join(output_dir, output_file_name)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    results_df.to_csv(output_name, index=False)

    filename = os.path.basename(dfa_path)
    results_df["filename"] = filename

    aggregate_results(results_df, results_df, num_of_queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files.")
    parser.add_argument(
        "--input_dfa", type=str, required=True, help="Path to the dfa CSV file."
    )
    parser.add_argument(
        "--input_dfb", type=str, required=True, help="Path to the dfb CSV file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output CSV file."
    )
    parser.add_argument(
        "--file-type",
        type=str,
        required=True,
        help="Type of the file being processed (either 'reports' or 'transcripts').",
    )
    
    args = parser.parse_args()
    main(args.input_dfa, args.input_dfb, args.output, args.file_type)
