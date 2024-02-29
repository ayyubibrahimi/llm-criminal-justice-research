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
]


def split_names(row):
    split_patterns = [r" and ", r",", r":"]
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
        df["Name"] = (
            df["Name"]
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
    dfa = pd.read_csv(dfa_path)[["Name"]]
    dfb = pd.read_csv(dfb_path)[["Officer Names to Match"]]
    df = pd.concat([dfa, dfb], axis=1).fillna("")
    df = clean_officer_names(df, title_patterns)
    return df


def compute_levenshtein_metrics(df, all_ground_truth_names, levenshtein_threshold=0.7):
    matched_names = {}

    for name in df["Officer Refined Adjusted Last Name"].unique():
        similarities = [
            (gt_name, levenshtein_ratio(name, gt_name))
            for gt_name in all_ground_truth_names
        ]
        best_match = find_best_match(name, similarities)

        max_similarity = max([similarity[1] for similarity in similarities]) if similarities else 0  # Handling empty similarities list
        if max_similarity > levenshtein_threshold:
            matched_names[name] = best_match

    TP = len(set(matched_names.values()).intersection(all_ground_truth_names))
    FP = len(matched_names) - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / len(all_ground_truth_names) if len(all_ground_truth_names) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0
    )
    beta = 2
    denominator = (beta**2 * precision) + recall
    f1_beta_score = ((1 + beta**2) * precision * recall) / denominator if denominator != 0 else 0  # Handling ZeroDivisionError

    if len(all_ground_truth_names) == 0:
        return pd.DataFrame([{
            "matched_count": 0,
            "total_ground_truth": 0,
            "percentage_matched": 0,
            "matched_names": {},
            "true_positives": 0,
            "false_positives": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "f1_beta_score": 0
        }])

    results = {
        "matched_count": len(matched_names),
        "total_ground_truth": len(all_ground_truth_names),
        "percentage_matched": len(matched_names) / len(all_ground_truth_names) * 100,
        "matched_names": matched_names,
        "true_positives": TP,
        "false_positives": FP,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "f1_beta_score": f1_beta_score
    }

    return pd.DataFrame([results])


def aggregate_results(df, output_name, results_df):
    metrics = {
        "Avg Precision": df["precision"].mean(),
        "Avg Recall": df["recall"].mean(),
        "Avg F1 Score": df["f1_score"].mean(),
        "Avg F1 Beta Score": df["f1_beta_score"].mean()  
    }

    overall_summary_file = "data/output/overall-summary.csv"
    overall_average_output_file = "data/output/overall-average-output.csv"
    
    # Write to overall-summary.csv
    if os.path.exists(overall_summary_file):
        summary_df = pd.read_csv(overall_summary_file)
    else:
        summary_df = pd.DataFrame()

    if "filename" not in summary_df.columns:
        summary_df["filename"] = pd.Series(dtype=str)

    filename = os.path.basename(output_name)
    if filename in summary_df["filename"].values:
        for key, value in metrics.items():
            existing_value = summary_df.loc[summary_df["filename"] == filename, key].values[0]
            summary_df.loc[summary_df["filename"] == filename, key] = (existing_value + value) / 2
    else:
        metrics["filename"] = filename
        summary_df = pd.concat([summary_df, pd.DataFrame([metrics])], ignore_index=True)

    summary_df.to_csv(overall_summary_file, index=False)

    # Compute averages and write to overall-average-output.csv
    avg_metrics = {
        "precision": summary_df["Avg Precision"].mean(),
        "Recall": summary_df["Avg Recall"].mean(),
        "F1": summary_df["Avg F1 Score"].mean(),
        "F_Beta": summary_df["Avg F1 Beta Score"].mean()
    }
    avg_output_df = pd.DataFrame([avg_metrics])
    avg_output_df.to_csv(overall_average_output_file, index=False)


def main(dfa_path, dfb_path, output_dir):
    # Check if the files exist
    if not os.path.exists(dfa_path):
        print(f"Error: {dfa_path} does not exist!")
        return
    if not os.path.exists(dfb_path):
        print(f"Error: {dfb_path} does not exist!")
        return
    
    print(f"Processing {dfa_path} and {dfb_path} ...")
    
    df = preprocess_data(dfa_path, dfb_path, title_patterns)
    df["Split Officer Names"] = df["Name"].apply(split_names).str[0]
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
    results_df = compute_levenshtein_metrics(df, all_ground_truth_names)
    output_file_name = dfa_path.split("/")[-1].replace(".csv", "-comparison.csv")
    output_name = os.path.join(output_dir, output_file_name)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    results_df.to_csv(output_name, index=False)

    filename = os.path.basename(dfa_path)
    # Add the filename as a new column to the results_df
    results_df['filename'] = filename
    aggregate_results(results_df, output_name, results_df)
    print(f"Finished processing {dfa_path} and {dfb_path}. Results saved to {output_name}")


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

    args = parser.parse_args()
    main(args.input_dfa, args.input_dfb, args.output)
