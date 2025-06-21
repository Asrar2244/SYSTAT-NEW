"""
Author: Asrar
Module for performing McNemar's Test (2x2) and McNemar's Symmetry Chi-square Test (NxN).
Supports NxN contingency tables and categorical before/after data input.
"""

import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from statsmodels.stats.contingency_tables import mcnemar
import scipy.stats as stats 
from datetime import datetime
from ..helpers.constant import (
    VALUE_ERROR_MSG, KEY_ERROR_MSG, TYPE_ERROR_MSG, INDEX_ERROR_MSG, UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR, LOG_KEY_ERROR, LOG_TYPE_ERROR, LOG_INDEX_ERROR, LOG_UNEXPECTED_ERROR,
    MCNEMAR_TEST_LOG_FILE_PATH
)
from ..helpers.logger import Logger

McNemars_test_api = Blueprint('mcnemar_api', __name__)
logger = Logger(MCNEMAR_TEST_LOG_FILE_PATH)

@McNemars_test_api.route('/mcnemar-test', methods=['POST'])
def perform_mcnemar_test():
    try:
        logger.info("Received request for McNemar's test.")
        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please provide JSON data."}), 400

        data = request.get_json()
        timestamp = datetime.now().strftime("%d %B %Y %H:%M:%S")

        input_format = data.get("input_data_format", "tabular").lower()
        yates_correction = data.get("yates_correction", False)

        # --- RAW INPUT FORMAT ---
        if input_format == "raw":
            long_format_data = data.get("long_format_data", [])
            if not isinstance(long_format_data, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in long_format_data):
                return jsonify({"error": "Invalid format for 'long_format_data'. Expected a list of [Category1, Category2] pairs."}), 400

            labels = list(set(label for pair in long_format_data for label in pair))
            contingency_matrix = pd.DataFrame(0, index=labels, columns=labels)

            for before, after in long_format_data:
                contingency_matrix.loc[before, after] += 1

            df = contingency_matrix
            rows, columns = list(df.index), list(df.columns)

        # --- TABULAR FORMAT ---
        elif input_format == "tabular":
            if not all(k in data for k in ["columns", "rows", "data"]):
                return jsonify({"error": "Missing 'columns', 'rows', or 'data' fields in 'tabular' input format."}), 400

            observed_data = data["data"]
            columns = data["columns"]
            rows = data["rows"]

            if not isinstance(observed_data, list) or not all(isinstance(row, list) for row in observed_data):
                return jsonify({"error": "Invalid 'data'. Must be a list of lists."}), 400

            df = pd.DataFrame(observed_data, index=rows, columns=columns)

        else:
            return jsonify({"error": "Invalid or missing 'input_data_format'. Use 'tabular' or 'raw'."}), 400

        logger.info(f"Processed data into contingency table of shape {df.shape}")

        # --- Basic Shape Validation ---
        if df.shape[0] != df.shape[1]:
            return jsonify({"error": "Input matrix must be square (NxN)."}), 400

        # --- McNemar's Test (2x2 Only) ---
        if df.shape == (2, 2):
            chi2_stat = mcnemar(df, exact=False, correction=yates_correction)
            p_value = chi2_stat.pvalue
            exact_p_value = mcnemar(df, exact=True).pvalue

            b, c = df.iloc[0, 1], df.iloc[1, 0]
            odds_ratio = round(b / c, 3) if c != 0 else "Undefined"
            ci_lower = round(np.exp(np.log(odds_ratio) - 1.96 * np.sqrt(1/b + 1/c)), 3) if c != 0 and b != 0 else "Undefined"
            ci_upper = round(np.exp(np.log(odds_ratio) + 1.96 * np.sqrt(1/b + 1/c)), 3) if c != 0 and b != 0 else "Undefined"

            p_case = round(b / (b + c), 3) if (b + c) != 0 else "Undefined"
            p_control = round(c / (b + c), 3) if (b + c) != 0 else "Undefined"
            relative_diff = round((p_case - p_control) * 100, 3) if p_case != "Undefined" and p_control != "Undefined" else "Undefined"

            conclusion = "Reject Null Hypothesis (Significant difference)" if p_value < 0.05 else "Fail to Reject Null Hypothesis (No significant difference)"

            result = {
                "Test Type": "McNemar's Test",
                "Timestamp": timestamp,
                "Chi-square": round(chi2_stat.statistic, 3),
                "Degrees of Freedom": 1,
                "P-Value": round(p_value, 5),
                "Exact McNemar Significance Probability": round(exact_p_value, 5),
                "Yates Correction Applied": yates_correction,
                "Odds Ratio": {
                    "Value": odds_ratio,
                    "95% Confidence Interval": [ci_lower, ci_upper]
                },
                "Proportions": {
                    "Cases Proportion": p_case,
                    "Controls Proportion": p_control
                },
                "Relative Difference (%)": relative_diff,
                "Observed Counts": df.round(3).to_dict(),
                "Conclusion": conclusion
            }

        # --- Symmetry Chi-Square for NxN ---
        else:
            expected = df.sum(axis=1).values.reshape(-1, 1) * df.sum(axis=0).values.reshape(1, -1) / df.values.sum()
            chi2_stat, p_value, dof, _ = stats.chi2_contingency(df)

            symmetry_chi2 = sum(
                (df.iloc[i, j] - df.iloc[j, i])**2 / (df.iloc[i, j] + df.iloc[j, i])
                for i in range(df.shape[0]) for j in range(i + 1, df.shape[1])
                if (df.iloc[i, j] + df.iloc[j, i]) > 0
            )

            symmetry_dof = (df.shape[0] * (df.shape[0] - 1)) // 2
            symmetry_p_value = 1 - stats.chi2.cdf(symmetry_chi2, symmetry_dof)

            conclusion = "Reject Null Hypothesis (Significant difference)" if p_value < 0.05 else "Fail to Reject Null Hypothesis (No significant difference)"

            result = {
                "Test Type": "McNemar's Symmetry Chi-square Test",
                "Timestamp": timestamp,
                "Pearson Chi-square": round(chi2_stat, 3),
                "Degrees of Freedom": dof,
                "P-Value": round(p_value, 5),
                "McNemar Symmetry Chi-square": round(symmetry_chi2, 3),
                "Symmetry Degrees of Freedom": symmetry_dof,
                "Symmetry P-Value": round(symmetry_p_value, 5),
                "Observed Counts": df.round(3).to_dict(),
                "Expected Counts": pd.DataFrame(expected, index=df.index, columns=df.columns).round(3).to_dict(),
                "Conclusion": conclusion
            }

        logger.info(f"Test result: {result}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500
