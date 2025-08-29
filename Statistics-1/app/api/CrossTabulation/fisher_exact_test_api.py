"""
Author: Asrar
Module for performing Fisher's Exact Test. Accepts input data in JSON format,
computes the test, and returns results including the odds ratio, p-value, 
expected counts, and conclusion. If any cell count or sum of values exceeds 100, 
it prompts the user to switch to a Chi-Square Test.
"""

import pandas as pd
from flask import Blueprint, request, jsonify
from scipy.stats import fisher_exact
from ..helpers.logger import Logger
import requests
import traceback
from ..helpers.constant import (
    VALUE_ERROR_MSG, KEY_ERROR_MSG, TYPE_ERROR_MSG, INDEX_ERROR_MSG, UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR, LOG_KEY_ERROR, LOG_TYPE_ERROR, LOG_INDEX_ERROR, LOG_UNEXPECTED_ERROR,
    FISHER_EXACT_TEST_LOG_FILE_PATH
)

fisher_exact_test_api = Blueprint('fisher_exact_test_api', __name__)
logger = Logger(FISHER_EXACT_TEST_LOG_FILE_PATH)

def fisher_exact_test_logic(data, input_format):
    try:
        switch_to_chi_square = data.get('switch_to_chi_square', None)

        # --- Handle Tabular Format ---
        if input_format == "tabular":
            observed_data = data.get('data')
            columns = data.get('columns')
            rows = data.get('rows')

            if not (isinstance(observed_data, list) and len(columns) == 2 and len(rows) == 2):
                return {"error": "Invalid input. 'data' must be 2x2, and 'columns' & 'rows' must each have 2 elements."}

            df = pd.DataFrame(observed_data, index=rows, columns=columns)

        # --- Handle Raw Format ---
        elif input_format == "raw":
            raw_df = pd.DataFrame(data.get('data'))

            if not {'Group', 'Category'}.issubset(raw_df.columns):
                return {"error": "Invalid input. Raw format must contain 'Group' and 'Category' columns."}

            contingency_table = pd.crosstab(raw_df['Category'], raw_df['Group'])

            if contingency_table.shape != (2, 2):
                return {"error": "Fisher's Exact Test can only be applied to 2x2 tables."}

            df = contingency_table
            observed_data = df.values.tolist()
            columns = list(df.columns)
            rows = list(df.index)

        else:
            return {"error": "Invalid or missing 'input_data_format'. Must be either 'tabular' or 'raw'."}

        # --- Validations ---
        if df.shape != (2, 2):
            return {"error": "Input must be a 2x2 contingency table."}

        max_cell_value = df.values.max()
        grand_total = df.values.sum()

        # --- Switch to Chi-Square if needed ---
        if max_cell_value > 100 or grand_total > 100:
            warning_msg = "Warning: At least one cell value or the sum of all 4 cells exceeds 100. Fisher's Exact Test may not be reliable."
            if switch_to_chi_square is None:
                return {
                    "warning": warning_msg,
                    "message": "Do you want to switch to Chi-Square Test? (yes/no)",
                    "recommended_action": "Set 'switch_to_chi_square': 'yes' in your request body."
                }

            elif switch_to_chi_square.lower() == "yes":
                logger.info("Switching to Chi-Square Test via internal dynamic API call.")
                try:
                    # Dynamically construct the Chi-Square API URL
                    base_url = request.host_url.rstrip('/')
                    chi_square_url = f"{base_url}/cross_tabulation/api/chi-square-test"

                    response = requests.post(chi_square_url, json=data)
                    return jsonify(response.json()), response.status_code
                except Exception as e:
                    logger.error(f"Error calling Chi-Square Test API: {str(e)}\n{traceback.format_exc()}")
                    return jsonify({"error": "Failed to call Chi-Square Test API"}), 500

        # --- Run Fisher Test ---
        odds_ratio, p_value = fisher_exact(df)
        row_totals = df.sum(axis=1)
        col_totals = df.sum(axis=0)

        expected_counts = pd.DataFrame(
            [[(row_totals.iloc[i] * col_totals.iloc[j]) / grand_total for j in range(2)] for i in range(2)],
            index=rows,
            columns=columns
        )

        row_percentages = (df.div(row_totals, axis=0) * 100).round(2)
        col_percentages = (df.div(col_totals, axis=1) * 100).round(2)
        total_percentages = (df / grand_total * 100).round(2)

        alpha = 0.05
        conclusion = "Reject Null Hypothesis (Variables are associated)" if p_value < alpha else "Fail to Reject Null Hypothesis (Variables are independent)"

        result = {
            "Test Used": "Fisher's Exact Test",
            "Counts and Percentages": {
                "Observed Counts": df.to_dict(),
                "Expected Counts": expected_counts.round(3).to_dict(),
                "% Row Total": row_percentages.to_dict(),
                "% Column Total": col_percentages.to_dict(),
                "% Total": total_percentages.to_dict()
            },
            "Fisher's Exact Test Results": {
                "Odds Ratio": round(odds_ratio, 3),
                "P-Value": round(p_value, 5),
                "Conclusion": conclusion
            }
        }

        return result

    except Exception as e:
        logger.error(f"Unexpected error in Fisher's Exact Test: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}


@fisher_exact_test_api.route('/fisher-exact-test', methods=['POST'])
def perform_fisher_exact_test():
    """
    API endpoint for Fisher's Exact Test.
    """
    try:
        data = request.get_json()
        input_format = data.get("input_data_format", "tabular").lower()

        if input_format not in ["tabular", "raw"]:
            return jsonify({"error": "Invalid input_data_format. Must be 'tabular' or 'raw'."}), 400

        result = fisher_exact_test_logic(data, input_format)

        # If result is already a Flask Response (when forwarded from chi-square), return as-is
        if isinstance(result, tuple):
            return result

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Unexpected error in perform_fisher_exact_test: {str(e)}")
        return jsonify({"error": str(e)}), 500
