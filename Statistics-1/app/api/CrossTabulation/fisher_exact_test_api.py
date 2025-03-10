"""
Author: Asrar
Module for performing Fisher's Exact Test. Accepts input data in JSON format,
computes the test, and returns results including the odds ratio, p-value, 
expected counts, and conclusion. If any cell count exceeds 100, it prompts
the user to switch to a Chi-Square Test.
"""

import pandas as pd
from flask import Blueprint, request, jsonify
from scipy.stats import fisher_exact
from ..helpers.logger import Logger
from ..helpers.constant import (
    VALUE_ERROR_MSG, KEY_ERROR_MSG, TYPE_ERROR_MSG, INDEX_ERROR_MSG, UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR, LOG_KEY_ERROR, LOG_TYPE_ERROR, LOG_INDEX_ERROR, LOG_UNEXPECTED_ERROR,
    FISHER_EXACT_TEST_LOG_FILE_PATH
)
import importlib  # Import module dynamically

# Initialize Blueprint for Fisher's Exact Test
fisher_exact_test_api = Blueprint('fisher_exact_test_api', __name__)

logger = Logger(FISHER_EXACT_TEST_LOG_FILE_PATH)

def fisher_exact_test_logic(data, db):
    """
    Perform Fisher's Exact Test and return results.
     Expected JSON input format:
{
    "columns": ["Smoker", "Non-Smoker"],
    "rows": ["Cancer", "Non-Cancer"],
    "data": [
        [5, 2], [4, 8]
    ],
      "switch_to_chi_square": "yes"
}
    If any cell has a value >100, prompt user to switch to Chi-Square Test.
    """
    try:
        switch_to_chi_square = data.get('switch_to_chi_square', None)
        if db:
              observed_data = data.get('data')
              columns = data.get('columns')
              rows = data.get('rows')

        else:
            df = pd.DataFrame(data.get('data'))
            if not {'Group', 'Category'}.issubset(df.columns):
                return {"error": "Invalid input. Long format must contain 'Group' and 'Category' columns."}
            
            contingency_table = pd.crosstab(df['Category'], df['Group'])
            observed_data = contingency_table.values.tolist()
            columns = list(contingency_table.columns)
            rows = list(contingency_table.index)
              

        if not isinstance(observed_data, list) or len(columns) != 2 or len(rows) != 2:
            return {"error": "Invalid input. 'data' must be a 2x2 list, and 'columns' & 'rows' must each have exactly 2 elements."}

        df = pd.DataFrame(observed_data, index=rows, columns=columns)
        
        if df.shape != (2, 2):
            return {"error": "Invalid input: Data must be a 2x2 contingency table."}

        if df.values.sum() > 100:
            warning_msg = "Warning: At least one cell has a count >100. Fisher's Exact Test may not be reliable."
            if switch_to_chi_square is None:
                return {"warning": warning_msg, "message": "Do you want to switch to Chi-Square Test? (yes/no)"}

            elif switch_to_chi_square.lower() == "yes":
                chi_square_module = importlib.import_module("app.api.CrossTabulation.chi_square_test_api")
                perform_chi_square_test = getattr(chi_square_module, "perform_chi_square_test")
                
                logger.info("Switching to Chi-Square Test due to large cell counts.")
                response = perform_chi_square_test()

                if isinstance(response, tuple):
                    response_data, _ = response
                else:
                    response_data = response

                return response_data.get_json()

        odds_ratio, p_value = fisher_exact(df)
        row_totals = df.sum(axis=1)
        col_totals = df.sum(axis=0)
        grand_total = df.values.sum()

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
        db = data.get("DB", False)  # Extract 'DB' from input, default to False if missing
        result = fisher_exact_test_logic(data, db)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
