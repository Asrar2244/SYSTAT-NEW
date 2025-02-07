"""
Module for performing a two-sample Z-test. Accepts input data in JSON format or as a file, 
computes the Z-test, and returns the results including the Z-statistic, p-value, 
confidence interval, and conclusion.
"""
import json
import pandas as pd
from statsmodels.stats.weightstats import ztest
from scipy.stats import norm
from flask import Blueprint, request, jsonify
from app.api.helpers.constant import (
    VALUE_ERROR_MSG,
    KEY_ERROR_MSG,
    TYPE_ERROR_MSG,
    ZERO_DIVISION_ERROR_MSG,
    INDEX_ERROR_MSG,
    UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR,
    LOG_KEY_ERROR,
    LOG_TYPE_ERROR,
    LOG_ZERO_DIVISION_ERROR,
    LOG_INDEX_ERROR,
    LOG_UNEXPECTED_ERROR,
    TWO_SAMPLE_Z_TEST_LOG_FILE_PATH,
    CONFIDENCE_INTERVAL_DEFAULT,
    ALTERNATIVE_DEFAULT
)
from app.api.helpers.logger import Logger

two_sample_ztest_api = Blueprint('two_sample_ztest', __name__)

@two_sample_ztest_api.route('/two-sample-ztest', methods=['POST'])
def perform_two_sample_ztest():
    """
    Perform a two-sample Z-test on input data and return statistical results.

    Expected JSON input format:
        {
            "column": "test_scores",
            "group_column": "group",
            "std1": 10.5,
            "std2": 9.8,
            "confidence": 0.95,
            "alternative": "two-sided",
            "data": [
            {
            "group": "A", "test_scores": 85
            },
            {
            "group": "A", "test_scores": 90
            },
            {
            "group": "B", "test_scores": 78
            },
            {
            "group": "B", "test_scores": 82
            }
            ]
        }

    """
    logger = Logger(TWO_SAMPLE_Z_TEST_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        column = input_data.get('column')
        group_col = input_data.get('group_column')
        confidence = float(input_data.get('confidence', CONFIDENCE_INTERVAL_DEFAULT ))
        alternative = input_data.get('alternative', ALTERNATIVE_DEFAULT).lower()

        if not column or not group_col:
            raise ValueError("Both 'column' and 'group_column' are required.")

        if not (0 < confidence < 1):
            raise ValueError("Confidence level must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        df = pd.DataFrame(input_data['data'])

        # Ensure there are exactly two groups
        groups = df[group_col].unique()
        if len(groups) != 2:
            logger.error(f"Invalid grouping variable. Found {len(groups)} groups instead of 2.")
            return jsonify({"error": "Ensure exactly two groups."}), 400

        # Separate the data into two groups
        group1_data = df[df[group_col] == groups[0]][column]
        group2_data = df[df[group_col] == groups[1]][column]

        # Perform two-sample Z-test
        z_stat, p_value = ztest(group1_data, group2_data, alternative=alternative)

        # Confidence interval calculation
        mean_diff = group1_data.mean() - group2_data.mean()
        std_err = (group1_data.std()**2 / len(group1_data) + group2_data.std()**2 / len(group2_data))**0.5
        z_critical = norm.ppf(1 - (1 - confidence) / 2)
        ci_low = mean_diff - z_critical * std_err
        ci_high = mean_diff + z_critical * std_err

        # Conclusion based on p-value
        conclusion = "Significant difference between the means." if p_value < (1 - confidence) else "No significant difference between the means."

        results = {
            "hypothesis": f"Ho: Mean1 = Mean2 vs H1: Mean1 {'!=' if alternative == 'two-sided' else ('<' if alternative == 'smaller' else '>')} Mean2",
            "grouping_variable": group_col,
            "summary": {
                f"{groups[0]}": {"N": len(group1_data), "Mean": group1_data.mean()},
                f"{groups[1]}": {"N": len(group2_data), "Mean": group2_data.mean()}
            },
            "confidence_interval": {
                "confidence_level": confidence,
                "mean_difference": mean_diff,
                "lower_bound": ci_low,
                "upper_bound": ci_high
            },
            "z_stat": z_stat,
            "p_value": p_value,
            "conclusion": conclusion
        }
        logger.info(json.dumps(results))
        logger.info("Two-sample Z-test completed successfully.")
        return jsonify(results), 200

    except ValueError as ve:
        logger.error(LOG_VALUE_ERROR.format(str(ve)))
        return jsonify({"error": VALUE_ERROR_MSG.format(str(ve))}), 400
    except KeyError as ke:
        logger.error(LOG_KEY_ERROR.format(str(ke)))
        return jsonify({"error": KEY_ERROR_MSG.format(str(ke))}), 400
    except TypeError as te:
        logger.error(LOG_TYPE_ERROR.format(str(te)))
        return jsonify({"error": TYPE_ERROR_MSG.format(str(te))}), 400
    except ZeroDivisionError as zde:
        logger.error(LOG_ZERO_DIVISION_ERROR.format(str(zde)))
        return jsonify({"error": ZERO_DIVISION_ERROR_MSG}), 400
    except IndexError as ie:
        logger.error(LOG_INDEX_ERROR.format(str(ie)))
        return jsonify({"error": INDEX_ERROR_MSG}), 400
    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500