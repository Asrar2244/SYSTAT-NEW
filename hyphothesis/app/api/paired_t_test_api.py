'''Module for performing a paired t test with statistical calculations.'''
import json
from statsmodels.stats.weightstats import ztest
import scipy.stats as stats
from flask import Blueprint, request, jsonify
import numpy as np

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
    PAIRED_T_TEST_LOG_FILE_PATH
)

from app.api.helpers.logger import Logger

paired_t_test_api = Blueprint('paired_t_test_api', __name__)

@paired_t_test_api.route('/paired-t-test-api', methods=['POST'])
def perform_paired_t_test():
    '''
    Endpoint for Paired T-Test. Expects a JSON input with two variables.
    {
    "vehicle": [55, 45, 65, 54, 43, 45, 54, 63, 73, 36, 65],
    "drugs": [74, 85, 76, 58, 67, 47, 56, 92, 71, 93, 86]
    }
    '''
    logger = Logger(PAIRED_T_TEST_LOG_FILE_PATH)
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid input. Please provide JSON data."}), 400
        
        data = request.get_json()
        logger.info(f"Received data: {data}")

        if not data or len(data) != 2:
            return jsonify({"error": "Invalid JSON format. Please provide exactly two variables."}), 400
        
        before, after = list(data.values())
        
        if not isinstance(before, list) or not isinstance(after, list):
            return jsonify({"error": "Both variables must be lists."}), 400
        
        if len(before) != len(after):
            return jsonify({"error": "The two lists must have the same length."}), 400
        
        # Calculate paired t-test results
        result = calculate_paired_t_test(before, after)
        logger.info(json.dumps(result))
        return jsonify(result), 200
    
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


def calculate_paired_t_test(before, after):
    """
    Perform the paired t-test and return a detailed structured response.
    """
    try:
        differences = np.array(after) - np.array(before)
        
        # Normality Test (Shapiro-Wilk)
        normality_p = stats.shapiro(differences)[1]
        normality_passed = normality_p > 0.05
        
        # Calculate statistics
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        mean_difference = np.mean(differences)
        std_dev_before = np.std(before, ddof=1)
        std_dev_after = np.std(after, ddof=1)
        std_dev_diff = np.std(differences, ddof=1)
        sem_diff = std_dev_diff / np.sqrt(len(before))
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(before, after)
        df = len(before) - 1
        ci_low, ci_high = mean_difference - 1.96 * sem_diff, mean_difference + 1.96 * sem_diff
        one_tailed_p = p_value / 2
        
        return {
            "Test": "Paired t-test",
            "Normality Test (Shapiro-Wilk)": {
                "P-Value": round(normality_p, 3),
                "Passed": bool(normality_passed)
            },
            "Sample Statistics": {
                "Before Treatment": {
                    "N": len(before),
                    "Mean": round(mean_before, 3),
                    "Std Dev": round(std_dev_before, 3),
                },
                "After Treatment": {
                    "N": len(after),
                    "Mean": round(mean_after, 3),
                    "Std Dev": round(std_dev_after, 3),
                },
                "Difference": {
                    "Mean Difference": round(mean_difference, 3),
                    "Std Dev": round(std_dev_diff, 3),
                    "SEM": round(sem_diff, 3)
                }
            },
            "T-Test Results": {
                "t-Statistic": round(t_stat, 3),
                "Degrees of Freedom": df,
                "95% Confidence Interval": [round(ci_low, 3), round(ci_high, 3)],
                "Two-Tailed P-Value": round(p_value, 5),
                "One-Tailed P-Value": round(one_tailed_p, 5)
            }
        }
    except Exception as e:
        raise ValueError(f"Error in paired t-test calculation: {str(e)}")


        

