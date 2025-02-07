'''This module defines an API endpoint for performing a Two-Sample T-Test on input data.'''
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
    TWO_SAMPLE_T_TEST_LOG_FILE_PATH,
    CONFIDENCE_INTERVAL_DEFAULT,
    ALTERNATIVE_DEFAULT
)

from app.api.helpers.logger import Logger

two_sample_t_test_api = Blueprint('two_sample_t_test', __name__)

@two_sample_t_test_api.route('/two-sample-t-test', methods=['POST'])
def perform_two_sample_t_test():
    '''
    Perform a Two-Sample T-Test based on the input json data
    Expected JSON input format:
    {
        "vehicle": [55, 45, 65, 54, 43, 45, 54, 63, 73, 36, 65],
        "drugs": [74, 85, 76, 58, 67, 47, 56, 92, 71, 93, 86]
    }
    '''
    logger = Logger(TWO_SAMPLE_T_TEST_LOG_FILE_PATH)
    try:
        data = request.get_json()
        keys = list(data.keys())

        if len(keys) < 2:
            raise ValueError("JSON input must contain at least two groups.")

        group1_key, group2_key = keys[:2]
        group1, group2 = data[group1_key], data[group2_key]

        if not isinstance(group1, list) or not isinstance(group2, list):
            raise ValueError("Both groups must be lists of numerical values.")

        alternative = ALTERNATIVE_DEFAULT
        confidence = CONFIDENCE_INTERVAL_DEFAULT

        output_data = calculate_and_format_two_sample_t_test(
            group1, group2, group1_key, group2_key, alternative, confidence
        )
        return jsonify(output_data), 200

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

def perform_normality_and_variance_tests(group1, group2):
    return {
        "shapiro_group1": stats.shapiro(group1),
        "shapiro_group2": stats.shapiro(group2),
        "equal_var_test": stats.levene(group1, group2)
    }

def compute_sample_statistics(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    sem1, sem2 = std1 / np.sqrt(len(group1)), std2 / np.sqrt(len(group2))
    return mean1, mean2, std1, std2, sem1, sem2

def perform_t_tests(group1, group2, alternative):
    t_stat_separate, p_value_separate = stats.ttest_ind(group1, group2, equal_var=False, alternative=alternative)
    df_separate = stats.ttest_ind(group1, group2, equal_var=False).df
    t_stat_pooled, p_value_pooled = stats.ttest_ind(group1, group2, equal_var=True, alternative=alternative)
    df_pooled = len(group1) + len(group2) - 2
    return t_stat_separate, p_value_separate, df_separate, t_stat_pooled, p_value_pooled, df_pooled

def compute_confidence_interval(mean_diff, std1, std2, group1_size, group2_size, confidence, df):
    t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)
    margin_of_error = t_critical * np.sqrt((std1 ** 2 / group1_size) + (std2 ** 2 / group2_size))
    return mean_diff - margin_of_error, mean_diff + margin_of_error

def calculate_and_format_two_sample_t_test(group1, group2, group1_name, group2_name, alternative, confidence):
    tests = perform_normality_and_variance_tests(group1, group2)
    mean1, mean2, std1, std2, sem1, sem2 = compute_sample_statistics(group1, group2)
    mean_difference = mean1 - mean2
    
    t_stat_separate, p_value_separate, df_separate, t_stat_pooled, p_value_pooled, df_pooled = perform_t_tests(group1, group2, alternative)
    lower_bound, upper_bound = compute_confidence_interval(mean_difference, std1, std2, len(group1), len(group2), confidence, df_separate)

    return {
        "Data Source": "Provided Data",
        "Normality Test (Shapiro-Wilk)": {
            group1_name: {"P-Value": round(tests["shapiro_group1"].pvalue, 3)},
            group2_name: {"P-Value": round(tests["shapiro_group2"].pvalue, 3)}
        },
        "Equal Variance Test (Levene's Test)": {
            "P-Value": round(tests["equal_var_test"].pvalue, 3)
        },
        "Sample Statistics": [
            {"Group": group1_name, "N": len(group1), "Mean": round(mean1, 3), "Std Dev": round(std1, 3), "SEM": round(sem1, 3)},
            {"Group": group2_name, "N": len(group2), "Mean": round(mean2, 3), "Std Dev": round(std2, 3), "SEM": round(sem2, 3)}
        ],
        "Difference of Means": round(mean_difference, 3),
        "Equal Variances Assumed (Student's t-test)": {
            "t": round(t_stat_pooled, 3),
            "df": df_pooled,
            "95% CI": [round(lower_bound, 3), round(upper_bound, 3)],
            "Two-tailed P-value": round(p_value_pooled, 5)
        },
        "Equal Variances Not Assumed (Welch's t-test)": {
            "t": round(t_stat_separate, 3),
            "df": round(df_separate, 3),
            "95% CI": [round(lower_bound, 3), round(upper_bound, 3)],
            "Two-tailed P-value": round(p_value_separate, 5)
        }
    }
