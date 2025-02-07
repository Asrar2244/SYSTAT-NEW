'''This module defines an API endpoint for performing a One-Sample T-Test on input data.'''
import json
import pandas as pd
from statsmodels.stats.weightstats import ztest
import scipy.stats as stats
import numpy as np
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
    ONE_SAMPLE_T_TEST_LOG_FILE_PATH,
    CONFIDENCE_INTERVAL_DEFAULT,
    POPULATION_MEAN_DEFAULT,
    ALTERNATIVE_DEFAULT
)
from app.api.helpers.logger import Logger

one_sample_t_test_api = Blueprint('one_sample_t_test', __name__)

@one_sample_t_test_api.route('/one-sample-t-test', methods=['POST'])
def perform_one_sample_t_test():
    """
    Perform a One-Sample T-Test based on the input sample data and population mean.

    Expected JSON input format:
    {
        "sample": [55, 45, 65, 54, 43, 45, 54, 63, 73, 36, 65],
        "population_mean": 50,
        "alternative": "two-sided",  # Options: "two-sided", "greater", "less"
        "confidence_level": 0.95
    }
    """
    logger = Logger(ONE_SAMPLE_T_TEST_LOG_FILE_PATH)
    try:
        data = request.get_json()

        sample_data = data.get("sample")
        population_mean = float(data.get("population_mean", POPULATION_MEAN_DEFAULT))
        alternative = data.get("alternative", ALTERNATIVE_DEFAULT).lower()
        confidence_level = float(data.get("confidence_level", CONFIDENCE_INTERVAL_DEFAULT))

        if not sample_data or len(sample_data) < 2:
            raise ValueError("Sample data must contain at least two data points.")
        
        sample_size = len(sample_data)
        sample_mean = np.mean(sample_data)
        sample_std = np.std(sample_data, ddof=1)
        degrees_of_freedom = sample_size - 1

        # One-Sample T-Test using scipy
        t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

        # Shapiro-Wilk test for normality
        shapiro_test_stat, shapiro_p_value = stats.shapiro(sample_data)
        normality_test_result = "Passed" if shapiro_p_value > 0.05 else "Failed"

        # Adjust p-value based on alternative hypothesis
        if alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, degrees_of_freedom)
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, degrees_of_freedom)
        elif alternative != "two-sided":
            raise ValueError(f"Invalid alternative hypothesis: {alternative}. Expected 'greater', 'less', or 'two-sided'.")

        # Compute confidence interval
        t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error

        conclusion = (
            "The null hypothesis is rejected, indicating a significant difference."
            if p_value < 0.05 else 
            f"The null hypothesis is not rejected (p-value: {p_value:.3f})."
        )

        result = {
            "Test Type": "One-Sample t-test",
            "Normality Test (Shapiro-Wilk)": {
                "Result": normality_test_result,
                "P-Value": round(shapiro_p_value, 3)
            },
            "Sample Statistics": {
                "Sample Size": sample_size,
                "Sample Mean": round(sample_mean, 3),
                "Sample Std Dev": round(sample_std, 3),
                "Standard Error of Mean": round(sample_std / np.sqrt(sample_size), 3),
                "Degrees of Freedom": degrees_of_freedom,
                "Hypothesized Population Mean": population_mean,
                "t-Statistic": round(t_stat, 3),
                "Two-tailed P-Value": round(p_value, 3),
                "One-tailed P-Value": round(p_value / 2, 3),
                "95% Confidence Interval": {
                    "Lower Bound": round(lower_bound, 3),
                    "Upper Bound": round(upper_bound, 3)
                }
            },
            "Conclusion": conclusion
        }

        logger.info(json.dumps(result))
        logger.info("One-sample t-test completed successfully.")
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
