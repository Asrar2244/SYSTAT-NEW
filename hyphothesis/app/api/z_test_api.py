"""Module for performing a two-proportion Z-test with statistical calculations."""
import math
from flask import Blueprint, request, jsonify
from scipy.stats import norm
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
    Z_TEST_LOG_FILE_PATH,
    ALPHA_VALUE_DEFAULT,
    YATES_CORRECTION_DEFAULT,
    CONFIDENCE_INTERVAL_DEFAULT
)
from app.api.helpers.logger import Logger

ztest_api = Blueprint('ztest_api', __name__)

@ztest_api.route('/z-test', methods=['POST'])
def perform_z_test():
    """Perform a two-proportion Z-test and return the statistical results and conclusion."""
    """
    Expected JSON input format:
    {
        "Alpha_value": 0.05,
        "Yates_correction": 0,
        "Confidence_interval": 95,
        "Data": [
            [40, 0.3],
            [60, 0.7]
        ]
    }
    """
    logger = Logger(Z_TEST_LOG_FILE_PATH)
    try:
        data = request.get_json()

        alpha_value = data.get('Alpha_value', ALPHA_VALUE_DEFAULT)
        yates_correction = data.get('Yates_correction', YATES_CORRECTION_DEFAULT)
        confidence_interval = data.get('Confidence_interval', CONFIDENCE_INTERVAL_DEFAULT)
        input_data = data.get('Data', [])

        logger.info(f"Received data: {data}")

        if not (0 <= alpha_value <= 1):
            logger.error(f"Invalid Alpha_value: {alpha_value}. Must be between 0 and 1.")
            return jsonify({"error": "Alpha_value must be between 0 and 1."}), 400
        if not (0 <= yates_correction <= 1):
            logger.error(f"Invalid Yates_correction: {yates_correction}. Must be either 0 or 1.")
            return jsonify({"error": "Yates_correction must be either 0 or 1."}), 400
        if not (1 <= confidence_interval <= 99):
            logger.error(f"Invalid Confidence_interval: {confidence_interval}. Must be between 1 and 99.")
            return jsonify({"error": "Confidence_interval must be between 1 and 99."}), 400
        if len(input_data) != 2 or len(input_data[0]) != 2 or len(input_data[1]) != 2:
            logger.error(f"Invalid Data: {input_data}. Data must contain two rows and two columns.")
            return jsonify({"error": "Data must contain two rows and two columns."}), 400

        size1, prop1 = input_data[0]
        size2, prop2 = input_data[1]

        if not (0 <= prop1 <= 1 and 0 <= prop2 <= 1):
            logger.error(f"Invalid Proportions: {prop1}, {prop2}. Proportions must be between 0 and 1.")
            return jsonify({"error": "Proportions must be between 0 and 1."}), 400

        # Calculate the Z-test
        # Pooled proportion
        pooled_p = (size1 * prop1 + size2 * prop2) / (size1 + size2)
        standard_error = math.sqrt(pooled_p * (1 - pooled_p) * ((1 / size1) + (1 / size2)))
        z_score = (prop1 - prop2) / standard_error
        p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed p-value

        # Confidence interval
        z_critical = norm.ppf(1 - (1 - confidence_interval / 100) / 2)
        margin_of_error = z_critical * standard_error
        confidence_interval_lower = (prop1 - prop2) - margin_of_error
        confidence_interval_upper = (prop1 - prop2) + margin_of_error

        # Power of the test (using alpha value)
        power = 1 - norm.cdf(z_critical - abs(z_score))

        # Conclusion based on p-value
        conclusion = "There is a significant difference in the proportions." if p_value < alpha_value else "No significant difference in the proportions."

        logger.info(f"Z-score: {z_score}")
        logger.info(f"P-value: {p_value}")
        logger.info(f"Confidence Interval: ({confidence_interval_lower}, {confidence_interval_upper})")
        logger.info(f"Power of the test: {power}")
        logger.info(f"Conclusion: {conclusion}")

        result = {
            "message": "Z-test calculation successful",
            "alpha_value": alpha_value,
            "yates_correction": yates_correction,
            "confidence_interval": confidence_interval,
            "group_1": {"size": size1, "proportion": prop1},
            "group_2": {"size": size2, "proportion": prop2},
            "results": {
                "difference_of_sample_proportions": prop1 - prop2,
                "pooled_estimate_for_p": pooled_p,
                "standard_error_of_difference": standard_error,
                "z_score": z_score,
                "p_value": p_value,
                "confidence_interval": {
                    "lower_bound": confidence_interval_lower,
                    "upper_bound": confidence_interval_upper
                },
                "power_of_test": power,
                "conclusion": conclusion
            }
        }

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
