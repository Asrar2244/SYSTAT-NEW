"""
Module for computing the required sample size for a two-sample T-test. 
Accepts input in JSON format, calculates the required sample size per group, 
and returns the result.
"""

import json
from flask import Blueprint, request, jsonify
from statsmodels.stats.power import TTestIndPower
from app.api.helpers.logger import Logger
from app.api.helpers.constant import (
    VALUE_ERROR_MSG,
    KEY_ERROR_MSG,
    TYPE_ERROR_MSG,
    UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR,
    LOG_KEY_ERROR,
    LOG_TYPE_ERROR,
    LOG_UNEXPECTED_ERROR,
    SAMPLE_SIZE_TTEST_LOG_FILE_PATH
)

ttest_sample_size_api = Blueprint('ttest_sample_size', __name__)

@ttest_sample_size_api.route('/ttest-sample-size', methods=['POST'])
def calculate_ttest_sample_size():
    """
    Compute required sample size for a two-sample T-test.

    Expected JSON input format:
        {
            "expected_difference": 5.0,
            "expected_std_dev": 10.0,
            "desired_power": 0.8,
            "alpha": 0.05
        }

    Returns:
        {
            "sample_size_per_group": 64
        }
    """
    logger = Logger(SAMPLE_SIZE_TTEST_LOG_FILE_PATH)

    try:
        input_data = request.get_json()
        
        # Extract parameters
        diff = float(input_data.get("expected_difference"))
        std_dev = float(input_data.get("expected_std_dev"))
        power = float(input_data.get("desired_power"))
        alpha = float(input_data.get("alpha"))

        if diff <= 0 or std_dev <= 0:
            raise ValueError("Expected difference and standard deviation must be positive.")

        if not (0 < power < 1):
            raise ValueError("Desired power must be between 0 and 1.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Compute effect size (Cohen's d)
        effect_size = diff / std_dev

        # Perform sample size calculation
        analysis = TTestIndPower()
        sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1)

        result = {"sample_size": round(sample_size)}

        logger.info(json.dumps(result))
        logger.info("T-test sample size calculation completed successfully.")

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
    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500
