"""
Module for computing the required sample size for a two-proportion Z-test.
Accepts input in JSON format, calculates the required sample size, 
and returns the result.
"""

import json
from flask import Blueprint, request, jsonify
from scipy.stats import norm
import numpy as np
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
    SAMPLE_SIZE_PROPORTION_TEST_LOG_FILE_PATH
)

proportion_sample_size_api = Blueprint('proportion_sample_size', __name__)

@proportion_sample_size_api.route('/proportion-sample-size', methods=['POST'])
def calculate_proportion_sample_size():
    """
    Compute required sample size for a two-proportion Z-test.

    Expected JSON input format:
        {
            "group1_proportion": 0.5,
            "group2_proportion": 0.6,
            "desired_power": 0.8,
            "alpha": 0.05,
            "yates_correction": false
        }

    Returns:
        {
            "sample_size": 388
        }
    """
    logger = Logger(SAMPLE_SIZE_PROPORTION_TEST_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        # Extract input parameters
        p1 = float(input_data.get("group1_proportion"))
        p2 = float(input_data.get("group2_proportion"))
        power = float(input_data.get("desired_power"))
        alpha = float(input_data.get("alpha"))
        yates_correction = bool(input_data.get("yates_correction", False))  # Default: False

        if not (0 < p1 < 1) or not (0 < p2 < 1):
            raise ValueError("Proportions must be between 0 and 1 (exclusive).")

        if not (0 < power < 1):
            raise ValueError("Desired power must be between 0 and 1.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Compute required Z-scores
        z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed test
        z_beta = norm.ppf(power)

        # Compute pooled variance
        pooled_variance = (p1 * (1 - p1) + p2 * (1 - p2))

        # Compute effect size
        effect_size = abs(p1 - p2)

        # Apply Yates' correction (only for small samples)
        if yates_correction and effect_size > 0.02:  # Apply correction only if effect size is small
            effect_size -= 0.01

        # Compute required sample size per group
        sample_size = ((z_alpha + z_beta) ** 2 * pooled_variance) / effect_size ** 2

        # Ensure a minimum sample size
        sample_size = max(sample_size, 10)

        result = {"sample_size": round(sample_size)}

        logger.info(json.dumps(result))
        logger.info("Proportion test sample size calculation completed successfully.")

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
