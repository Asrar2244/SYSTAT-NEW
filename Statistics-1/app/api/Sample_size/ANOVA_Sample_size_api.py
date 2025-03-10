"""
Module for computing the required sample size for an ANOVA test.
Accepts input in JSON format, calculates the required sample size per group, 
and returns the result.
"""

import json
from flask import Blueprint, request, jsonify
from statsmodels.stats.power import FTestAnovaPower
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
    SAMPLE_SIZE_ANOVA_LOG_FILE_PATH
)

anova_sample_size_api = Blueprint('anova_sample_size', __name__)

@anova_sample_size_api.route('/anova-sample-size', methods=['POST'])
def calculate_anova_sample_size():
    """
    Compute required sample size for an ANOVA test.

    Expected JSON input format:
        {
            "minimum_detectable_difference": 5.0,
            "expected_std_dev_residuals": 10.0,
            "num_groups": 3,
            "desired_power": 0.8,
            "alpha": 0.05
        }

    Returns:
        {
            "sample_size_per_group": 79
        }
    """
    logger = Logger(SAMPLE_SIZE_ANOVA_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        # Extract parameters
        mdd = float(input_data.get("minimum_detectable_difference"))
        sigma_residuals = float(input_data.get("expected_std_dev_residuals"))
        num_groups = int(input_data.get("num_groups"))
        power = float(input_data.get("desired_power"))
        alpha = float(input_data.get("alpha"))

        if num_groups < 2:
            raise ValueError("ANOVA requires at least two groups.")

        if mdd <= 0 or sigma_residuals <= 0:
            raise ValueError("Minimum detectable difference and residual standard deviation must be positive.")

        if not (0 < power < 1):
            raise ValueError("Desired power must be between 0 and 1.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Adjusted Cohen's f formula
        effect_size = mdd / (1.41 * sigma_residuals)  # More aligned with SigmaPlot

        # Compute sample size using power analysis
        analysis = FTestAnovaPower()
        sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, k_groups=num_groups)

        # Ensure rounding up
        sample_size_adjusted = np.ceil(sample_size)

        result = {"sample_size": int(sample_size_adjusted)}

        logger.info(json.dumps(result))
        logger.info("ANOVA sample size calculation completed successfully.")

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
