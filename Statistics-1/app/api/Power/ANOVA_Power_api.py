"""
Module for computing the statistical power of an ANOVA test.
Accepts input in JSON format, calculates the power, and returns the result.
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
    ANOVA_POWER_LOG_FILE_PATH
)

anova_power_api = Blueprint('anova_power', __name__)

@anova_power_api.route('/anova-power', methods=['POST'])
def calculate_anova_power():
    """
    Compute the statistical power for an ANOVA test.

    Expected JSON input format:
        {
            "minimum_detectable_difference": 5.0,
            "expected_std_dev_residuals": 10.0,
            "num_groups": 3,
            "group_size": 80,
            "alpha": 0.05
        }

    Returns:
        {
            "power": 0.82
        }
    """
    logger = Logger(ANOVA_POWER_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        # Extract parameters
        mdd = float(input_data.get("minimum_detectable_difference"))
        sigma_residuals = float(input_data.get("expected_std_dev_residuals"))
        num_groups = int(input_data.get("num_groups"))
        group_size = int(input_data.get("group_size"))
        alpha = float(input_data.get("alpha"))

        if num_groups < 2:
            raise ValueError("ANOVA requires at least two groups.")

        if mdd <= 0 or sigma_residuals <= 0:
            raise ValueError("Minimum detectable difference and residual standard deviation must be positive.")

        if group_size <= 0:
            raise ValueError("Group size must be a positive integer.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Adjusted Cohen's f formula
        effect_size = mdd / (1.41 * sigma_residuals)  # More aligned with SigmaPlot

        # Compute power using power analysis
        analysis = FTestAnovaPower()
        power = analysis.solve_power(effect_size=effect_size, alpha=alpha, k_groups=num_groups, nobs=group_size)

        # Ensure power is between 0 and 1
        power = max(0, min(1, power))

        result = {"power": round(power, 4)}

        logger.info(json.dumps(result))
        logger.info("ANOVA power calculation completed successfully.")

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
