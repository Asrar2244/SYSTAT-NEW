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
    PROPORTION_TEST_POWER_LOG_FILE_PATH
)

proportion_power_api = Blueprint('proportion_power', __name__)

@proportion_power_api.route('/proportion-power', methods=['POST'])
def calculate_proportion_power():
    """
    Compute statistical power for a two-proportion Z-test.

    Expected JSON input format:
        {
            "group1_proportion": 0.5,
            "group2_proportion": 0.6,
            "group1_size": 200,
            "group2_size": 200,
            "alpha": 0.05,
            "yates_correction": true
        }

    Returns:
        {
            "power": 0.48
        }
    """
    logger = Logger(PROPORTION_TEST_POWER_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        # Extract input parameters
        p1 = float(input_data.get("group1_proportion"))
        p2 = float(input_data.get("group2_proportion"))
        n1 = int(input_data.get("group1_size"))
        n2 = int(input_data.get("group2_size"))
        alpha = float(input_data.get("alpha"))
        yates_correction = bool(input_data.get("yates_correction", False))  # Default to False

        if not (0 < p1 < 1) or not (0 < p2 < 1):
            raise ValueError("Proportions must be between 0 and 1 (exclusive).")

        if n1 <= 0 or n2 <= 0:
            raise ValueError("Group sizes must be positive integers.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Compute pooled standard error
        pooled_se = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))

        # Compute raw effect size
        effect_size = abs(p1 - p2)

        # Apply Yates' correction (only if enabled)
        if yates_correction:
            yates_adjustment = (1 / min(n1, n2))  # Small correction term
            effect_size = max(0, effect_size - yates_adjustment)  # Ensure non-negative effect size

        # Compute test statistic (Z-score for effect size)
        z_effect = effect_size / pooled_se

        # Compute critical Z-score for alpha (two-tailed test)
        z_alpha = norm.ppf(1 - alpha / 2)

        # Compute power using the correct formula
        power = norm.cdf(z_effect - z_alpha)

        # Ensure power is between 0 and 1
        power = max(0, min(1, power))

        result = {"power": round(power, 4)}

        logger.info(json.dumps(result))
        logger.info("Proportion test power calculation completed successfully.")

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
