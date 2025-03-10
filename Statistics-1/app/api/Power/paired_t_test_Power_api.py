import json
from flask import Blueprint, request, jsonify
from statsmodels.stats.power import TTestPower
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
    POWER_PAIRED_TTEST_LOG_FILE_PATH
)

paired_ttest_power_api = Blueprint('paired_ttest_power', __name__)

@paired_ttest_power_api.route('/paired-ttest-power', methods=['POST'])
def calculate_paired_ttest_power():
    """
    Compute power for a paired t-test given the sample size.

    Expected JSON input format:
        {
            "change_to_be_detected": 5.0,
            "expected_std_dev_of_change": 10.0,
            "desired_sample_size": 34,
            "alpha": 0.05,
        }

    Returns:
        {
            "power": 0.85
        }
    """
    logger = Logger(POWER_PAIRED_TTEST_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        # Extract input parameters
        mean_diff = float(input_data.get("change_to_be_detected"))
        std_dev_change = float(input_data.get("expected_std_dev_of_change"))
        sample_size = int(input_data.get("desired_sample_size"))
        alpha = float(input_data.get("alpha"))
        correlation = float(input_data.get("correlation", 0.5))  # Default correlation = 0.5

        if mean_diff <= 0 or std_dev_change <= 0:
            raise ValueError("Change to be detected and standard deviation must be positive.")

        if sample_size <= 0:
            raise ValueError("Desired sample size must be a positive integer.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        if not (-1 < correlation < 1):
            raise ValueError("Correlation must be between -1 and 1.")

        logger.info(f"Received data: {input_data}")

        # Adjust standard deviation using correlation
        adjusted_std_dev = std_dev_change * np.sqrt(2 * (1 - correlation))

        # Compute effect size (Cohen's dz for paired test)
        effect_size = mean_diff / adjusted_std_dev

        # Perform power calculation
        analysis = TTestPower()
        power = analysis.solve_power(effect_size=effect_size, nobs=sample_size, alpha=alpha)

        result = {"power": round(power, 4)}

        logger.info(json.dumps(result))
        logger.info("Paired t-test power calculation completed successfully.")

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
