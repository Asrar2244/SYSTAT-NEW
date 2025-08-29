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
    TTEST_POWER_LOG_FILE_PATH
)

ttest_power_api = Blueprint('ttest_power', __name__)

@ttest_power_api.route('/ttest-power', methods=['POST'])
def calculate_ttest_power():
    """
    Compute the power of a two-sample T-test.

    Expected JSON input format:
        {
            "expected_difference": 5.0,
            "expected_std_dev": 10.0,
            "group1_size": 30,
            "group2_size": 30,
            "alpha": 0.05
        }

    Returns:
        {
            "power": 0.85
        }
    """
    logger = Logger(TTEST_POWER_LOG_FILE_PATH)

    try:
        input_data = request.get_json()
        
        # Extract parameters
        diff = float(input_data.get("expected_difference"))
        std_dev = float(input_data.get("expected_std_dev"))
        n1 = int(input_data.get("group1_size"))
        n2 = int(input_data.get("group2_size"))
        alpha = float(input_data.get("alpha"))

        if diff <= 0 or std_dev <= 0:
            raise ValueError("Expected difference and standard deviation must be positive.")

        if n1 <= 0 or n2 <= 0:
            raise ValueError("Group sizes must be positive integers.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Compute effect size (Cohen's d)
        effect_size = diff / std_dev

        # Perform power calculation
        analysis = TTestIndPower()
        power = analysis.power(effect_size=effect_size, nobs1=n1, alpha=alpha, ratio=n2/n1)

        result = {"power": round(power, 4)}

        logger.info(json.dumps(result))
        logger.info("T-test power calculation completed successfully.")

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