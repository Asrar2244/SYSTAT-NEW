"""
Module for computing the statistical power of a Chi-Square test.
Accepts input in JSON format, calculates the power, and returns the result.
"""

import json
import numpy as np
from flask import Blueprint, request, jsonify
from statsmodels.stats.power import GofChisquarePower
from scipy.stats import chi2_contingency
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
    CHISQUARE_POWER_LOG_FILE_PATH
)

chisquare_power_api = Blueprint('chisquare_power', __name__)

@chisquare_power_api.route('/chisquare-power', methods=['POST'])
def calculate_chisquare_power():
    """
    Compute the statistical power for a Chi-Square test.
    
    Expected JSON input format:
        {
            "data": [[30, 40, 50], [20, 35, 45]],  # Contingency table (wide format)
            "desired_sample_size": 200,
            "alpha": 0.05,
            "yates_correction": true  # Optional, only applies to 2×2 tables
        }

    Returns:
        {
            "power": 0.82
        }
    """
    logger = Logger(CHISQUARE_POWER_LOG_FILE_PATH)

    try:
        input_data = request.get_json()

        # Extract parameters
        data = input_data.get("data")
        sample_size = int(input_data.get("desired_sample_size", 0))
        alpha = float(input_data.get("alpha", 0))
        yates_correction = bool(input_data.get("yates_correction", False))

        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a list of lists representing a contingency table.")

        if sample_size <= 0:
            raise ValueError("Desired sample size must be a positive integer.")

        if not (0 < alpha < 1):
            raise ValueError("Alpha (significance level) must be between 0 and 1.")

        logger.info(f"Received data: {input_data}")

        # Convert data to numpy array
        observed = np.array(data)

        # Determine if it's a 2×2 table
        if observed.shape == (2, 2) and yates_correction:
            logger.info("Applying Yates' correction for 2×2 table.")
            chi2_stat, _, _, expected = chi2_contingency(observed, correction=True)
        else:
            chi2_stat, _, _, expected = chi2_contingency(observed, correction=False)

        # Compute effect size (Cramér’s V) for larger tables
        n_total = np.sum(observed)
        min_dim = min(observed.shape) - 1  # Adjust for larger tables
        effect_size = np.sqrt(chi2_stat / (n_total * (min_dim)))  

        if effect_size == 0:
            raise ValueError("Effect size is zero, check input data.")

        # Compute power using effect size
        analysis = GofChisquarePower()
        power = analysis.solve_power(effect_size=effect_size, nobs=sample_size, alpha=alpha)

        # Ensure power is between 0 and 1
        power = max(0, min(1, power))

        result = {"power": round(power, 4)}

        logger.info(json.dumps(result))
        logger.info("Chi-Square power calculation completed successfully.")

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
