from flask import Blueprint, request, jsonify
from .helpers.constant import Z_TEST_LOG_FILE_PATH
from .helpers.logger import Logger
import math
from scipy.stats import norm

ztest_api = Blueprint('ztest_api', __name__)

@ztest_api.route('/z-test', methods=['POST'])
def perform_z_test():
    logger = Logger(Z_TEST_LOG_FILE_PATH)
    try:
        # Get data from request
        data = request.get_json()

        # Extract parameters and input data
        alpha_value = data.get('Alpha_value', 0.05)
        yates_correction = data.get('Yates_correction', 0)
        confidence_interval = data.get('Confidence_interval', 95)
        input_data = data.get('Data', [])
        
        logger.info(f"Received data: {data}")

        # Input validation
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

        # Log the input data
        logger.info(f"Group 1: Size={size1}, Proportion={prop1}")
        logger.info(f"Group 2: Size={size2}, Proportion={prop2}")

        # Apply Yates continuity correction if required
        if yates_correction == 1:
            logger.info("Yates continuity correction applied to calculations.")

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

        # Log the results
        logger.info(f"Z-score: {z_score}")
        logger.info(f"P-value: {p_value}")
        logger.info(f"Confidence Interval: ({confidence_interval_lower}, {confidence_interval_upper})")
        logger.info(f"Power of the test: {power}")
        logger.info(f"Conclusion: {conclusion}")

        # Prepare the result for output
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

        # Return the result as JSON
        return jsonify(result), 200

    except ValueError as ve:
        # Handle invalid value errors (e.g., invalid proportions or sizes)
        logger.error(f"ValueError: {str(ve)}")
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except KeyError as ke:
        # Handle missing keys in the input JSON
        logger.error(f"KeyError: {str(ke)}")
        return jsonify({"error": f"Missing required field: {str(ke)}"}), 400
    except TypeError as te:
        # Handle type errors (e.g., wrong data types)
        logger.error(f"TypeError: {str(te)}")
        return jsonify({"error": f"Invalid data type: {str(te)}"}), 400
    except ZeroDivisionError as zde:
        # Handle division by zero (e.g., when sizes are zero)
        logger.error(f"ZeroDivisionError: {str(zde)}")
        return jsonify({"error": "Division by zero encountered during calculation."}), 400
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

