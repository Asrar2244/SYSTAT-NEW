import json
import numpy as np
import scipy.stats as stats
from flask import Blueprint, request, jsonify
from statsmodels.stats.diagnostic import lilliefors
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
    ONE_SAMPLE_T_TEST_LOG_FILE_PATH,
    CONFIDENCE_INTERVAL_DEFAULT,
    POPULATION_MEAN_DEFAULT,
    P_VALUE_REJECT_DEFAULT,
    ALPHA_VALUE_DEFAULT
)
from app.api.helpers.logger import Logger

one_sample_t_test_api = Blueprint('one_sample_t_test', __name__)

@one_sample_t_test_api.route('/one-sample-t-test', methods=['POST'])
def perform_one_sample_t_test():
    """
    Perform a One-Sample T-Test based on the input sample data or provided summary statistics.
    """
    logger = Logger(ONE_SAMPLE_T_TEST_LOG_FILE_PATH)
    try:
        data = request.get_json()
        
        population_mean = float(data.get("population_mean", POPULATION_MEAN_DEFAULT))
        confidence_level = float(data.get("confidence_level", CONFIDENCE_INTERVAL_DEFAULT))
        p_value_reject = float(data.get("P_value_reject", P_VALUE_REJECT_DEFAULT))
        alpha = float(data.get("alpha_value", ALPHA_VALUE_DEFAULT))
        
        # NOOR - true by default
        shaprio_walk = data.get("shaprio_walk", False)
        kolmo_with_correction = data.get("kolmo_with_correction", False)
        db_fetched = data.get("DB", False)
        
 # Handle input: Either "sample" or "values"
        if "sample" in data:
            sample_data = data["sample"]
            if not isinstance(sample_data, list) or len(sample_data) < 2:
                raise ValueError("Sample data must contain at least two data points.")
            sample_size = len(sample_data)
            sample_mean = np.mean(sample_data)
            sample_std = np.std(sample_data, ddof=1)

        elif "values" in data:
            values = data["values"]
            sample_size = values.get("size")
            sample_mean = values.get("mean")
            sample_std = values.get("deviation")
            standard_error = values.get("standard_error")

            # Validate values input
            if not all(isinstance(x, (int, float)) for x in [sample_size, sample_mean]) or (sample_std is None and standard_error is None):
                raise ValueError("Invalid input: 'size', 'mean', and either 'deviation' or 'standard_error' must be numeric.")

            if sample_std is None and standard_error is None:
                raise ValueError("Either 'deviation' or 'standard_error' must be provided.")
            
            if sample_std is not None:
                if not isinstance(sample_std, (int, float)):
                    raise ValueError("Invalid input: 'deviation' must be numeric.")
                standard_error = sample_std / np.sqrt(sample_size)
            else:
                if not isinstance(standard_error, (int, float)):
                    raise ValueError("Invalid input: 'standard_error' must be numeric.")
                sample_std = standard_error * np.sqrt(sample_size)

            if sample_size < 2:
                raise ValueError("Sample size must be at least 2.")
        
        else:
            raise KeyError("Missing required input: Provide either 'sample' or 'values'.")


        standard_error = sample_std / np.sqrt(sample_size)
        degrees_of_freedom = sample_size - 1
        
        # Compute t-statistic and p-value
        t_stat = (sample_mean - population_mean) / standard_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_of_freedom))
        p_value_two_tailed = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_of_freedom))  # Two-tailed p-value
        p_value_one_tailed = 1 - stats.t.cdf(abs(t_stat), degrees_of_freedom)  # One-tailed p-value
        

        # Compute confidence interval
        t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_critical * standard_error
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error

        conclusion = (
            "The null hypothesis is rejected, indicating a significant difference."
            if p_value < p_value_reject else 
            f"The null hypothesis is not rejected (p-value: {round(p_value, 3)})."
        )

        result = {
            "Test Type": "One-Sample t-test",
            "Sample Statistics": {
                "Sample Size": sample_size,
                "Sample Mean": round(sample_mean, 3),
                "Sample Std Dev": round(sample_std, 3),
                "Standard Error of Mean": round(standard_error, 3),
                "Degrees of Freedom": degrees_of_freedom,
                "Hypothesized Population Mean": population_mean,
                "t-Statistic": round(t_stat, 3),
                "P-Value": round(p_value, 3),
                "Two-tailed P-Value": round(p_value_two_tailed, 8),
                "One-tailed P-Value": round(p_value_one_tailed, 8),

                "Confidence Interval": {
                    "Lower Bound": round(lower_bound, 3),
                    "Upper Bound": round(upper_bound, 3)
                }
            },
            "Conclusion": conclusion
        }
        
        # Perform normality tests if specified
        if db_fetched and "sample" in data:
            normality_tests = {}
            
            if shaprio_walk:
                shapiro_test_stat, shapiro_p_value = stats.shapiro(sample_data)
                normality_tests["Shapiro-Wilk"] = {
                    "Result": "Passed" if shapiro_p_value > 0.05 else "Failed",
                    "P-Value": round(shapiro_p_value, 3)
                }
            
            if kolmo_with_correction:
                ks_stat, ks_p_value = lilliefors(sample_data)
                normality_tests["Kolmogorov-Smirnov"] = {
                    "Result": "Passed" if ks_p_value > 0.05 else "Failed",
                    "P-Value": round(ks_p_value, 3)
                }
            
            result["Normality Tests"] = normality_tests

        # Compute statistical power
        effect_size = abs(sample_mean - population_mean) / sample_std
        power_two_tailed = stats.nct.sf(stats.t.ppf(1 - alpha / 2, degrees_of_freedom), degrees_of_freedom, effect_size * np.sqrt(sample_size))
        power_one_tailed = stats.nct.sf(stats.t.ppf(1 - alpha, degrees_of_freedom), degrees_of_freedom, effect_size * np.sqrt(sample_size))
            
        result["Statistical Power"] = {
                f"Power of performed two-tailed test with alpha = {alpha}": round(power_two_tailed, 3),
                "Power Conclusion Two-tailed": f"The power of the performed test ({round(power_two_tailed, 3)}) is below the desired power of 0.800. Less than desired power indicates you are less likely to detect a difference when one actually exists. Negative results should be interpreted cautiously.",
                f"Power of performed one-tailed test with alpha = {alpha}": round(power_one_tailed, 3),
                "Power Conclusion One-tailed": f"The power of the performed test ({round(power_one_tailed, 3)}) is below the desired power of 0.800. Less than desired power indicates you are less likely to detect a difference when one actually exists. Negative results should be interpreted cautiously."
            }

    
        logger.info(json.dumps(result))
        return jsonify(result), 200

    except ValueError as ve:
        logger.error(LOG_VALUE_ERROR.format(str(ve)))
        return jsonify({"error": VALUE_ERROR_MSG.format(str(ve))}), 400
    except KeyError as ke:
        logger.error(LOG_KEY_ERROR.format(str(ke)))
        return jsonify({"error": KEY_ERROR_MSG.format(str(ke))}), 400
    except TypeError as te:
        logger.error(LOG_TYPE_ERROR.format(str(te)))
        return jsonify({"error": TYPE_ERROR_MSG}), 400
    except ZeroDivisionError as zde:
        logger.error(LOG_ZERO_DIVISION_ERROR.format(str(zde)))
        return jsonify({"error": ZERO_DIVISION_ERROR_MSG}), 400
    except IndexError as ie:
        logger.error(LOG_INDEX_ERROR.format(str(ie)))
        return jsonify({"error": INDEX_ERROR_MSG}), 400
    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500
