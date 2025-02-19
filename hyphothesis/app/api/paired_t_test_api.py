import json
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.power import TTestPower
from flask import Blueprint, request, jsonify
import pandas as pd  # Required for converting long format to wide format

from app.api.helpers.constant import (
    CONFIDENCE_INTERVAL_DEFAULT,
    ALPHA_VALUE_DEFAULT,
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
    PAIRED_T_TEST_LOG_FILE_PATH
)
from app.api.helpers.logger import Logger

paired_t_test_api = Blueprint('paired_t_test_api', __name__)

@paired_t_test_api.route('/paired-t-test-api', methods=['POST'])
def perform_paired_t_test():
    logger = Logger(PAIRED_T_TEST_LOG_FILE_PATH)

    try:
        if not request.is_json:
            return jsonify({"error": "Invalid input. Please provide JSON data."}), 400

        data = request.get_json()
        logger.info(f"Received data: {data}")

        # Extract parameters
        confidence_level = float(data.get("confidence_level", CONFIDENCE_INTERVAL_DEFAULT))
        alpha = float(data.get("alpha", ALPHA_VALUE_DEFAULT))
        # NOOR - this is True by default
        shaprio_walk = data.get("shaprio_walk", False)
        kolmo_with_correction = data.get("kolmo_with_correction", False)
        db_fetched = data.get("db", False)  # Ensure 'db' is fetched

        if not (0 < confidence_level < 1):
            return jsonify({"error": "Confidence level must be between 0 and 1."}), 400

        # Determine Input Format
        if "before" in data and "after" in data:
            before, after = data["before"], data["after"]
            before_col, after_col = "before", "after"

        # NOOR - check for some data validity - if pivot func on line 70 catches any data mismatch then
        # that would be sufficient
        # treatment - contains only two alternating string (like before/after, today/tomorrow, now/then etc)
        # each subject - has 2 values of each treatment type
        # values - int/float
        elif "subject" in data and "treatment" in data and "values" in data:
            # Convert long format to wide format
            df = pd.DataFrame({
                "subject": data["subject"],
                "treatment": data["treatment"],
                "values": data["values"]
            })

            # Pivot the data to wide format
            wide_df = df.pivot(index="subject", columns="treatment", values="values")
            # NOOR - this may not always be the case, can't we have other tratment strings other than befor/after ?
            if "before" not in wide_df.columns or "after" not in wide_df.columns:
                return jsonify({"error": "Invalid format: Missing 'before' or 'after' values."}), 400

            # Extract paired values
            before = wide_df["before"].tolist()
            after = wide_df["after"].tolist()
            before_col, after_col = "before", "after"

        else:
            return jsonify({"error": "Invalid input format. Provide 'before'/'after' lists or 'subject', 'treatment', 'values'."}), 400

        # Ensure equal length for paired test
        if len(before) != len(after) or len(before) < 2:
            return jsonify({"error": "Lists 'before' and 'after' must have the same length and at least 2 values."}), 400

        # Perform the Paired T-Test
        result = calculate_paired_t_test(before, after, before_col, after_col, confidence_level, alpha, shaprio_walk, db_fetched, kolmo_with_correction)

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
        return jsonify({"error": TYPE_ERROR_MSG.format(str(te))}), 400
    except ZeroDivisionError as zde:
        logger.error(LOG_ZERO_DIVISION_ERROR.format(str(zde)))
        return jsonify({"error": ZERO_DIVISION_ERROR_MSG.format(str(zde))}), 400
    except IndexError as ie:
        logger.error(LOG_INDEX_ERROR.format(str(ie)))
        return jsonify({"error": INDEX_ERROR_MSG.format(str(ie))}), 400   
    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG.format(str(e))}), 500

def calculate_paired_t_test(before, after, before_col, after_col, confidence_level, alpha, shaprio_walk, db_fetched, kolmo_with_correction):
    differences = np.array(before) - np.array(after)  # Corrected: before - after

    # Compute sample statistics
    mean_before, mean_after = np.mean(before), np.mean(after)
    std_dev_before, std_dev_after = np.std(before, ddof=1), np.std(after, ddof=1)
    mean_diff, std_diff = np.mean(differences), np.std(differences, ddof=1)
    sem_diff = std_diff / np.sqrt(len(before))

    # Perform Paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)
    df = len(before) - 1

    # Compute confidence interval
    ci_low, ci_high = stats.t.interval(confidence_level, df, loc=mean_diff, scale=sem_diff)
    one_tailed_p = p_value / 2

    # Perform normality tests
    normality_tests = {}
    if shaprio_walk:
            shapiro_test_stat, shapiro_p_value = stats.shapiro(differences)
            normality_tests["Shapiro-Wilk"] = {
                "Result": "Passed" if shapiro_p_value > 0.05 else "Failed",
                "P-Value": round(shapiro_p_value, 3)
            }

    if kolmo_with_correction:
            ks_stat, ks_p_value = lilliefors(differences)
            normality_tests["Kolmogorov-Smirnov"] = {
                "Result": "Passed" if ks_p_value > 0.05 else "Failed",
                "P-Value": round(ks_p_value, 3)
            }

    # Compute power analysis
    power_results = {}
    if alpha is not None:
        power_analysis = TTestPower()
        power_two_tailed = power_analysis.solve_power(effect_size=mean_diff / std_diff, nobs=len(before), alpha=alpha, alternative='two-sided')
        power_one_tailed = power_analysis.solve_power(effect_size=mean_diff / std_diff, nobs=len(before), alpha=alpha, alternative='smaller')

        power_results = {
            f"Power of performed two-tailed test with alpha = {alpha:.3f}": round(power_two_tailed, 3),
            f"Power of performed one-tailed test with alpha = {alpha:.3f}": round(power_one_tailed, 3)
        }

    # Generate the final output
    result = {
        "Paired t-test": "15 February 2025 22:20:19",
        "Data Source": "Data 1 in Notebook1.JNB",
        "Normality Test": normality_tests,
        "Sample Statistics": {
            before_col: {
                "N": len(before),
                "Missing": 0,
                "Mean": round(mean_before, 3),
                "Std Dev": round(std_dev_before, 3),
                "SEM": round(std_dev_before / np.sqrt(len(before)), 3)
            },
            after_col: {
                "N": len(after),
                "Missing": 0,
                "Mean": round(mean_after, 3),
                "Std Dev": round(std_dev_after, 3),
                "SEM": round(std_dev_after / np.sqrt(len(after)), 3)
            },
            "Difference": {
                "N": len(differences),
                "Missing": 0,
                "Mean Difference": round(mean_diff, 3),
                "Std Dev": round(std_diff, 3),
                "SEM": round(sem_diff, 3)
            }
        },
        "T-Test Results": {
            "t-Statistic": round(t_stat, 3),
            "Degrees of Freedom": df,
            "95% Confidence Interval for Difference of Means": [round(ci_low, 3), round(ci_high, 3)],
            "Two-Tailed P-Value": round(p_value, 5),
            "One-Tailed P-Value": round(one_tailed_p, 5),
            "Interpretation": f"The change that occurred with the treatment is {'statistically significant' if p_value < alpha else 'not statistically significant'} (P = {round(p_value, 5)})"
        },
        "Power Analysis": power_results
    }

    return result
