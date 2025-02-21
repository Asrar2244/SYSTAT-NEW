import json
from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
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
    TWO_SAMPLE_T_TEST_LOG_FILE_PATH,
    CONFIDENCE_INTERVAL_DEFAULT,
    P_VALUE_REJECT_DEFAULT,
    ALPHA_VALUE_DEFAULT,
    EQUAL_VARIANCE_DEFAULT

)
from app.api.helpers.logger import Logger

# Initialize Flask Blueprint
two_sample_t_test_api = Blueprint('two_sample_t_test', __name__)

@two_sample_t_test_api.route('/two-sample-t-test', methods=['POST'])
def perform_two_sample_t_test():
    logger = Logger(TWO_SAMPLE_T_TEST_LOG_FILE_PATH)
    try:
        data = request.get_json()
        p_value = data.get("p_value", "two-tailed")
        normality_reject_threshold = data.get("normality_p_value_to_reject", P_VALUE_REJECT_DEFAULT)
        equal_variance_reject_threshold = data.get("equal_variance_p_value_to_reject", EQUAL_VARIANCE_DEFAULT)
        confidence_level = data.get("confidence_level", CONFIDENCE_INTERVAL_DEFAULT)
        alpha = float(data.get("alpha_value", ALPHA_VALUE_DEFAULT))
        # NOOR - true by default
        shapiro_wilk = data.get("shapiro_wilk", True)
        kolmo_with_correction = data.get("kolmo_with_correction", False)
        # NOOR students_ttest and welchs - both are True by default
        students_ttest = data.get("students_ttest", True)
        welchs_ttest = data.get("welchs_ttest", True)
        db_flag = data.get("DB", False)

        if db_flag:
           if "Group1" not in data or "Group2" not in data:
              raise ValueError("DB=True input must contain 'Group1' and 'Group2' keys.")
    
           group1, group2 = data["Group1"], data["Group2"]

           if not isinstance(group1, list) or not isinstance(group2, list):
              raise ValueError("'Group1' and 'Group2' must be lists.")

           if not all(isinstance(x, (int, float)) for x in group1 + group2):
              raise ValueError("'Group1' and 'Group2' must contain only numeric values.")

           if len(group1) < 2 or len(group2) < 2:
              raise ValueError("Each group must have at least two values.")
    
           group1_key, group2_key = "Group1", "Group2"

        elif "indexed" in data:
            indexed_df = data["indexed"]
            if "Group" not in indexed_df or "Data" not in indexed_df:
                raise ValueError("Indexed input must contain 'Group' and 'Data' keys.")
            df = pd.DataFrame(indexed_df)
            if not np.issubdtype(df["Data"].dtype, np.number):
                raise ValueError("Column 'Data' must contain only numeric values.")
            grouped = df.groupby("Group")["Data"].apply(list).to_dict()
            if len(grouped) != 2:
                raise ValueError("There must be exactly two groups for a two-sample t-test.")
            group1_key, group2_key = list(grouped.keys())
            group1, group2 = grouped[group1_key], grouped[group2_key]

        # Asrar: Updated input handling for summary statistics:
        # - Introduced structured keys: 'mean_size_deviation' and 'mean_size_error' to distinguish input types.
        # - Data is now passed in a structured list format, ensuring clarity and reducing ambiguity.
        elif "values" in data:
            values = data["values"]
            mean_size_deviation = values.get("mean_size_deviation", False)
            mean_size_error = values.get("mean_size_error", False)
    
            if mean_size_deviation:
                if "mean_size_dev_values" not in values:
                   raise ValueError("Missing 'mean_size_dev_values' key for mean, size, and deviation input.")
        
                groups = values["mean_size_dev_values"]
                if len(groups) != 2:
                    raise ValueError("'mean_size_dev_values' must contain exactly two group dictionaries.")
        
                mean1, size1, std1 = groups[0]["mean"], groups[0]["size"], groups[0]["devi"]
                mean2, size2, std2 = groups[1]["mean2"], groups[1]["size2"], groups[1]["dev2"]
    
            elif mean_size_error:
                if "mean_size_error_values" not in values:
                   raise ValueError("Missing 'mean_size_error_values' key for mean, size, and standard error input.")
        
                groups = values["mean_size_error_values"]
                if len(groups) != 2:
                    raise ValueError("'mean_size_error_values' must contain exactly two group dictionaries.")
        
                mean1, size1, se1 = groups[0]["mean"], groups[0]["size"], groups[0]["err_val"]
                mean2, size2, se2 = groups[1]["mean2"], groups[1]["size2"], groups[1]["err_val"]
                std1, std2 = se1 * np.sqrt(size1), se2 * np.sqrt(size2)
    
            else:
                raise ValueError("Either 'mean_size_deviation' or 'mean_size_error' must be True.")
    
            group1 = np.random.normal(mean1, std1, size1)
            group2 = np.random.normal(mean2, std2, size2)
            group1_key, group2_key = "Group1", "Group2"


            
        else:
            raise ValueError("Invalid input format.")

        output_data = calculate_and_format_two_sample_t_test(
            group1, group2, group1_key, group2_key, alpha, confidence_level,
            shapiro_wilk, kolmo_with_correction, students_ttest, welchs_ttest
        )
        return jsonify(output_data), 200

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
        return jsonify({"error": ZERO_DIVISION_ERROR_MSG}), 400
    except IndexError as ie:
        logger.error(LOG_INDEX_ERROR.format(str(ie)))
        return jsonify({"error": INDEX_ERROR_MSG}), 400
    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500


from statsmodels.stats.diagnostic import lilliefors

def perform_normality_and_variance_tests(group1, group2, reject_threshold, shapiro_wilk=False, kolmo_with_correction=False):
    test_results = {}

    # Shapiro-Wilk Test (Run Separately for Each Group)
    if shapiro_wilk:
         shapiro_p1 = stats.shapiro(group1).pvalue
         shapiro_p2 = stats.shapiro(group2).pvalue
         shapiro_final_p = min(shapiro_p1, shapiro_p2)
         test_results["Normality Test (Shapiro-Wilk)"] = {
            "P-Value": round(shapiro_final_p, 3),
            "Result": "Passed" if shapiro_final_p > reject_threshold else "Failed"
        }

    # Kolmogorov-Smirnov (Lilliefors) Test (Run Separately for Each Group)
    if kolmo_with_correction:
          kolmo_p1 = lilliefors(group1)[1]
          kolmo_p2 = lilliefors(group2)[1]
          kolmo_final_p = min(kolmo_p1, kolmo_p2)
          test_results["Kolmogorov-Smirnov (Lilliefors)"] = {
             "P-Value": round(kolmo_final_p, 3),
             "Result": "Passed" if kolmo_final_p > reject_threshold else "Failed"
        }

    # Levene’s Test (Equal Variance)
    levene_p_value = stats.levene(group1, group2, center='mean').pvalue 
    test_results["Equal Variance Test (Levene's Test)"] = {
        "P-Value": round(levene_p_value, 3),
        "Result": "Passed" if levene_p_value > reject_threshold else "Failed"
    }

    # Brown-Forsythe Test (Centered at Median)
    brown_forsythe_p_value = stats.levene(group1, group2, center='trimmed').pvalue
    test_results["Equal Variance Test (Brown-Forsythe)"] = {
        "P-Value": round(brown_forsythe_p_value, 3),
        "Result": "Passed" if brown_forsythe_p_value > reject_threshold else "Failed"
    }

    return test_results

def perform_t_tests(group1, group2):
    t_stat_pooled, p_value_pooled = stats.ttest_ind(group1, group2, equal_var=True)
    df_pooled = len(group1) + len(group2) - 2
    
    t_stat_separate, p_value_separate = stats.ttest_ind(group1, group2, equal_var=False)
    df_separate = ((np.var(group1, ddof=1) / len(group1)) + (np.var(group2, ddof=1) / len(group2))) ** 2 / (
        (np.var(group1, ddof=1) / len(group1)) ** 2 / (len(group1) - 1) + (np.var(group2, ddof=1) / len(group2)) ** 2 / (len(group2) - 1)
    )
    
    return t_stat_pooled, p_value_pooled, df_pooled, t_stat_separate, p_value_separate, df_separate

def compute_confidence_interval(mean_diff, std1, std2, n1, n2, confidence, df):
    se_diff = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
    t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)
    return mean_diff - t_critical * se_diff, mean_diff + t_critical * se_diff

def calculate_power(n1, n2, std1, std2, alpha, mean_diff):
    effect_size = mean_diff / np.sqrt((std1 ** 2 + std2 ** 2) / 2)
    power_analysis = TTestIndPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs1=n1, ratio=n2/n1, alpha=alpha)
    return round(power, 3)

def calculate_and_format_two_sample_t_test(group1, group2, group1_name, group2_name, alpha, confidence, shapiro_wilk, kolmo_with_correction, Students_ttest, welchs_ttest):
    test_results = perform_normality_and_variance_tests(group1, group2, alpha, shapiro_wilk, kolmo_with_correction)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    mean_difference = mean1 - mean2
    
    t_stat_pooled, p_value_pooled, df_pooled, t_stat_separate, p_value_separate, df_separate = perform_t_tests(group1, group2)
    lower_bound, upper_bound = compute_confidence_interval(mean_difference, std1, std2, len(group1), len(group2), confidence, df_separate)
    
    power_two_tailed = calculate_power(len(group1), len(group2), std1, std2, alpha, mean_difference)
    power_one_tailed = calculate_power(len(group1), len(group2), std1, std2, alpha / 2, mean_difference)

    t_test_results = {}

    if Students_ttest:
        t_test_results["Equal Variances Assumed (Student's t-test)"] = {
            "t": round(t_stat_pooled, 3), "df": df_pooled,
            "95% CI": [round(lower_bound, 3), round(upper_bound, 3)],
            "Two-tailed P-value": round(p_value_pooled, 5), "One-tailed P-value": round(p_value_pooled / 2, 5),
            "Power (Two-tailed)": power_two_tailed, "Power (One-tailed)": power_one_tailed
        }

    if welchs_ttest:
        t_test_results["Equal Variances Not Assumed (Welch's t-test)"] = {
            "t": round(t_stat_separate, 3), "df": round(df_separate, 3),
            "95% CI": [round(lower_bound, 3), round(upper_bound, 3)],
            "Two-tailed P-value": round(p_value_separate, 5), "One-tailed P-value": round(p_value_separate / 2, 5),
            "Power (Two-tailed)": power_two_tailed, "Power (One-tailed)": power_one_tailed
        }

    
    return {
        "Normality Test": test_results,
        "Sample Statistics": [
            {"Group": group1_name, "Mean": round(mean1, 3), "Std Dev": round(std1, 3)},
            {"Group": group2_name, "Mean": round(mean2, 3), "Std Dev": round(std2, 3)}
        ],
        "Difference of Means": round(mean_difference, 3),
        "T-Test Results": t_test_results
        }