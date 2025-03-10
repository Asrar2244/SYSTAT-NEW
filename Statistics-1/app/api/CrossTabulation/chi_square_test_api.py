import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from scipy.stats import chi2_contingency
from .fisher_exact_test_api import fisher_exact_test_logic
from statsmodels.stats.power import GofChisquarePower
from ..helpers.constant import (
    VALUE_ERROR_MSG,
    KEY_ERROR_MSG, 
    TYPE_ERROR_MSG, 
    INDEX_ERROR_MSG, 
    UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR, 
    LOG_KEY_ERROR, 
    LOG_TYPE_ERROR, 
    LOG_INDEX_ERROR, 
    LOG_UNEXPECTED_ERROR,
    CHI_SQUARE_LOG_FILE_PATH,
    ALPHA_VALUE_DEFAULT
)
from ..helpers.logger import Logger

chi_square_test_api = Blueprint('chi_square_test_api', __name__)

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

@chi_square_test_api.route('/chi-square-test', methods=['POST'])
def perform_chi_square_test():
    """
    Perform a Chi-Square Test for Independence on input data and return statistical results.
    """
    logger = Logger(CHI_SQUARE_LOG_FILE_PATH)
    
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid input: Expected JSON format."}), 400

        data = request.get_json()

        use_fishers_test = data.get("use_fishers_test", False)
        alpha = data.get("alpha", ALPHA_VALUE_DEFAULT)
        yates_correction = data.get("yates_correction", False)
        tables = data.get("tables", {})
        test_statistics = data.get("test_statistics", {})
        other_statistics = data.get("other_statistics", {})
        

        # Check if data follows Wide-Format (Rows, Columns, Data)
        if "columns" in data and "rows" in data and "data" in data:
            columns = data["columns"]
            rows = data["rows"]
            contingency_table = pd.DataFrame(data["data"], index=rows, columns=columns)

        # Check if data follows Long-Format ('Group', 'Category')
        elif "data" in data and isinstance(data["data"], list):
            raw_data = data["data"]
            df = pd.DataFrame(raw_data)

            if "Group" in df.columns and "Category" in df.columns:
                contingency_table = df.groupby(["Group", "Category"]).size().unstack(fill_value=0)
            else:
                return jsonify({"error": "Invalid format: Expected 'Group' and 'Category' columns."}), 400
        else:
            return jsonify({"error": "Invalid input format: Expected 'columns', 'rows', and 'data' or long-format data."}), 400

        if contingency_table.empty:
            return jsonify({"error": "Generated contingency table is empty. Check input data."}), 400

        logger.info(f"Processed data into contingency table of shape {contingency_table.shape}")

        # Fisher's Exact Test check
        if use_fishers_test:
            if contingency_table.shape == (2, 2):
                # Perform Chi-Square Test first to get the expected values
                chi2_stat, p_value, dof, expected_counts = chi2_contingency(contingency_table, correction=yates_correction)
        
                # Flatten the expected counts and check the percentage of expected values less than 5
                expected_values_flat = expected_counts.flatten()
                percentage_less_than_5 = (expected_values_flat < 5).sum() / len(expected_values_flat) * 100
        
                # If more than 20% of the expected values are less than 5, use Fisher's Exact Test
                if percentage_less_than_5 > 20:
                    logger.info("More than 20% of expected values are less than 5 in a 2x2 table, using Fisher's Exact Test.")
                    fisher_result = fisher_exact_test_logic(data)
                    return jsonify(fisher_result), 200
        
                # Otherwise, proceed with Chi-Square test (if explicitly requested)
                logger.info("Fisher's Exact Test not needed, using Chi-Square test for 2x2 table.")
                fisher_result = fisher_exact_test_logic(data)  # This will run Fisher's test even if it's not explicitly requested, if above condition is not met.
                return jsonify(fisher_result), 200
            else:
                return jsonify({"error": "Fisher's Exact Test can only be applied to 2x2 tables."}), 400
            
        # Perform Chi-Square Test
        chi2_stat, p_value, dof, expected_counts = chi2_contingency(contingency_table, correction=yates_correction)
        expected_df = pd.DataFrame(expected_counts, index=contingency_table.index, columns=contingency_table.columns)
        total_count = contingency_table.sum().sum()
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)

        result = {
            "Chi-Squared Test for Independence": {
                "Pearson Chi-Square": round(chi2_stat, 3),
                "Degrees of Freedom": dof,
                "P-Value": round(p_value, 5),
                "Conclusion": "Reject Null Hypothesis" if p_value < alpha else "Fail to Reject Null Hypothesis"
            }
        }

        # Add Counts and Percentages in the required format
        if tables.get("counts", False) or tables.get("percentages", False):
            formatted_table = []

            # Construct table with the required format
            for index, row in contingency_table.iterrows():
                formatted_row = {"Row": index}

                for col in contingency_table.columns:
                    # Check if both counts and percentages are needed
                    if tables.get("counts", False) and tables.get("percentages", False):
                       formatted_row[col] = {
                         "Count": round(row[col], 3),
                         "Expected Count": round(expected_df.at[index, col], 3),
                         "% Row Total": round((row[col] / row_totals[index]) * 100, 3),
                         "% Column Total": round((row[col] / col_totals[col]) * 100, 3),
                         "% Total": round((row[col] / total_count) * 100, 3)
                      }
                    # Check if only counts are needed
                    elif tables.get("counts", False):
                        formatted_row[col] = {
                           "Count": round(row[col], 3),
                           "Expected Count": round(expected_df.at[index, col], 3)
                        }
                    # Check if only percentages are needed
                    elif tables.get("percentages", False):
                        formatted_row[col] = {
                           "% Row Total": round((row[col] / row_totals[index]) * 100, 3),
                           "% Column Total": round((row[col] / col_totals[col]) * 100, 3),
                           "% Total": round((row[col] / total_count) * 100, 3)
                        }

                # Add "Total" for the row (sum of the row across columns)
                formatted_row["Total"] = {}

                if tables.get("counts", False) and tables.get("percentages", False):
                   formatted_row["Total"] = {
                      "Count": round(row.sum(), 3),
                      "Expected Count": round(row.sum(), 3),
                      "% Row Total": 100.000,
                      "% Column Total": round((row.sum() / total_count) * 100, 3),
                      "% Total": round((row.sum() / total_count) * 100, 3)
                    }
                elif tables.get("counts", False):
                   formatted_row["Total"] = {
                      "Count": round(row.sum(), 3),
                      "Expected Count": round(row.sum(), 3)
                    }
                elif tables.get("percentages", False):
                   formatted_row["Total"] = {
                      "% Row Total": 100.000,
                      "% Column Total": round((row.sum() / total_count) * 100, 3),
                      "% Total": round((row.sum() / total_count) * 100, 3)
                    }

                formatted_table.append(formatted_row)

            # Add grand totals
            total_row = {"Row": "Total"}
            for col in contingency_table.columns:
                if tables.get("counts", False) and tables.get("percentages", False):
                   total_row[col] = {
                       "Count": round(contingency_table[col].sum(), 3),
                       "Expected Count": round(contingency_table[col].sum(), 3),
                       "% Row Total": round((contingency_table[col].sum() / total_count) * 100, 3),
                       "% Column Total": 100.000,
                       "% Total": round((contingency_table[col].sum() / total_count) * 100, 3)
                    }
                elif tables.get("counts", False):
                   total_row[col] = {
                     "Count": round(contingency_table[col].sum(), 3),
                     "Expected Count": round(contingency_table[col].sum(), 3)
                    }
                elif tables.get("percentages", False):
                   total_row[col] = {
                     "% Row Total": round((contingency_table[col].sum() / total_count) * 100, 3),
                     "% Column Total": 100.000,
                     "% Total": round((contingency_table[col].sum() / total_count) * 100, 3)
                    }

            # Add "Total" for the total row (sum of all columns)
            total_row["Total"] = {}

            if tables.get("counts", False) and tables.get("percentages", False):
              total_row["Total"] = {
                   "Count": round(contingency_table.sum().sum(), 3),
                   "Expected Count": round(contingency_table.sum().sum(), 3),
                   "% Row Total": 100.000,
                   "% Column Total": 100.000,
                   "% Total": 100.000
                }
            elif tables.get("counts", False):
              total_row["Total"] = {
                 "Count": round(contingency_table.sum().sum(), 3),
                 "Expected Count": round(contingency_table.sum().sum(), 3)
                }
            elif tables.get("percentages", False):
              total_row["Total"] = {
                 "% Row Total": 100.000,
                 "% Column Total": 100.000,
                 "% Total": 100.000
                }

            formatted_table.append(total_row)
            result["Formatted Contingency Table"] = formatted_table


        
        # Add Residuals if required
        if tables.get("residuals", False):
            residuals = contingency_table - expected_df
            std_residuals = residuals / expected_df.pow(0.5)
            result["Residuals"] = residuals.round(3).to_dict()
            result["Standardized Residuals"] = std_residuals.round(3).to_dict()

        # Add Log-Likelihood if requested
        if test_statistics.get("log_likelihood", False):
            log_likelihood = chi2_contingency(contingency_table, lambda_="log-likelihood")[0]
            result["Chi-Squared Test for Independence"]["Log-Likelihood Ratio"] = round(log_likelihood, 3)

        # Add Phi Coefficient if requested
        if other_statistics.get("phi", False):
            phi_value = (chi2_stat / total_count) ** 0.5
            result.setdefault("Measures of Association", {})["Phi Coefficient"] = round(phi_value, 3)

        # Add Cramér's V if requested
        if other_statistics.get("cramers_v", False):
            k = min(contingency_table.shape) - 1
            cramer_v = ((chi2_stat / (total_count * k)) ** 0.5) if k > 0 else 0
            result.setdefault("Measures of Association", {})["Cramer's V"] = round(cramer_v, 3)


        # Calculate Power of the performed test only if alpha is provided
        if alpha is not None:
            power_analysis = GofChisquarePower()
            effect_size = cramer_v
            power = power_analysis.solve_power(effect_size=effect_size, nobs=total_count, alpha=alpha)
            result["Chi-Squared Test for Independence"]["Power of Test"] = {round(alpha, 3): round(power, 3)}

        logger.info(f"Chi-Square Test result: {result}")
        return jsonify(convert_numpy_types(result)), 200

    except ValueError as ve:
        logger.error(LOG_VALUE_ERROR.format(str(ve)))
        return jsonify({"error": VALUE_ERROR_MSG.format(str(ve))}), 400
    except KeyError as ke:
        logger.error(LOG_KEY_ERROR.format(str(ke)))
        return jsonify({"error": KEY_ERROR_MSG.format(str(ke))}), 400
    except TypeError as te:
        logger.error(LOG_TYPE_ERROR.format(str(te)))
        return jsonify({"error": TYPE_ERROR_MSG.format(str(te))}), 400
    except IndexError as ie:
        logger.error(LOG_INDEX_ERROR.format(str(ie)))
        return jsonify({"error": INDEX_ERROR_MSG}), 400
    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500
