import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from scipy.stats import pearsonr, chi2, norm
from scipy.spatial.distance import cdist
from typing import Optional
from datetime import datetime

from app.api.helpers.constant import (
    VALUE_ERROR_MSG,
    KEY_ERROR_MSG,
    TYPE_ERROR_MSG,
    UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR,
    LOG_KEY_ERROR,
    LOG_TYPE_ERROR,
    LOG_UNEXPECTED_ERROR,
    PEARSON_TEST_LOG_FILE_PATH
)
from app.api.helpers.logger import Logger

pearson_api = Blueprint('pearson_api', __name__)


def mardia_test(x: np.ndarray, y: np.ndarray, alpha=0.05):
    data = np.column_stack((x, y))
    n, p = data.shape

    mean = np.mean(data, axis=0)
    centered = data - mean

    cov = np.cov(centered, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    D_squared = np.array([row @ cov_inv @ row.T for row in centered])

    skew = np.sum([(row1 @ cov_inv @ row2.T) ** 3 for row1 in centered for row2 in centered]) / (n ** 2)
    skew_stat = n * skew / 6
    skew_pval = 1 - chi2.cdf(skew_stat, df=p * (p + 1) * (p + 2) // 6)

    kurtosis = np.sum(D_squared ** 2) / n
    expected_kurtosis = p * (p + 2)
    z_kurtosis = (kurtosis - expected_kurtosis) / np.sqrt(8 * p * (p + 2) / n)
    kurtosis_pval = 2 * (1 - norm.cdf(np.abs(z_kurtosis)))

    return {
        "Skewness": round(skew, 4),
        "Skewness Statistic": round(skew_stat, 4),
        "Skewness p-value": round(skew_pval, 4),
        "Skewness Result": "Passed" if skew_pval > alpha else "Failed",
        "Kurtosis": round(kurtosis, 4),
        "Kurtosis Statistic": round(z_kurtosis, 4),
        "Kurtosis p-value": round(kurtosis_pval, 4),
        "Kurtosis Result": "Passed" if kurtosis_pval > alpha else "Failed"
    }


def henze_zirkler_test(x: np.ndarray, y: np.ndarray, alpha=0.05):
    X = np.column_stack((x, y))
    n, p = X.shape
    mean = np.mean(X, axis=0)
    centered = X - mean
    cov = np.cov(centered, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    # Estimate beta (parameter)
    beta = 1 / np.sqrt(2)
    
    # Mahalanobis distances between all pairs
    dist_matrix = cdist(centered, centered, metric='mahalanobis', VI=cov_inv)
    dist_sq = dist_matrix ** 2

    term1 = np.exp(-beta**2 / 2 * dist_sq).sum() / (n ** 2)
    term2 = 2 * np.mean(np.exp(-beta**2 / 2 * np.sum(centered @ cov_inv * centered, axis=1)))
    hz_stat = n * (term1 - term2 + 1)

    # Asymptotic mean and variance
    mu_hz = 1 - (1 + 2 * beta**2) ** (-p / 2)
    sigma_hz_sq = 2 * (1 + 4 * beta**2) ** (-p / 2) - 2 * (1 + 2 * beta**2) ** (-p) + 1 - 2 * mu_hz

    z_hz = (hz_stat / n - mu_hz) / np.sqrt(sigma_hz_sq / n)
    p_val = 1 - norm.cdf(z_hz)

    return {
        "HZ Statistic": round(hz_stat, 4),
        "z-Value": round(z_hz, 4),
        "p-value": round(p_val, 4),
        "Result": "Passed" if p_val > alpha else "Failed"
    }


@pearson_api.route("/pearson-correlation", methods=["POST"])
def pearson_correlation():
    logger = Logger(PEARSON_TEST_LOG_FILE_PATH)

    try:
        results = {}
        assumption_results = {}
        p_threshold = 0.05
        assumption_checking = False
        normality_stats = None
        result_display_format = "matrix"
        source_info = "Unknown"

        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            source_info = f"Data source: {filename}"
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Only .csv, .xls, or .xlsx files are supported.")

            col_x = request.form.get('column_x')
            col_y = request.form.get('column_y')
            if not col_x or not col_y:
                raise KeyError("Both 'column_x' and 'column_y' are required.")

            if col_x not in df.columns or col_y not in df.columns:
                raise ValueError(f"Columns '{col_x}' and/or '{col_y}' not found.")

            df = df[[col_x, col_y]].dropna()
            x = df[col_x].values
            y = df[col_y].values

            assumption_checking = request.form.get('assumption_checking', 'false').lower() == 'true'
            normality_stats = request.form.get('normality_statistics')
            p_threshold = float(request.form.get('p_value_threshold', 0.05))
            result_display_format = request.form.get('result_display_format', 'matrix').lower()

        else:
            body = request.get_json()
            x = body.get("x")
            y = body.get("y")
            source_info = "Data from JSON input"

            if not isinstance(x, list) or not isinstance(y, list):
                raise ValueError("Both 'x' and 'y' must be lists of numeric values.")
            if len(x) != len(y):
                raise ValueError("The lists 'x' and 'y' must have the same length.")
            if len(x) < 3:
                raise ValueError("At least 3 observations are required.")
            if not all(isinstance(i, (int, float)) for i in x + y):
                raise TypeError("All elements must be numeric.")

            x = np.array(x)
            y = np.array(y)

            assumption_checking = body.get('assumption_checking', {}).get('normality', False)
            normality_stats = body.get('assumption_checking', {}).get('normality_statistics')
            p_threshold = float(body.get('assumption_checking', {}).get('p_value_threshold', 0.05))
            result_display_format = body.get('result_display_format', 'matrix').lower()

        if len(x) != len(y):
            raise ValueError("Mismatched array lengths after NaN filtering.")

        if assumption_checking and normality_stats in ("henze-zirkler", "mardia"):
            if normality_stats == "henze-zirkler":
                try:
                    hz = henze_zirkler_test(x, y, alpha=p_threshold)
                    assumption_results["Bivariate Normality Test (Henze-Zirkler)"] = hz
                except Exception as e:
                    assumption_results["Henze-Zirkler Test"] = f"Error: {str(e)}"

            elif normality_stats == "mardia":
                try:
                    mardia = mardia_test(x, y, alpha=p_threshold)
                    assumption_results["Bivariate Normality Test (Mardia)"] = mardia
                except Exception as e:
                    assumption_results["Mardia's Test"] = f"Error: {str(e)}"

        corr, p_val = pearsonr(x, y)

        if result_display_format == "matrix":
            corr_matrix = np.array([[1.0, corr], [corr, 1.0]])
            pval_matrix = np.array([[0.0, p_val], [p_val, 0.0]])
            results = {
                "Correlation Matrix": corr_matrix.round(4).tolist(),
                "P-Value Matrix": pval_matrix.round(4).tolist(),
                "Conclusion": "Significant" if p_val < p_threshold else "Not Significant"
            }

        elif result_display_format == "table":
            date_now = datetime.now().strftime('%d %b %Y %H:%M:%S')
            interpretation = (
                f"Pearson Product Moment Correlation\t{date_now}\n\n"
                f"{source_info}\n\n"
            )

            if "Bivariate Normality Test (Mardia)" in assumption_results:
                res = assumption_results["Bivariate Normality Test (Mardia)"]
                interpretation += (
                    "Bivariate Normality Test (Mardia):\n\n"
                    f"Variable Pair\tSkewness\tStatistic\tP\tResult\n"
                    f"x x y\t\t{res['Skewness']}\t\t{res['Skewness Statistic']}\t\t{res['Skewness p-value']}\t{res['Skewness Result']}\n\n"
                    f"Variable Pair\tKurtosis\tStatistic\tP\tResult\n"
                    f"x x y\t\t{res['Kurtosis']}\t\t{res['Kurtosis Statistic']}\t\t{res['Kurtosis p-value']}\t{res['Kurtosis Result']}\n\n"
                )

            if "Bivariate Normality Test (Henze-Zirkler)" in assumption_results:
                res = assumption_results["Bivariate Normality Test (Henze-Zirkler)"]
                interpretation += (
                    "Bivariate Normality Test (Henze-Zirkler):\n\n"
                    f"HZ Statistic: {res['HZ Statistic']}, z-Value: {res['z-Value']}, P-value: {res['p-value']}, Result: {res['Result']}\n\n"
                )

            interpretation += (
                "Correlation Results:\n\n"
                f"Variable Pair\tCorrelation Coefficient\tP\tSample Size\n"
                f"x x y\t\t{round(corr, 4)}\t\t{round(p_val, 4)}\t{len(x)}\n\n"
            )

            results = {
                "Formatted Report": interpretation
            }

        else:
            results = {
                "Pearson Correlation Coefficient": round(corr, 4),
                "P-Value": round(p_val, 4),
                "Conclusion": "Significant" if p_val < p_threshold else "Not Significant"
            }

        return jsonify({
            "success": True,
            "assumption_checking_results": assumption_results,
            "results": results
        })

    except ValueError as e:
        logger.log_exception(LOG_VALUE_ERROR, str(e))
        return jsonify({"success": False, "error_message": str(e), "error_code": "VALUE_ERROR"}), 400
    except KeyError as e:
        logger.log_exception(LOG_KEY_ERROR, str(e))
        return jsonify({"success": False, "error_message": str(e), "error_code": "KEY_ERROR"}), 400
    except TypeError as e:
        logger.log_exception(LOG_TYPE_ERROR, str(e))
        return jsonify({"success": False, "error_message": str(e), "error_code": "TYPE_ERROR"}), 400
    except Exception as e:
        logger.log_exception(LOG_UNEXPECTED_ERROR, str(e))
        return jsonify({"success": False, "error_message": str(e), "error_code": "UNEXPECTED_ERROR"}), 500
