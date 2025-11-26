import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from scipy.stats import shapiro, kstest, norm
from typing import Optional

from app.api.helpers.constant import (
    VALUE_ERROR_MSG,
    KEY_ERROR_MSG,
    TYPE_ERROR_MSG,
    UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR,
    LOG_KEY_ERROR,
    LOG_TYPE_ERROR,
    LOG_UNEXPECTED_ERROR,
    NORMALITY_TEST_LOG_FILE_PATH,
    P_VALUE_REJECT_DEFAULT
)
from app.api.helpers.logger import Logger

# Define the Flask Blueprint
normality_test_api = Blueprint('pca_api', __name__)

@normality_test_api.route("/normality-test", methods=["POST"])
def normality_test():
    logger = Logger(NORMALITY_TEST_LOG_FILE_PATH)

    try:
        # Read parameters from form or JSON, default values
        shaprio_walk = None
        kolmogorov_smirnov = None
        P_value_reject = None

        if 'file' in request.files:
            # File upload mode
            file = request.files['file']
            filename = file.filename
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                raise ValueError("Only .csv, .xls, or .xlsx files are supported.")

            # Read params from form data
            shaprio_walk = request.form.get('shaprio_walk', 'true').lower() == 'true'
            kolmogorov_smirnov = request.form.get('kolmo_with_correction', 'false').lower() == 'true'
            P_value_reject = float(request.form.get('P_value_reject', P_VALUE_REJECT_DEFAULT))

            numeric_cols = df.select_dtypes(include='number')
            if numeric_cols.empty:
                raise ValueError("No numeric columns found in the uploaded file.")

            results = {}
            for col in numeric_cols.columns:
                sample = numeric_cols[col].dropna().values
                if len(sample) < 3:
                    results[col] = {"error": "At least 3 values required."}
                    continue

                col_result = {}

                if shaprio_walk:
                    stat, p = shapiro(sample)
                    col_result["Shapiro-Wilk"] = {
                        "W-Statistic": round(stat, 4),
                        "P-Value": round(p, 4),
                        "Conclusion": "Passed" if p > P_value_reject else "Failed"
                    }

                if kolmogorov_smirnov:
                    mu = np.mean(sample)
                    sigma = np.std(sample, ddof=1)
                    stat, p = kstest(sample, 'norm', args=(mu, sigma))
                    col_result["Kolmogorov-Smirnov"] = {
                        "K-S Statistic": round(stat, 4),
                        "P-Value": round(p, 4),
                        "Conclusion": "Passed" if p > P_value_reject else "Failed"
                    }

                results[col] = col_result

        else:
            # No file: expect JSON body with multiple columns of data
            body = request.get_json()

            data = body.get("data")
            if not data or not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with column names and lists of numeric values.")

            shaprio_walk = body.get("shaprio_walk", True)
            kolmogorov_smirnov = body.get("kolmo_with_correction", False)
            P_value_reject = float(body.get("P_value_reject", P_VALUE_REJECT_DEFAULT))

            if not shaprio_walk and not kolmogorov_smirnov:
                raise KeyError("At least one of 'shaprio_walk' or 'kolmo_with_correction' must be True.")

            results = {}

            for col_name, sample_data in data.items():
                if not isinstance(sample_data, list) or len(sample_data) < 3:
                    results[col_name] = {"error": "At least 3 numeric values required."}
                    continue

                if not all(isinstance(x, (int, float)) for x in sample_data):
                    results[col_name] = {"error": "All sample values must be numeric."}
                    continue

                sample = np.array(sample_data)
                col_result = {}

                if shaprio_walk:
                    stat, p = shapiro(sample)
                    col_result["Shapiro-Wilk"] = {
                        "W-Statistic": round(stat, 4),
                        "P-Value": round(p, 4),
                        "Conclusion": "Passed" if p > P_value_reject else "Failed"
                    }

                if kolmogorov_smirnov:
                    mu = np.mean(sample)
                    sigma = np.std(sample, ddof=1)
                    stat, p = kstest(sample, 'norm', args=(mu, sigma))
                    col_result["Kolmogorov-Smirnov"] = {
                        "K-S Statistic": round(stat, 4),
                        "P-Value": round(p, 4),
                        "Conclusion": "Passed" if p > P_value_reject else "Failed"
                    }

                results[col_name] = col_result

        return jsonify({
            "success": True,
            "results": results
        })

    except ValueError as e:
        logger.log_exception(LOG_VALUE_ERROR, str(e))
        return jsonify({"success": False, "error_message": VALUE_ERROR_MSG}), 400
    except KeyError as e:
        logger.log_exception(LOG_KEY_ERROR, str(e))
        return jsonify({"success": False, "error_message": KEY_ERROR_MSG}), 400
    except TypeError as e:
        logger.log_exception(LOG_TYPE_ERROR, str(e))
        return jsonify({"success": False, "error_message": TYPE_ERROR_MSG}), 400
    except Exception as e:
        logger.log_exception(LOG_UNEXPECTED_ERROR, str(e))
        return jsonify({"success": False, "error_message": UNEXPECTED_ERROR_MSG}), 500
