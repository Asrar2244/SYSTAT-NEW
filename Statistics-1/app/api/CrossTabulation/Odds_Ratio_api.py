import pandas as pd
import numpy as np
import datetime
from flask import Blueprint, request, jsonify
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.power import NormalIndPower
from ..helpers.logger import Logger
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
    ODDS_RATIO_LOG_FILE_PATH, 
    CONFIDENCE_INTERVAL_DEFAULT, 
    ALPHA_VALUE_DEFAULT
)

odds_ratio_api = Blueprint('odds_ratio_api', __name__)
logger = Logger(ODDS_RATIO_LOG_FILE_PATH)

@odds_ratio_api.route('/odds-ratio', methods=['POST'])
def perform_odds_ratio():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please send JSON data."}), 400

        data = request.get_json()
        input_format = data.get("input_data_format", "tabular").lower()

        alpha = data.get("alpha", ALPHA_VALUE_DEFAULT)
        yates_correction = data.get("yates_correction", False)
        confidence_level = data.get("confidence_level", CONFIDENCE_INTERVAL_DEFAULT)
        first_row_treatment = data.get("first_row_treatment", False)

        if confidence_level is not None:
            confidence_level = int(confidence_level)

        # Convert input into 2x2 DataFrame
        if input_format == "tabular":
            if not all(k in data for k in ["columns", "rows", "data"]):
                return jsonify({"error": "Missing 'columns', 'rows', or 'data' in 'tabular' format."}), 400

            df = pd.DataFrame(data["data"], columns=data["columns"], index=data["rows"])
            if df.shape != (2, 2):
                return jsonify({"error": "Tabular data must be a 2x2 table."}), 400

        elif input_format == "raw":
            raw_data = data.get("data", [])
            if not isinstance(raw_data, list):
                return jsonify({"error": "Raw input 'data' must be a list of records."}), 400

            df_raw = pd.DataFrame(raw_data)
            if not {"Group", "Outcome"}.issubset(df_raw.columns):
                return jsonify({"error": "Raw data must contain 'Group' and 'Outcome' columns."}), 400

            if df_raw["Outcome"].nunique() != 2:
                return jsonify({"error": "Outcome column must contain exactly 2 unique values."}), 400

            df = df_raw.pivot_table(index="Group", columns="Outcome", aggfunc="size", fill_value=0)
            df = df.reindex(columns=sorted(df.columns))
            if df.shape != (2, 2):
                return jsonify({"error": "Raw data must result in a 2x2 table after pivot."}), 400

        else:
            return jsonify({"error": "Invalid or missing 'input_data_format'. Use 'tabular' or 'raw'."}), 400

        row_labels = df.index.tolist()
        col_labels = df.columns.tolist()

        # Assign cell values depending on treatment row
        if first_row_treatment:
            a, b = df.iloc[0, 0], df.iloc[0, 1]
            c, d = df.iloc[1, 0], df.iloc[1, 1]
        else:
            c, d = df.iloc[0, 0], df.iloc[0, 1]
            a, b = df.iloc[1, 0], df.iloc[1, 1]

        odds_treatment = (a / b) if b != 0 else float('inf')
        odds_control = (c / d) if d != 0 else float('inf')
        odds_ratio = odds_treatment / odds_control if odds_control != 0 else float('inf')

        # Confidence Interval
        confidence_interval = None
        if confidence_level is not None:
            try:
                ci = float(confidence_level) / 100.0
                if not (0.01 <= ci <= 0.99):
                    return jsonify({"error": "Confidence level must be between 1 and 99."}), 400
                se_log_or = np.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
                z = norm.ppf(1 - (1 - ci) / 2)
                ci_lower = np.exp(np.log(odds_ratio) - z * se_log_or)
                ci_upper = np.exp(np.log(odds_ratio) + z * se_log_or)
                confidence_interval = [round(ci_lower, 3), round(ci_upper, 3)]
            except Exception as e:
                return jsonify({"error": f"Confidence interval calculation error: {str(e)}"}), 400

        # Chi-Square Test
        chi2_stat, p_value, _, _ = chi2_contingency(df.values, correction=yates_correction)

        # Power Analysis
        power_result = "N/A"
        if alpha is not None:
            try:
                effect_size = np.sqrt(chi2_stat / df.values.sum())
                power_analysis = NormalIndPower().power(effect_size=effect_size, nobs1=a + b, alpha=alpha)
                power_result = {round(alpha, 3): round(power_analysis, 3)}
            except Exception as e:
                logger.warning(f"Power calculation failed: {str(e)}")

        # Build Result
        conclusion = (
            "The treatment group significantly affects the outcome."
            if p_value <= 0.05 else
            "The treatment group does not significantly affect the outcome."
        )

        result = {
            "Timestamp": datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"),
            "Contingency Table": {
                col_labels[0]: {row_labels[0]: int(a), row_labels[1]: int(c)},
                col_labels[1]: {row_labels[0]: int(b), row_labels[1]: int(d)}
            },
            "Odds Ratio": round(odds_ratio, 3),
            "Chi-square Statistic": round(chi2_stat, 3),
            "P-Value": round(p_value, 5),
            "Conclusion": conclusion,
            "Yates Correction Used": yates_correction,
            "First Row as Treatment": first_row_treatment
        }

        if confidence_interval is not None:
            result[f"{confidence_level}% Confidence Interval"] = confidence_interval

        if alpha is not None:
            result["Power of Test"] = power_result

        logger.info(f"Odds Ratio Test result: {result}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(LOG_UNEXPECTED_ERROR.format(str(e)))
        return jsonify({"error": UNEXPECTED_ERROR_MSG}), 500
