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
    RELATIVE_RISK_LOG_FILE_PATH, 
    CONFIDENCE_INTERVAL_DEFAULT, 
    ALPHA_VALUE_DEFAULT
)

# Initialize Blueprint and Logger
relative_risk_api = Blueprint('relative_risk_api', __name__)
logger = Logger(RELATIVE_RISK_LOG_FILE_PATH)

def calculate_relative_risk(data, db=False, alpha=None, yates_correction=None, ci_level=None, first_row_treatment=None):
    """
    Computes Relative Risk from either wide or long format.
    """
    try:
        if db:  
            # **DB=True → Wide Format (2x2 Contingency Table)**
            df = pd.DataFrame(data['data'], columns=data['columns'], index=data['rows'])
            if df.shape != (2, 2):
                return {"error": "Invalid input: Wide format must be a 2x2 table."}
        else:
            # **DB=False → Long Format (Raw Data)**
            raw_df = pd.DataFrame(data['data'])
            if not {'Group', 'Outcome'}.issubset(raw_df.columns):
                return {"error": "Invalid input: Long format requires 'Group' and 'Outcome' columns."}
            
            # Ensure Outcome has only 2 categories
            outcome_counts = raw_df['Outcome'].nunique()
            if outcome_counts != 2:
                return {"error": "Invalid Outcome: Must have exactly two unique values (e.g., Yes/No)."}

            # Convert Long Format to Wide Format
            df = raw_df.pivot_table(index='Group', columns='Outcome', aggfunc='size', fill_value=0)
            df = df.reindex(columns=sorted(df.columns))  # Ensure Yes/No order

            if df.shape != (2, 2):
                return {"error": "Invalid conversion: Groups and Outcomes must form a 2x2 table."}
            
         # Assign values dynamically
        row_labels = df.index.tolist()
        col_labels = df.columns.tolist()

        # Assign values dynamically
        if first_row_treatment:
            a, b = df.iloc[0, 0], df.iloc[0, 1]
            c, d = df.iloc[1, 0], df.iloc[1, 1]
        else:
            c, d = df.iloc[0, 0], df.iloc[0, 1]
            a, b = df.iloc[1, 0], df.iloc[1, 1]

        # Compute Relative Risk
        p_treatment = a / (a + b) if (a + b) != 0 else 0
        p_control = c / (c + d) if (c + d) != 0 else 0
        relative_risk = p_treatment / p_control if p_control != 0 else float('inf')

        # Compute Confidence Intervals (if provided)
        confidence_interval = None
        if ci_level:
            
            try:
                ci_level = float(ci_level) / 100.0  # Convert to decimal
                if not (0.01 <= ci_level <= 0.99):
                  return {"error": "Invalid confidence level: Must be between 1% and 99%."}
                

                se_log_rr = np.sqrt((1/a) + (1/b) + (1/c) + (1/d))
                z_score = norm.ppf(1 - (1 - ci_level) / 2)
                ci_lower = np.exp(np.log(relative_risk) - z_score * se_log_rr)
                ci_upper = np.exp(np.log(relative_risk) + z_score * se_log_rr)
                confidence_interval = [round(ci_lower, 3), round(ci_upper, 3)]

            except ValueError:
                   return {"error": "Confidence level must be a number."}
   

        # Chi-square test
        chi2_stat, p_value, _, _ = chi2_contingency(df.values, correction=(yates_correction if yates_correction is not None else False))

        # Power Calculation (if alpha provided)
        power_analysis = None
        if alpha:
            effect_size = np.sqrt(chi2_stat / np.sum(df.values))
            power_analysis = NormalIndPower().power(effect_size=effect_size, nobs1=a + b, alpha=alpha)
            power_result = {round(alpha, 3): round(power_analysis, 3)}

        # Conclusion
        conclusion = "The likelihood of the outcome is greater in the treatment group." if p_value < 0.05 else "No significant difference observed."

        # Response Object
        response = {
            "Timestamp": datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"),
            "Contingency Table": {
                col_labels[0]: {row_labels[0]: int(a), row_labels[1]: int(c)},
                col_labels[1]: {row_labels[0]: int(b), row_labels[1]: int(d)}
            },
            "Relative Risk": round(relative_risk, 3),
            "Chi-square Statistic": round(chi2_stat, 3),
            "P-Value": round(p_value, 3),
            "Conclusion": conclusion
        }

        # Add Optional Fields Only If Provided
        if ci_level is not None and confidence_interval is not None:
            response[f"{int(ci_level * 100)}% Confidence Interval"] = confidence_interval
        if alpha is not None:
            response["Power of Test"] = power_result if power_analysis is not None else "N/A"
        if yates_correction is not None:
            response["Yates Correction Used"] = yates_correction
        if first_row_treatment is not None:
            response["First Row as Treatment"] = first_row_treatment

        return response
    except Exception as e:
        logger.error(f"Error in Relative Risk Calculation: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

@relative_risk_api.route('/relative-risk', methods=['POST'])
def perform_relative_risk():
    """
    API Endpoint for Relative Risk Calculation.
    """
    try:
        data = request.get_json()
        db = data.get("DB", False)  # Default to False (Long Format)
        alpha = data.get("alpha", ALPHA_VALUE_DEFAULT)
        yates_correction = data.get("yates_correction", False)
        confidence_level = data.get("confidence_level",  CONFIDENCE_INTERVAL_DEFAULT)
        first_row_treatment = data.get("first_row_treatment", False)

        if confidence_level is not None:
            confidence_level = int(confidence_level)

        result = calculate_relative_risk(data, db, alpha, yates_correction, confidence_level, first_row_treatment)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500