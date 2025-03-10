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

# Initialize Blueprint and Logger
odds_ratio_api = Blueprint('odds_ratio_api', __name__)
logger = Logger(ODDS_RATIO_LOG_FILE_PATH)

def calculate_odds_ratio(data, db=False, alpha=None, yates_correction=None, ci_level=None, first_row_treatment=None):
    """
    Computes Odds Ratio from either wide or long format.
    """
    try:
        if db:
            df = pd.DataFrame(data['data'], columns=data['columns'], index=data['rows'])
            if df.shape != (2, 2):
                return {"error": "Invalid input: Wide format must be a 2x2 table."}
        else:
            raw_df = pd.DataFrame(data['data'])
            if not {'Group', 'Outcome'}.issubset(raw_df.columns):
                return {"error": "Invalid input: Long format requires 'Group' and 'Outcome' columns."}
            
            outcome_counts = raw_df['Outcome'].nunique()
            if outcome_counts != 2:
                return {"error": "Invalid Outcome: Must have exactly two unique values (e.g., Yes/No)."}
            
            df = raw_df.pivot_table(index='Group', columns='Outcome', aggfunc='size', fill_value=0)
            df = df.reindex(columns=sorted(df.columns))
            
            if df.shape != (2, 2):
                return {"error": "Invalid conversion: Groups and Outcomes must form a 2x2 table."}
            
        row_labels = df.index.tolist()
        col_labels = df.columns.tolist()
        
        if first_row_treatment:
            a, b = df.iloc[0, 0], df.iloc[0, 1]
            c, d = df.iloc[1, 0], df.iloc[1, 1]
        else:
            c, d = df.iloc[0, 0], df.iloc[0, 1]
            a, b = df.iloc[1, 0], df.iloc[1, 1]
        
        odds_treatment = (a / b) if b != 0 else float('inf')
        odds_control = (c / d) if d != 0 else float('inf')
        odds_ratio = odds_treatment / odds_control if odds_control != 0 else float('inf')
        
        confidence_interval = None
        if ci_level:
            try:
                ci_level = float(ci_level) / 100.0
                if not (0.01 <= ci_level <= 0.99):
                    return {"error": "Invalid confidence level: Must be between 1% and 99%."}
                
                se_log_or = np.sqrt((1/a) + (1/b) + (1/c) + (1/d))
                z_score = norm.ppf(1 - (1 - ci_level) / 2)
                ci_lower = np.exp(np.log(odds_ratio) - z_score * se_log_or)
                ci_upper = np.exp(np.log(odds_ratio) + z_score * se_log_or)
                confidence_interval = [round(ci_lower, 3), round(ci_upper, 3)]
            except ValueError:
                return {"error": "Confidence level must be a number."}
        
        chi2_stat, p_value, _, _ = chi2_contingency(df.values, correction=(yates_correction if yates_correction is not None else False))
        
        power_analysis = None
        if alpha:
            effect_size = np.sqrt(chi2_stat / np.sum(df.values))
            power_analysis = NormalIndPower().power(effect_size=effect_size, nobs1=a + b, alpha=alpha)
            power_result = {round(alpha, 3): round(power_analysis, 3)}
        
        conclusion = "The treatment group does not significantly affect the outcome." if p_value > 0.05 else "The treatment group significantly affects the outcome."
        
        response = {
            "Timestamp": datetime.datetime.now().strftime("%d %B %Y %H:%M:%S"),
            "Contingency Table": {
                col_labels[0]: {row_labels[0]: int(a), row_labels[1]: int(c)},
                col_labels[1]: {row_labels[0]: int(b), row_labels[1]: int(d)}
            },
            "Odds Ratio": round(odds_ratio, 3),
            "Chi-square Statistic": round(chi2_stat, 3),
            "P-Value": round(p_value, 3),
            "Conclusion": conclusion
        }
        
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
        logger.error(f"Error in Odds Ratio Calculation: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

@odds_ratio_api.route('/odds-ratio', methods=['POST'])
def perform_odds_ratio():
    """
    API Endpoint for Odds Ratio Calculation.
    """
    try:
        data = request.get_json()
        db = data.get("DB", False)
        alpha = data.get("alpha", ALPHA_VALUE_DEFAULT)
        yates_correction = data.get("yates_correction", False)
        confidence_level = data.get("confidence_level", CONFIDENCE_INTERVAL_DEFAULT)
        first_row_treatment = data.get("first_row_treatment", False)
        
        if confidence_level is not None:
            confidence_level = int(confidence_level)
        
        result = calculate_odds_ratio(data, db, alpha, yates_correction, confidence_level, first_row_treatment)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
