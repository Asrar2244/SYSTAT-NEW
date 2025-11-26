from flask import Blueprint, request, jsonify
import pandas as pd
from lifelines import CoxPHFitter
from io import StringIO
from datetime import datetime

cox_stratified_model_api = Blueprint('cox_stratified_model_api', __name__)

@cox_stratified_model_api.route('/cox_stratified_model', methods=['POST'])
def cox_stratified_model():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Missing JSON body")

        # Required Data Inputs
        raw_data = data.get("data")
        time_column = data.get("time_column")
        event_column = data.get("event_column")
        covariates = data.get("covariates")
        stratify_by = data.get("stratify_by", None)

        if not all([raw_data, time_column, event_column, covariates]):
            raise ValueError("Missing one or more required fields: data, time_column, event_column, covariates")

        # Convert raw data to DataFrame
        df = pd.DataFrame(raw_data)

        # Handle missing values (if any)
        df = df.dropna(subset=[time_column, event_column] + covariates)

        # Criterion Options
        method = data.get("criterion", {}).get("method", "complete")
        
        # Stepwise options (only for stepwise method)
        p_to_enter = float(data.get("criterion", {}).get("p_to_enter", 0.05)) if method == "stepwise" else None
        p_to_remove = float(data.get("criterion", {}).get("p_to_remove", 0.10)) if method == "stepwise" else None
        max_steps = int(data.get("criterion", {}).get("max_steps", 10)) if method == "stepwise" else None
        tolerance = float(data.get("criterion", {}).get("tolerance", 1e-8))
        step_length = float(data.get("criterion", {}).get("step_length", 1.0))
        max_iter = int(data.get("criterion", {}).get("max_iterations", 20))

        # Result Options
        include_descriptive_stats = data.get("results", {}).get("include_descriptive_stats", False)
        include_covariance_matrix = data.get("results", {}).get("include_covariance_matrix", False)
        include_survival_table = data.get("results", {}).get("include_survival_table", True)

        # Graph Options
        graph_options = data.get("graph", {})
        group_color = graph_options.get("group_color", "grayscale")
        show_censored = graph_options.get("show_censored", True)
        show_failures = graph_options.get("show_failures", False)
        survival_scale = graph_options.get("survival_scale", "fraction")

        selected_covariates = []
        best_model = None
        best_aic = float("inf")

        # Stepwise Method
        if method == "stepwise":
            for _ in range(max_steps):
                best_candidate = None
                for cov in [c for c in covariates if c not in selected_covariates]:
                    try:
                        model = CoxPHFitter(tie_method="Efron")
                        model.fit(
                            df[[time_column, event_column] + selected_covariates + [cov]],
                            duration_col=time_column,
                            event_col=event_column,
                            step_size=step_length,
                            robust=True
                        )
                        p_value = model.summary.loc[cov, 'p']
                        print(f"Covariate: {cov}, p-value: {p_value}")  # Debug print
                        if p_value < p_to_enter:
                            best_candidate = cov
                            break
                    except Exception as e:
                        print(f"Error with covariate {cov}: {e}")
                        continue

                if best_candidate:
                    selected_covariates.append(best_candidate)
                else:
                    break

            if not selected_covariates:
                raise ValueError("No covariates met the p-to-enter criterion.")
            best_model = CoxPHFitter()
            best_model.fit(df[[time_column, event_column] + selected_covariates], duration_col=time_column, event_col=event_column)

        # Complete Method
        elif method == "complete":
            selected_covariates = covariates
            best_model = CoxPHFitter()

            # Fit the model with a manual iteration approach
            likelihood_values = []
            
            # Fit the model step by step
            best_model.fit(df[[time_column, event_column] + selected_covariates], duration_col=time_column, event_col=event_column)

            # Report Likelihood Values after each iteration if required
            report_likelihood = data.get("criterion", {}).get("report_likelihood", False)  # Check if report_likelihood is provided
            if report_likelihood:
                # The `log_likelihood_` stores the log-likelihood value of the fitted model
                likelihood_values.append(best_model.log_likelihood_)
                print(f"Log-Likelihood after model fit: {best_model.log_likelihood_}")

            # The model is now fit, we report the log-likelihood after fitting
            result = {
                "log_likelihood_values": likelihood_values,  # This will return a list of likelihood values
                "model_summary": best_model.summary.to_dict()
            }

        summary_df = best_model.summary.reset_index().rename(columns={'index': 'covariate'})
        result = {
            "title": f"Cox Proportional Hazards Model Results - {datetime.now().strftime('%d %B %Y %H:%M:%S')}",
            "model_info": {
                "method": method,
                "alpha": p_to_enter,
                "selected_covariates": selected_covariates,
                "AIC_partial": round(best_model.AIC_partial_, 3),
                "tolerance": tolerance,
                "max_iterations": max_iter
            },
            "cox_model_summary": summary_df.to_dict(orient="records")
        }

        if include_descriptive_stats:
            result["descriptive_stats"] = df[selected_covariates].describe().to_dict()

        if include_covariance_matrix:
            result["covariance_matrix"] = best_model.variance_matrix_.to_dict()

        if include_survival_table:
            try:
                survival_table = best_model.predict_survival_function(df[selected_covariates])
                result["survival_table"] = survival_table.to_dict()
            except Exception:
                result["survival_table"] = "Could not compute survival table due to input or convergence issue."

        # Optional Graph output configs can be stored for frontend use
        result["graph_settings"] = {
            "group_color": group_color,
            "show_censored": show_censored,
            "show_failures": show_failures,
            "survival_scale": survival_scale
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error_message": str(e),
            "error_code": "COX_STRATIFIED_MODEL_ERROR"
        }), 400
