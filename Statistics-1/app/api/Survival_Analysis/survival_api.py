from flask import Blueprint, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from lifelines import KaplanMeierFitter
import numpy as np

survival_api = Blueprint('survival_api', __name__)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def apply_group_color_style(style):
    """
    Apply color styles:
    - black: single black line
    - grayscale: dimgray
    - incrementing: tab:blue (for single group)
    """
    if style == "black":
        return 'black'
    elif style == "grayscale":
        return 'dimgray'
    elif style == "incrementing":
        return 'tab:blue'
    else:
        return 'black'  # fallback

@survival_api.route('/api/survival/single-group', methods=['POST'])
def single_group_survival():
    try:
        # Determine input type: JSON or multipart form with file
        if request.content_type and "application/json" in request.content_type:
            input_json = request.get_json(force=True)
            data = input_json.get("data")
            graph_options = input_json.get("graph_options", {})
            results_options = input_json.get("results_options", {})
            if data is None:
                return jsonify({
                    "success": False,
                    "error_message": "Missing 'data' field in JSON input",
                    "error_code": "NO_DATA"
                }), 400
            df = pd.DataFrame(data)

        elif 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file, sep=None, engine='python', names=["time", "status"], header=None)

            # Extract graph_options from form with defaults
            graph_options = {
                "plot_title": request.form.get('plot_title', 'Kaplan-Meier Survival Curve'),
                "x_label": request.form.get('x_label', 'Time'),
                "y_label": request.form.get('y_label', 'Survival Probability'),
                "group_color": request.form.get('group_color', 'black'),
                "survival_scale": request.form.get('survival_scale', 'fraction'),
                "additional_plot_statistics": {
                    "enabled": request.form.get('additional_plot_statistics.enabled', 'false').lower() == 'true',
                    "type": request.form.get('additional_plot_statistics.type', '95% CI')
                }
            }
            results_options = {
                "report": {
                    "cumulative_probability_table": request.form.get('report.cumulative_probability_table', 'false').lower() == 'true',
                    "p_value_for_multiple_comparisons": request.form.get('report.p_value_for_multiple_comparisons', 'false').lower() == 'true'
                },
                "time_unit": request.form.get('time_unit', 'days'),
                "worksheet_options": {
                    "confidence_interval_95": request.form.get('worksheet_options.confidence_interval_95', 'false').lower() == 'true'
                }
            }
        else:
            return jsonify({
                "success": False,
                "error_message": "No input data provided. Send JSON with 'data' or upload CSV file.",
                "error_code": "NO_INPUT"
            }), 400

        # Validate required columns
        if "time" not in df.columns or "status" not in df.columns:
            return jsonify({
                "success": False,
                "error_message": "Input data must contain 'time' and 'status' columns",
                "error_code": "INVALID_DATA"
            }), 400

        # Map status strings to 1/0
        if df["status"].dtype == object:
            df["status"] = df["status"].str.lower().map({"failure": 1, "censored": 0})
        else:
            # If status is numeric, accept 0/1 directly
            df["status"] = df["status"].astype(int)

        if df["status"].isnull().any():
            return jsonify({
                "success": False,
                "error_message": "Status values must be 'failure' or 'censored' (case insensitive) or 1/0.",
                "error_code": "INVALID_STATUS"
            }), 400

        # Fit Kaplan-Meier model
        kmf = KaplanMeierFitter()
        kmf.fit(df["time"], event_observed=df["status"])

        # Prepare plotting
        line_color = apply_group_color_style(graph_options.get("group_color", "black"))

        plt.figure(figsize=(8, 5))
        ax = plt.subplot(111)

        # Determine if showing CI or SE bars
        add_stats = graph_options.get("additional_plot_statistics", {})
        show_ci = add_stats.get("enabled", False) and add_stats.get("type", "").lower() == "95% ci"
        show_se = add_stats.get("enabled", False) and add_stats.get("type", "").lower() == "standard error bars"

        # Handle survival scale
        survival_scale = graph_options.get("survival_scale", "fraction").lower()
        if survival_scale == "percentage":
            plt.ylabel(graph_options.get("y_label", "Survival Probability") + " (%)")
        else:
            plt.ylabel(graph_options.get("y_label", "Survival Probability"))

        # Plot survival function
        kmf.plot_survival_function(ci_show=show_ci, ax=ax, color=line_color)

        # If SE bars requested, approximate from CI
        if show_se and kmf.confidence_interval_ is not None:
            ci_lower = kmf.confidence_interval_.iloc[:, 0]
            ci_upper = kmf.confidence_interval_.iloc[:, 1]
            se = (ci_upper - ci_lower) / 3.92  # ~1.96*2 for 95% CI
            times = kmf.survival_function_.index.values
            survival_prob = kmf.survival_function_.values.flatten()
            plt.errorbar(times, survival_prob, yerr=se, fmt='o', color=line_color, alpha=0.3, label='SE bars')

        plt.title(graph_options.get("plot_title", "Kaplan-Meier Survival Curve"))
        plt.xlabel(graph_options.get("x_label", "Time"))
        plt.grid(True)
        plt.tight_layout()

        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        graph_base64 = base64.b64encode(img.getvalue()).decode()

        # Prepare response
        response = {
            "success": True,
            "plot_base64": f"data:image/png;base64,{graph_base64}",
        }

        # Add cumulative probability table if requested
        report_opts = results_options.get("report", {})
        if report_opts.get("cumulative_probability_table", False):
            sf_df = kmf.survival_function_.reset_index().rename(
                columns={"timeline": "time", kmf.survival_function_.columns[0]: "survival_probability"}
            )
            if survival_scale == "percentage":
                sf_df["survival_probability"] *= 100
            response["cumulative_probability_table"] = sf_df.to_dict(orient="records")

        # Single group => no multiple comparison p-value
        if report_opts.get("p_value_for_multiple_comparisons", False):
            response["p_value_for_multiple_comparisons"] = None

        response["summary"] = {
            "median_survival_time": float(kmf.median_survival_time_) if kmf.median_survival_time_ is not None else None,
            "n_observations": int(len(df)),
            "n_events": int(df["status"].sum()),
            "timeline": kmf.timeline.tolist()
        }

        response["time_unit"] = results_options.get("time_unit", "days")

        # Convert numpy types before jsonify
        response = convert_numpy_types(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success": False,
            "error_message": str(e),
            "error_code": "EXCEPTION"
        }), 500    