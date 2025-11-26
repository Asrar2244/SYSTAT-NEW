from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import norm
import tempfile
import os
from datetime import datetime

gehan_breslow_api = Blueprint('gehan_breslow_api', __name__)

def km_summary_table(kmf, timeline):
    table = []
    survival_probs = kmf.predict(timeline)
    se = kmf.confidence_interval_survival_function_
    se = (se[kmf._label + "_upper_0.95"] - se[kmf._label + "_lower_0.95"]) / (2 * norm.ppf(0.975))  # approximate SE

    for i, t in enumerate(timeline):
        if t in kmf.event_table.index and kmf.event_table.loc[t]['observed'] > 0:
            row = {
                'event_time': round(t, 3),
                'n_events': int(kmf.event_table.loc[t]['observed']),
                'n_at_risk': int(kmf.event_table.loc[t]['at_risk']),
                'survival_prob': round(survival_probs.loc[t], 3),
                'std_error': round(se.loc[t], 3) if t in se.index else None
            }
            table.append(row)
    return table

def calculate_summary_stats(kmf):
    try:
        mean = kmf.mean_survival_time_
    except:
        mean = None
    conf_int = kmf.confidence_interval_
    lower = conf_int.iloc[:, 0].iloc[-1]
    upper = conf_int.iloc[:, 1].iloc[-1]

    percentiles = {}
    for p in [25, 50, 75]:
        try:
            percentiles[str(p)] = float(kmf.percentile(p))
        except:
            percentiles[str(p)] = None

    return {
        'mean': round(mean, 3) if mean else None,
        'std_error': round(kmf._cumulative_density_.std().values[0], 3) if hasattr(kmf, "_cumulative_density_") else None,
        'conf_int_lower': round(lower, 3) if lower else None,
        'conf_int_upper': round(upper, 3) if upper else None,
        'percentiles': percentiles
    }

@gehan_breslow_api.route('/gehan_breslow_test', methods=['POST'])
def gehan_breslow_test():
    try:
        data = request.get_json()
        input_format = data.get('input_format', 'raw')
        datasets = data.get('datasets', [])
        options = data.get('options', {})
        results_options = options.get('results_options', {})
        posthoc_options = options.get('posthoc_options', {})

        time_unit = results_options.get('time_unit', 'None')
        show_cumulative_probability_table = results_options.get('show_cumulative_probability_table', True)
        show_confidence_intervals = results_options.get('show_confidence_intervals', True)

        if input_format == 'raw':
            df = pd.DataFrame(datasets)
        elif input_format == 'indexed':
            df_parts = []
            for group_name, group_data in datasets.items():
                group_df = pd.DataFrame({
                    'time': group_data['time'],
                    'status': group_data['status'],
                    'group': group_name
                })
                df_parts.append(group_df)
            df = pd.concat(df_parts, ignore_index=True)
        else:
            raise ValueError("input_format must be 'raw' or 'indexed'")

        if 'group' not in df.columns or 'time' not in df.columns or 'status' not in df.columns:
            raise ValueError("Missing required columns: 'group', 'time', 'status'")

        groups = df['group'].unique()
        if len(groups) != 2:
            raise ValueError("Exactly 2 groups are required for Gehan-Breslow test.")

        overall_summary = {
            'total': len(df),
            'missing': int(df.isnull().any(axis=1).sum()),
            'events': int(df['status'].sum()),
            'censored': int((1 - df['status']).sum()),
            'percent_censored': round((1 - df['status']).mean() * 100, 2)
        }

        group_details = []
        for group in groups:
            sub = df[df['group'] == group]
            kmf = KaplanMeierFitter()
            kmf.fit(sub['time'], sub['status'], label=str(group))

            timeline = kmf.timeline
            table = km_summary_table(kmf, timeline)
            stats = calculate_summary_stats(kmf)

            group_summary = {
                'group': group,
                'survival_table': table,
                'summary': {
                    'number_of_cases': len(sub),
                    'missing': int(sub.isnull().any(axis=1).sum()),
                    'events': int(sub['status'].sum()),
                    'censored': int((1 - sub['status']).sum()),
                    'percent_censored': round((1 - sub['status']).mean() * 100, 2),
                    'survival_time': {
                        'mean': stats['mean'],
                        'std_error': stats['std_error'],
                        '95%_CI_lower': stats['conf_int_lower'],
                        '95%_CI_upper': stats['conf_int_upper']
                    },
                    'percentiles': {
                        '25': stats['percentiles']['25'],
                        '50_median': stats['percentiles']['50'],
                        '75': stats['percentiles']['75']
                    }
                }
            }
            group_details.append(group_summary)

        test_result = multivariate_logrank_test(
            df['time'], df['group'], df['status'], weightings='wilcoxon'
        )

        stat = round(test_result.test_statistic, 4)
        p_val = round(test_result.p_value, 4)
        df_test = int(test_result.degrees_of_freedom)

        interpretation = (
            "There is no statistically significant difference (P = {})".format(p_val)
            if p_val >= 0.05
            else "There is a statistically significant difference (P = {})".format(p_val)
        )

        return jsonify({
            "title": f"Kaplan-Meier Survival Analysis:  Gehan-Breslow\t{datetime.now().strftime('%d %B %Y %H:%M:%S')}",
            "data_source": options.get("data_source", "Not Provided"),
            "event_label": 1,
            "censor_label": 0,
            "time_unit": time_unit,
            "results_options": {
                "time_unit": time_unit,
                "show_cumulative_probability_table": show_cumulative_probability_table,
                "show_confidence_intervals": show_confidence_intervals
            },
            "posthoc_options": posthoc_options,
            "groups": group_details,
            "overall_summary": overall_summary,
            "gehan_breslow_test": {
                "statistic": stat,
                "df": df_test,
                "p_value": p_val,
                "interpretation": interpretation
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error_message': str(e),
            'error_code': 'GEHAN_BRESLOW_ERROR'
        }), 400
