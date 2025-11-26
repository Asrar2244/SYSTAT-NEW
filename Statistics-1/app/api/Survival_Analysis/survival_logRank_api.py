from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, statistics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.multitest import multipletests
import tempfile
import os
from datetime import datetime

survival_logrank_api = Blueprint('survival_logrank_api', __name__)

def calculate_mean_survival_time(kmf):
    survival_probs = kmf.survival_function_.iloc[:, 0].values
    time_points = kmf.timeline
    mean_survival_time = np.sum(survival_probs * time_points)
    return mean_survival_time

def calculate_percentiles(kmf, percentiles=[25, 50, 75]):
    percentiles_values = {}
    for percentile in percentiles:
        time_at_percentile = kmf.survival_function_[kmf.survival_function_ <= (100 - percentile) / 100].index[-1]
        percentiles_values[percentile] = time_at_percentile
    return percentiles_values

def convert_to_native(value):
    if isinstance(value, (np.int64, np.float64)):
        return value.item()
    return value

@survival_logrank_api.route('/logrank_test', methods=['POST'])
def logrank_test():
    try:
        data = request.get_json()

        # Extract options
        input_format = data.get('input_format', 'raw')
        datasets = data.get('datasets', [])
        options = data.get('options', {})

        # Options for graphing and results
        graph_options = options.get('graph_options', {})
        results_options = options.get('results_options', {})

        if input_format == 'raw':
            df = pd.DataFrame(datasets)
            groups = df['group'].unique()
            if len(groups) != 2:
                raise ValueError("Logrank test requires exactly two groups in raw format.")
            group1 = df[df['group'] == groups[0]]
            group2 = df[df['group'] == groups[1]]

        elif input_format == 'indexed':
            if len(datasets) != 2:
                raise ValueError("Logrank test requires exactly two groups in indexed format.")
            group_names = list(datasets.keys())
            group1_data = datasets[group_names[0]]
            group2_data = datasets[group_names[1]]
            group1 = pd.DataFrame({'time': group1_data['time'], 'status': group1_data['status'], 'group': group_names[0]})
            group2 = pd.DataFrame({'time': group2_data['time'], 'status': group2_data['status'], 'group': group_names[1]})
            df = pd.concat([group1, group2], ignore_index=True)
        else:
            raise ValueError("Invalid input format. Must be 'raw' or 'indexed'.")

        kmf1 = KaplanMeierFitter()
        kmf2 = KaplanMeierFitter()

        kmf1.fit(group1['time'], group1['status'], label=str(group1['group'].iloc[0]))
        kmf2.fit(group2['time'], group2['status'], label=str(group2['group'].iloc[0]))

        # Perform the Logrank Test
        results = statistics.logrank_test(
            group1['time'], group2['time'],
            event_observed_A=group1['status'],
            event_observed_B=group2['status']
        )

        # Calculate mean survival times and percentiles
        mean_survival_time_group1 = calculate_mean_survival_time(kmf1)
        mean_survival_time_group2 = calculate_mean_survival_time(kmf2)
        percentiles_group1 = calculate_percentiles(kmf1)
        percentiles_group2 = calculate_percentiles(kmf2)

        # Prepare response data
        response = {
            'title': 'Kaplan-Meier Survival Analysis: Log-Rank ' + datetime.now().strftime('%d %B %Y %H:%M:%S'),
            'date': datetime.now().strftime('%d %B %Y %H:%M:%S'),
            'data_source': 'Data 1 in Notebook1',
            'event_labels': {'1': 'Event', '0': 'Censor'},
            'time_unit': None,
            'groups': [str(group1['group'].iloc[0]), str(group2['group'].iloc[0])],
            'group_1_details': {
                'group_name': str(group1['group'].iloc[0]),
                'events': convert_to_native(group1['status'].sum()),
                'censored': convert_to_native(len(group1) - group1['status'].sum()),
                'mean_survival_time': convert_to_native(mean_survival_time_group1),
                'percent_censored': round((len(group1) - group1['status'].sum()) / len(group1) * 100, 2),
                'survival_times': {
                    'mean': convert_to_native(mean_survival_time_group1),
                    '25_percentile': convert_to_native(percentiles_group1.get(25, '--')),
                    '50_percentile': convert_to_native(percentiles_group1.get(50, '--')),
                    '75_percentile': convert_to_native(percentiles_group1.get(75, '--'))
                }
            },
            'group_2_details': {
                'group_name': str(group2['group'].iloc[0]),
                'events': convert_to_native(group2['status'].sum()),
                'censored': convert_to_native(len(group2) - group2['status'].sum()),
                'mean_survival_time': convert_to_native(mean_survival_time_group2),
                'percent_censored': round((len(group2) - group2['status'].sum()) / len(group2) * 100, 2),
                'survival_times': {
                    'mean': convert_to_native(mean_survival_time_group2),
                    '25_percentile': convert_to_native(percentiles_group2.get(25, '--')),
                    '50_percentile': convert_to_native(percentiles_group2.get(50, '--')),
                    '75_percentile': convert_to_native(percentiles_group2.get(75, '--'))
                }
            },
            'logrank_test': {
                'statistic': convert_to_native(results.test_statistic),
                'p_value': convert_to_native(results.p_value),
                'result': "Different survival functions" if results.p_value < 0.05 else "No significant difference"
            }
        }

        return jsonify({
            'status': 'success',
            'data': response
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
