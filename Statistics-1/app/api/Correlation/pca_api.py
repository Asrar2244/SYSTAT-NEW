import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.decomposition import PCA
from scipy.stats import chi2
from pingouin import multivariate_normality

from app.api.helpers.logger import Logger
from app.api.helpers.constant import (
    VALUE_ERROR_MSG, KEY_ERROR_MSG, TYPE_ERROR_MSG, UNEXPECTED_ERROR_MSG,
    LOG_VALUE_ERROR, LOG_KEY_ERROR, LOG_TYPE_ERROR, LOG_UNEXPECTED_ERROR, PCA_LOG_FILE_PATH
)

pca_api = Blueprint('correlation_api', __name__)

def perform_normality_test(data, method, pval_threshold):
    if method == "henze-zirkler":
        hz_stat, p_value, normality = multivariate_normality(data, alpha=pval_threshold)
        return {
            "method": "Henze-Zirkler",
            "statistic": round(float(hz_stat), 4),
            "p_value": round(float(p_value), 4),
            "passed": bool(p_value > pval_threshold)
        }
    else:
        raise ValueError("Invalid normality statistic method. Only 'henze-zirkler' is supported currently.")

def get_selected_indices(method, eigenvalues, explained_ratio, eigen_cutoff, percent_cutoff, n_components):
    method = method.lower()
    if method == "average eigenvalue":
        threshold = np.mean(eigenvalues)
        return [i for i, val in enumerate(eigenvalues) if val >= threshold]

    elif method == "minimum eigenvalue":
        return [i for i, val in enumerate(eigenvalues) if val >= eigen_cutoff]

    elif method == "minimum percentage":
        cumulative, selected = 0, []
        for i, val in enumerate(explained_ratio):
            cumulative += val * 100
            selected.append(i)
            if cumulative >= percent_cutoff:
                break
        return selected

    elif method == "number of components":
        if n_components is None or n_components < 1:
            raise ValueError("Number of components must be a positive integer.")
        return list(range(min(n_components, len(eigenvalues))))

    elif method == "all":
        return list(range(len(eigenvalues)))

    else:
        raise ValueError("Invalid selection method.")

def bartlett_test_eigenvalues(eigenvalues, n):
    """
    Bartlett's test for sphericity / equality of eigenvalues.
    Returns statistic, df, pvalue.
    Uses formula:
    M = - (n - 1 - (2p + 5)/6 ) * ln( prod(lambda_i) / ( (mean lambda)^p ) )
    df = p(p-1)/2
    """
    p = len(eigenvalues)
    # ensure positive eigenvalues
    eig = np.array(eigenvalues, dtype=float)
    # avoid zeros by small epsilon
    eps = 1e-12
    eig = np.where(eig <= 0, eps, eig)
    prod_term = np.prod(eig)
    mean_eig = np.mean(eig)
    numerator = prod_term
    denominator = (mean_eig ** p)
    # If denom zero or negative, handle:
    if denominator <= 0:
        statistic = np.nan
        p_value = np.nan
    else:
        M = - (n - 1 - (2 * p + 5) / 6.0) * np.log(numerator / denominator)
        df = p * (p - 1) / 2.0
        p_value = 1 - chi2.cdf(M, df)
        statistic = float(M)
    return statistic, p, p_value

def bootstrap_eigenvector_se(corr_matrix, n_boot=500, random_state=None):
    """
    Bootstraps rows (observations) with replacement to estimate standard errors
    of eigenvector entries of the correlation matrix.
    Returns eigenvectors (sorted descending) and standard errors for each entry.
    """
    rng = np.random.default_rng(random_state)
    n_obs = corr_matrix.shape[0]  # NOTE: corr_matrix here is computed from standardized data (observations x vars)? We'll pass original data for bootstrapping.
    # The function calling this will handle passing the original data matrix (n_samples x n_features)
    raise RuntimeError("bootstrap_eigenvector_se should be called with data matrix, not correlation matrix.")

@pca_api.route("/principal-components", methods=["POST"])
def principal_components():
    logger = Logger(PCA_LOG_FILE_PATH)

    try:
        # Load data
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename.lower()
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Use .csv, .xls or .xlsx")
            options = request.form.to_dict()
        else:
            body = request.get_json()
            if not body or "data" not in body:
                raise ValueError("Missing 'data' field in JSON.")
            df = pd.DataFrame(body["data"])
            options = body.get("options", {})

        numeric_df = df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            raise ValueError("Input data must have at least 2 numeric columns for PCA.")

        # Extract PCA options
        criterion = options.get("criterion", {})
        matrix_type = criterion.get("matrix", "correlation").lower()
        significance_level = float(criterion.get("significance_level", 0.05))
        confidence_interval = float(criterion.get("confidence_interval", 95))
        if not (1 <= confidence_interval <= 99):
            raise ValueError("Confidence interval must be between 1 and 99.")

        selection_method = criterion.get("selection_method", {})
        method = selection_method.get("method", "all").lower()
        eigen_cutoff = float(selection_method.get("min_eigenvalue", 0))
        percent_cutoff = float(selection_method.get("min_percent", 0))
        n_components = int(selection_method.get("n_components", 0)) if selection_method.get("n_components") else None

        assumption_checking = options.get("assumption_checking", {})
        normality = bool(assumption_checking.get("normality", False))
        pval_reject = float(assumption_checking.get("p_value", 0.05))
        normality_stat = assumption_checking.get("normality_statistic", "henze-zirkler").lower()

        residuals_opts = options.get("residuals", {})
        include_component_scores = bool(residuals_opts.get("component_scores", False))
        include_residuals = bool(residuals_opts.get("residuals", False))

        results_opts = options.get("results", {})
        include_corr_matrix = bool(results_opts.get("correlation_matrix", False))
        include_component_loadings = bool(results_opts.get("component_loadings", False))
        include_prop_variance = bool(results_opts.get("proportion_of_variance_explained", False))
        include_fitted_corr = bool(results_opts.get("fitted_correlation_matrix", False))
        include_diff_corr = bool(results_opts.get("difference_between_original_and_fitted", False))
        include_scree_data = bool(results_opts.get("scree_plot_data", False))
        include_eigen_se = bool(results_opts.get("eigenvector_standard_errors", False))
        bootstrap_samples = int(results_opts.get("bootstrap_samples", 500))

        # Standardize or center data based on matrix_type
        if matrix_type == "correlation":
            # zscore uses ddof=0; but for PCA/eigen on correlation we standardize by subtracting mean and dividing by sample std (ddof=1)
            # However zscore from scipy uses ddof=0; to replicate SigmaPlot/typical correlation matrix we compute manually:
            standardized = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=1)
            data_processed = standardized.values
        elif matrix_type == "covariance":
            data_processed = (numeric_df - numeric_df.mean()).values
        else:
            raise ValueError("Invalid matrix type specified. Must be 'correlation' or 'covariance'.")

        n_samples, n_vars = data_processed.shape

        # Descriptive statistics (use sample std ddof=1 to match SigmaPlot)
        desc_stats = {
            col: {
                "mean": round(float(numeric_df[col].mean()), 4),
                "std": round(float(numeric_df[col].std(ddof=1)), 4)
            }
            for col in numeric_df.columns
        }

        obs_details = {
            "total_observations": int(df.shape[0]),
            # Count a row as missing if any cell in its row non-numeric OR NaN as per user's note
            "missing_observations": int(df.shape[0] - numeric_df.dropna().shape[0]),
            "valid_observations": int(numeric_df.dropna().shape[0])
        }

        # Assumption checking
        assumption_results = {}
        if normality:
            assumption_results["normality_test"] = perform_normality_test(data_processed, normality_stat, pval_reject)

        # Compute correlation or covariance matrix used for eigen-decomposition
        if matrix_type == "correlation":
            corr_matrix = np.corrcoef(data_processed, rowvar=False)
            total_variance = float(np.trace(corr_matrix))  # should equal n_vars
        else:
            corr_matrix = np.cov(data_processed, rowvar=False)
            total_variance = float(np.trace(corr_matrix))

        # Eigen decomposition on the correlation/covariance matrix (use eigh for symmetric)
        eigvals_raw, eigvecs_raw = np.linalg.eigh(corr_matrix)
        # eigh returns ascending eigenvalues, so reverse to descending
        idx_desc = eigvals_raw.argsort()[::-1]
        eigvals = eigvals_raw[idx_desc]
        eigvecs = eigvecs_raw[:, idx_desc]

        # Normalize eigenvectors sign convention: ensure first element positive for deterministic sign
        for j in range(eigvecs.shape[1]):
            if eigvecs[0, j] < 0:
                eigvecs[:, j] *= -1.0

        # explained variance ratio
        explained_ratio_all = eigvals / np.sum(eigvals)

        # Bartlett-style chi-square test for equality of eigenvalues (all)
        bart_stat_all, df_chi_all, bart_p_all = bartlett_test_eigenvalues(eigvals, n_samples)

        # Chi-square test for last K eigenvalues equal: default K = n_vars - 1 (i.e., last 3 if p=4)
        # User asked specifically "The last 3 eigenvalues are equal." We'll compute for k = n_vars - 1 by default
        k_last = int(selection_method.get("last_k", n_vars - 1)) if isinstance(selection_method := criterion.get("selection_method", {}), dict) else (n_vars - 1)
        # compute test for last k_last eigenvalues: treat the subset eigenvals[-k_last:]
        if k_last >= 1 and k_last < n_vars:
            subset = eigvals[-k_last:]
            bart_stat_last, df_last, bart_p_last = bartlett_test_eigenvalues(subset, n_samples)
        else:
            bart_stat_last, df_last, bart_p_last = None, None, None

        # Build eigen_summary comparable to SigmaPlot output (list of dicts)
        eigen_summary = []
        cumulative = 0.0
        for i, val in enumerate(eigvals):
            prop = float(val / np.sum(eigvals))
            cumulative += prop
            diff = float(val - eigvals[i + 1]) if i + 1 < len(eigvals) else None
            eigen_summary.append({
                "Component": f"PC{i+1}",
                "Eigenvalue": round(float(val), 4),
                "Difference": round(diff, 4) if diff is not None else None,
                "Proportion": round(prop, 4),
                "Cumulative": round(cumulative, 4)
            })

        # Determine selected indices by selection method applied on eigenvalues or explained_ratio
        selected_indices = get_selected_indices(method, eigvals, explained_ratio_all, eigen_cutoff, percent_cutoff, n_components)
        if not selected_indices:
            raise ValueError("No components selected with given criteria.")

        # Number of in-model principal components (average eigenvalue criterion for correlation matrix average = 1.0)
        average_eig = np.mean(eigvals)
        in_model_count = int(np.sum(eigvals >= average_eig))

        # Prepare principal components using sklearn PCA on standardized data for component scores (keeps ordering consistent)
        # Use n_components = full to allow extracting selected components later
        pca_full = PCA(n_components=n_vars)
        pca_full.fit(data_processed)
        # sklearn's components_ are in order of explained variance descending
        components_sklearn = pca_full.components_  # shape (n_vars, n_vars)
        transformed_full = pca_full.transform(data_processed)

        # Build selected matrices
        sel_idx = selected_indices
        # Eigenvectors and eigenvalues for selected components (from eigvecs/eigvals)
        eigvals_selected = eigvals[sel_idx]
        eigvecs_selected = eigvecs[:, sel_idx]  # columns are eigenvectors
        # Component loadings: for correlation matrix loadings = eigenvectors * sqrt(eigenvalue)
        loadings = eigvecs_selected * np.sqrt(eigvals_selected[np.newaxis, :])

        # Component scores: project standardized data onto eigenvectors (scores consistent with PCA)
        # Use eigenvectors selected: scores = data_processed dot eigvecs_selected
        component_scores = (data_processed @ eigvecs_selected)
        pc_labels = [f"PC{i+1}" for i in sel_idx]

        # Fitted correlation matrix from selected components: sum_j lambda_j * v_j v_j^T
        reconstructed_corr = np.zeros_like(corr_matrix, dtype=float)
        for j, idx in enumerate(sel_idx):
            v = eigvecs[:, idx][:, np.newaxis]  # column
            reconstructed_corr += float(eigvals[idx]) * (v @ v.T)
        # If not full model then the remainder approximated by zeros; reconstructed_corr approximates corr_matrix using selected PCs

        # Fitted correlation (as correlation values) — ensure diagonal ones if small numeric adjustments
        # For correlation matrix, reconstructed_corr should be close but diagonal might equal eigenvalues sum for selected components; to get fitted correlation as in SigmaPlot we use:
        fitted_corr_matrix = reconstructed_corr.copy()
        # Convert to correlations by dividing by sqrt of diag if necessary (but for sum λ v v^T on correlation matrix, diagonals equal sum λ * v_i^2; not necessarily 1 for partial components)
        # SigmaPlot's "fitted correlation matrix" is the correlation matrix approximated by in-model principal components: for that use
        # fitted = V_selected * diag(λ_selected) * V_selected^T
        # Which we already computed as reconstructed_corr. We'll output that.

        # Difference between original and fitted
        diff_orig_fitted = corr_matrix - fitted_corr_matrix

        # Standard errors for eigenvector entries — approximate via bootstrap over observations if requested
        eigenvector_se = None
        if include_eigen_se:
            # Bootstrap rows of data_processed with replacement and compute eigenvectors of correlation matrix each time
            n_boot = max(50, bootstrap_samples)  # ensure at least some
            boot_eigvecs = np.zeros((n_boot, eigvecs.shape[0], eigvecs.shape[1]))
            rng = np.random.default_rng(42)
            for b in range(n_boot):
                sample_idx = rng.integers(0, n_samples, n_samples)
                sample = data_processed[sample_idx, :]
                # compute corr on sample
                try:
                    sample_corr = np.corrcoef(sample, rowvar=False)
                    e_vals_b, e_vecs_b = np.linalg.eigh(sample_corr)
                    idxb = e_vals_b.argsort()[::-1]
                    e_vecs_b = e_vecs_b[:, idxb]
                    # Align sign to original eigenvectors for comparability
                    for j in range(e_vecs_b.shape[1]):
                        if np.dot(e_vecs_b[:, j], eigvecs[:, j]) < 0:
                            e_vecs_b[:, j] *= -1.0
                    boot_eigvecs[b, :, :] = e_vecs_b
                except Exception:
                    # If sample corr fails (e.g., singular), skip by filling with NaNs
                    boot_eigvecs[b, :, :] = np.nan

            # compute std dev across bootstrap replicates for each eigenvector entry
            se_matrix = np.nanstd(boot_eigvecs, axis=0, ddof=1)
            # se_matrix shape (n_vars, n_vars); we want standard errors for entries of PC columns
            eigenvector_se = {f"PC{i+1}": [round(float(se_matrix[r, i]), 4) for r in range(se_matrix.shape[0])] for i in range(se_matrix.shape[1])}

        # Standard errors for loadings: can be approximated by se of eigenvector * sqrt(eigenvalue)
        loadings_se = None
        if eigenvector_se is not None:
            loadings_se = {}
            for i, comp in enumerate([f"PC{i+1}" for i in sel_idx]):
                se_list = eigenvector_se.get(comp.replace("PC", "PC"), None)
                if se_list is None:
                    # attempt mapping using full PCs
                    comp_index = sel_idx[i]
                    se_list = [round(float(se_matrix[r, comp_index] * np.sqrt(eigvals[comp_index]) ), 4) for r in range(n_vars)]
                else:
                    # se_list length may match number of vars
                    comp_index = sel_idx[i]
                    se_list = [round(float(s * np.sqrt(eigvals[comp_index])), 4) for s in se_list]
                loadings_se[comp] = se_list

        # Chi-square test for last k eigenvalues (if requested earlier)
        # If user specifically asked "last 3 eigenvalues are equal" we'll include it as second test
        chi_square_tests = {
            "all_eigenvalues_equal": {
                "Statistic": round(float(bart_stat_all), 3) if bart_stat_all is not None else None,
                "df": float(df_chi_all),
                "p_value": round(float(bart_p_all), 4) if bart_p_all is not None else None,
                "interpretation": "Reject null hypothesis: eigenvalues are significantly different" if bart_p_all is not None and bart_p_all < significance_level else "Fail to reject null hypothesis: eigenvalues may be equal"
            }
        }
        if bart_stat_last is not None:
            chi_square_tests["last_k_eigenvalues_equal"] = {
                "Statistic": round(float(bart_stat_last), 3),
                "df": float(df_last),
                "p_value": round(float(bart_p_last), 4),
                "interpretation": "Reject null hypothesis: last k eigenvalues are significantly different" if bart_p_last is not None and bart_p_last < significance_level else "Fail to reject null hypothesis for last k eigenvalues"
            }

        # Build output
        output = {
            "descriptive_statistics": desc_stats,
            "observation_details": obs_details,
            "total_variance": round(float(total_variance), 4),
            "eigenvalues_of_correlation_matrix": [round(float(v), 6) for v in eigvals],
            "eigen_summary": eigen_summary,
            "number_in_model_components": in_model_count,
            "chi_square_tests": chi_square_tests,
            "explained_variance": [round(float(eigvals[i]), 4) for i in sel_idx],
            "explained_variance_ratio": [round(float(explained_ratio_all[i]), 4) for i in sel_idx],
            "proportion_of_variance_explained": round(float(np.sum(explained_ratio_all[sel_idx])), 4)
        }

        # component loadings (as correlations between original variables and PCs) if requested
        if include_component_loadings:
            # loadings as correlation: for standardized variables, loadings = eigenvectors * sqrt(eigenvalue)
            loading_df = pd.DataFrame(loadings, index=numeric_df.columns, columns=[f"PC{idx+1}" for idx in sel_idx])
            output["component_loadings"] = loading_df.round(4).to_dict(orient="list")

        # component scores if requested
        if include_component_scores:
            pc_df = pd.DataFrame(component_scores, columns=[f"PC{idx+1}" for idx in sel_idx])
            output["component_scores"] = {col: [round(float(x), 10) for x in pc_df[col].tolist()] for col in pc_df.columns}

        # residuals (original standardized - reconstructed using selected components)
        if include_residuals:
            # reconstructed data using selected components: scores * eigvecs_selected^T
            reconstructed_data = component_scores @ eigvecs_selected.T
            residuals = data_processed - reconstructed_data
            residuals_df = pd.DataFrame(residuals, columns=numeric_df.columns)
            output["residuals"] = {col: [round(float(x), 10) for x in residuals_df[col].tolist()] for col in residuals_df.columns}

        # correlation matrix if requested (original)
        if include_corr_matrix:
            output["correlation_matrix"] = pd.DataFrame(corr_matrix, index=numeric_df.columns, columns=numeric_df.columns).round(6).to_dict()

        # fitted correlation matrix if requested
        if include_fitted_corr:
            output["fitted_correlation_matrix"] = pd.DataFrame(fitted_corr_matrix, index=numeric_df.columns, columns=numeric_df.columns).round(6).to_dict()

        # difference between original and fitted
        if include_diff_corr:
            output["difference_between_original_and_fitted"] = pd.DataFrame(diff_orig_fitted, index=numeric_df.columns, columns=numeric_df.columns).round(6).to_dict()

        # scree / explained variance data
        if include_scree_data:
            output["scree_plot_data"] = {
                "explained_variance": [round(float(v), 6) for v in eigvals],
                "explained_variance_ratio": [round(float(v), 6) for v in explained_ratio_all]
            }

        # eigenvector standard errors if requested
        if eigenvector_se is not None:
            # Build mapping similar to SigmaPlot: for PC1, list std errors for each variable
            pc_se_dict = {}
            for j in range(eigvecs.shape[1]):
                pc_name = f"PC{j+1}"
                pc_se_dict[pc_name] = [round(float(se_matrix[r, j]), 4) for r in range(se_matrix.shape[0])]
            output["standard_errors_eigenvectors"] = pc_se_dict

        # standard errors for loadings
        if loadings_se is not None:
            output["standard_errors_loadings"] = loadings_se

        # Component eigenvectors (coefficients)
        eigenvectors_out = {f"PC{i+1}": [round(float(x), 6) for x in eigvecs[:, i].tolist()] for i in range(eigvecs.shape[1])}
        output["eigenvectors_of_correlation_matrix"] = eigenvectors_out

        # component loadings (full set rounded) also for reference if not requested earlier
        output.setdefault("component_loadings_full", pd.DataFrame(eigvecs * np.sqrt(eigvals[np.newaxis, :]), index=numeric_df.columns, columns=[f"PC{i+1}" for i in range(len(eigvals))]).round(6).to_dict(orient="list"))

        # Add assumption_results at top-level as before
        return jsonify({
            "success": True,
            "results": output,
            "assumption_results": assumption_results if assumption_results else None
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
