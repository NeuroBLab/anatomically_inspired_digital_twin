"""
Performs classification analysis on simulated neural responses to object stimuli as described in Froudarakis et al. (2020)

This script loads simulated neural data of video stimuli of moving objects and model metadata, filters neurons
based on brain area and performance, and then trains a classifier to decode
object identity from neural activity. The analysis is repeated for different
subsample sizes of neurons, and results are aggregated across multiple
simulated sessions.
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def setup_plotting_style():
    """Configures Matplotlib settings for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 20,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'figure.figsize': (8, 5),
        'figure.dpi': 300,
    })


def filter_and_select_neurons(data, model_data, area, performance_threshold,
                              sessions, metric='corr_to_ave', layer=None):
    """
    Filters and selects neurons for specified sessions, area, and performance.

    Args:
        data (dict): Raw neural response data.
        model_data (dict): Model metadata containing brain area and performance.
        area (str): The visual area to select neurons from (e.g., 'V1').
        performance_threshold (float): The minimum performance metric for a
            neuron to be included.
        sessions (list): A list of session IDs to process.
        metric (str): The performance metric to use for thresholding.
        layer (str, optional): The cortical layer to select neurons from.

    Returns:
        dict: A dictionary mapping object class labels to a 2D NumPy array of
            neural responses (time x neurons) aggregated across sessions.
    """
    selected_neurons = {}
    for session in sessions:
        area_indices = np.where(
            model_data['model'][session]['brain_area'] == area)[0]
        if layer is not None:
            layer_indices = np.where(
                model_data['model'][session]['layer'] == layer)[0]
            area_indices = np.intersect1d(area_indices, layer_indices)

        performances = model_data['results']['val'][session][metric][area_indices]
        valid_mask = (performances >= performance_threshold)
        selected_neurons[session] = area_indices[valid_mask]

    filtered_data = {}
    for obj_class in data:
        all_session_data = []
        for session in data[obj_class]:
            if session in selected_neurons and len(selected_neurons[session]) > 0:
                session_data = data[obj_class][session][:, :, selected_neurons[session]]
                num_clips, num_frames, _ = session_data.shape
                reshaped = session_data.reshape(num_clips * num_frames, -1)
                all_session_data.append(reshaped)

        if all_session_data:
            filtered_data[obj_class] = np.concatenate(all_session_data, axis=1)
        else:
            filtered_data[obj_class] = np.empty((0, 0))

    return filtered_data


def create_dataframe_from_dict(data_dict):
    """
    Converts a dictionary of neural data into a Pandas DataFrame.

    Args:
        data_dict (dict): A dictionary mapping class labels to 2D NumPy arrays
            of features (time x neurons).

    Returns:
        pd.DataFrame: A DataFrame where rows are samples (time points),
            columns are features (neurons), and a final 'label' column
            contains the class labels.
    """
    features_list = []
    labels_list = []

    for class_label, features in data_dict.items():
        if features.shape[0] > 0:
            features_list.append(features)
            labels_list.append(np.full(features.shape[0], class_label))

    if not features_list:
        return pd.DataFrame(columns=[0, 'label'])

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)

    df = pd.DataFrame(X)
    df['label'] = y
    return df


def mutual_information(y_true, y_pred):
    """
    Computes the mutual information between true and predicted labels.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The mutual information score in bits.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    total_samples = conf_matrix.sum()
    pi = conf_matrix.sum(axis=1) / total_samples
    pj = conf_matrix.sum(axis=0) / total_samples
    pij = conf_matrix / total_samples

    # Add a small epsilon to avoid log(0)
    mi = np.nansum(pij * np.log2(pij / (np.outer(pi, pj) + 1e-12) + 1e-12))
    return mi


def evaluate_classification_model(df, C=1.0, fit_intercept=False, random_state=42):
    """
    Evaluates a classification pipeline using 10-fold stratified CV.

    Args:
        df (pd.DataFrame): DataFrame with features and a 'label' column.
        C (float): Inverse of regularization strength for Logistic Regression.
        fit_intercept (bool): Whether to fit an intercept.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary with mean test accuracy and mutual information.
    """
    if df.shape[0] == 0 or df.shape[1] < 2:
        return {'test_accuracy': np.nan, 'test_mi': np.nan}

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('classifier', OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                C=C,
                fit_intercept=fit_intercept,
                random_state=random_state,
                penalty=None,
                n_jobs=-1
            ),
            n_jobs=-1
        ))
    ])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    mi_scorer = make_scorer(mutual_information, greater_is_better=True)

    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring={'accuracy': 'accuracy', 'mi': mi_scorer},
        return_train_score=False, n_jobs=-1
    )
    return {
        'test_accuracy': np.mean(scores['test_accuracy']),
        'test_mi': np.mean(scores['test_mi']),
    }


def subsample_and_evaluate(df, num_cells_list, num_repetitions=10, C=1.0,
                           fit_intercept=False, base_seed=42):
    """
    Performs classification on random subsamples of neurons.

    Args:
        df (pd.DataFrame): DataFrame with features and labels.
        num_cells_list (list): A list of integers specifying the number of
            neurons to subsample.
        num_repetitions (int): Number of random subsamples to draw for each size.
        C (float): Regularization parameter for the classifier.
        fit_intercept (bool): Whether the classifier should fit an intercept.
        base_seed (int): Base seed for the random number generator.

    Returns:
        dict: A dictionary mapping each number of cells to a list of
            evaluation results from each repetition.
    """
    if df.shape[0] == 0 or df.shape[1] < 2:
        return {
            c: [{'test_accuracy': np.nan, 'test_mi': np.nan}]
            for c in num_cells_list
        }

    results = {}
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    for i, num_cells in enumerate(num_cells_list):
        count_results = []
        for j in range(num_repetitions):
            seed = base_seed + i * 100 + j
            rng = np.random.default_rng(seed)

            if num_cells >= len(feature_cols):
                selected_features = list(feature_cols)
            else:
                selected_features = rng.choice(
                    feature_cols, size=num_cells, replace=False)

            subsampled_df = df[list(selected_features) + [label_col]]

            eval_result = evaluate_classification_model(
                subsampled_df, C=C, fit_intercept=fit_intercept,
                random_state=seed
            )
            count_results.append(eval_result)
        results[num_cells] = count_results
    return results


def find_adaptive_threshold(data, model_data, session_id, min_neuron_count,
                            initial_threshold, metric, layer):
    """
    Finds a performance threshold that yields at least `min_neuron_count`.

    Starts from `initial_threshold` and decreases it until the condition is met
    for area 'AL', which is used as a reference.

    Args:
        data (dict): Raw neural response data.
        model_data (dict): Model metadata.
        session_id (str): The session to check.
        min_neuron_count (int): The minimum number of neurons required.
        initial_threshold (float): The starting performance threshold.
        metric (str): The performance metric for filtering.
        layer (str, optional): The cortical layer.

    Returns:
        float or None: The adjusted threshold, or None if the condition cannot
            be met even at a threshold of 0.
    """
    threshold = initial_threshold
    step = 0.01
    while threshold >= 0:
        filtered_data_al = filter_and_select_neurons(
            data=data, model_data=model_data, area="AL",
            performance_threshold=threshold, sessions=[session_id],
            metric=metric, layer=layer
        )
        df_al = create_dataframe_from_dict(filtered_data_al)
        num_neurons = max(0, df_al.shape[1] - 1)
        if num_neurons >= min_neuron_count:
            return threshold
        threshold -= step
    return None


def main(args):
    """
    Main function to run the classification analysis pipeline.
    """
    setup_plotting_style()

    print("Loading model summary data...")
    model_summary_path = os.path.join(
        args.base_path, f'save/results/{args.model}/summary.pkl')
    with open(model_summary_path, 'rb') as f:
        model_data = pickle.load(f)

    print("Loading simulated object responses...")
    neural_data_path = os.path.join(
        args.base_path, f'save/data/froudarakis/{args.model}_{args.input_data}.pkl')
    with open(neural_data_path, 'rb') as f:
        data = pickle.load(f)

    sessions = args.microns_l23_sessions if args.l23 else args.microns_sessions
    max_neurons_subsampled = max(args.num_sampling_neurons)
    all_session_results = {}

    for session_id in tqdm(sessions, desc="Processing Sessions"):
        print(f"\n--- Processing Session: {session_id} ---")

        # Adaptively find a threshold to ensure enough neurons are available.
        final_threshold = find_adaptive_threshold(
            data, model_data, session_id, max_neurons_subsampled,
            args.performance_threshold, args.selection_metric, args.layer
        )

        if final_threshold is None:
            print(f"Skipping session {session_id}: Not enough neurons in AL.")
            continue
        if final_threshold < args.performance_threshold:
            print(f"  Adjusted threshold to {final_threshold:.2f} for this session.")

        session_dict = {}
        for area in args.visual_areas:
            filtered_data_area = filter_and_select_neurons(
                data=data, model_data=model_data, area=area,
                performance_threshold=final_threshold, sessions=[session_id],
                metric=args.selection_metric, layer=args.layer
            )
            dataframe_area = create_dataframe_from_dict(filtered_data_area)

            if args.no_neuron_sampling:
                eval_res = evaluate_classification_model(
                    dataframe_area, fit_intercept=args.fit_intercept,
                    random_state=args.random_seed
                )
                n_feats = max(0, dataframe_area.shape[1] - 1)
                session_dict[area] = {n_feats: [eval_res]}
            else:
                subsampling_results = subsample_and_evaluate(
                    dataframe_area,
                    num_cells_list=args.num_sampling_neurons,
                    num_repetitions=args.sampling_trials,
                    fit_intercept=args.fit_intercept,
                    base_seed=args.random_seed
                )
                session_dict[area] = subsampling_results
        all_session_results[session_id] = session_dict

    # --- Save results ---
    date_str = datetime.today().strftime('%Y-%m-%d')
    results_dir = os.path.join(args.base_path, "save/data/froudarakis/results")
    os.makedirs(results_dir, exist_ok=True)

    suffix = "l23" if args.l23 else "all_layers"
    filename = (f"{args.model}_{max_neurons_subsampled}_{suffix}_"
                f"{args.performance_threshold}_{args.sampling_trials}_{date_str}.json")
    output_path = os.path.join(results_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(all_session_results, f, indent=4)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classification analysis of simulated neural responses."
    )
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path for project data and results.")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model to analyze.")
    parser.add_argument("--input_data", type=str, required=True,
                        choices=['objects', 'objects_scaled', 'objects_original'],
                        help="Name of the input dataset.")
    parser.add_argument("--microns_sessions", nargs='+', type=str,
                        default=['7_3', '9_3', '4_7', '9_6', '6_7', '6_6',
                                 '6_4', '8_5', '9_4', '7_4', '5_7', '7_5',
                                 '5_6', '6_2'])
    parser.add_argument("--l23", action='store_true',
                        help="Flag to use only L2/3 sessions.")
    parser.add_argument("--microns_l23_sessions", nargs='+', type=str,
                        default=['7_3', '4_7', '9_6', '6_7', '6_6', '6_4',
                                 '8_5', '9_4', '7_4', '5_7', '7_5', '5_6', '6_2'])
    parser.add_argument("--visual_areas", nargs='+', type=str,
                        default=['AL', 'LM', 'V1', 'RL'])
    parser.add_argument("--layer", type=str, default=None,
                        help="Specify cortical layer (e.g., 'L2/3').")
    parser.add_argument("--no_neuron_sampling", action='store_true',
                        help="If set, evaluate with all available neurons.")
    parser.add_argument("--selection_metric", type=str, default='corr_to_ave',
                        help="Metric for neuron selection.")
    parser.add_argument("--performance_threshold", type=float, default=0.15,
                        help="Initial performance threshold for neuron selection.")
    parser.add_argument("--num_sampling_neurons", nargs='+', type=int,
                        default=[1, 16, 32, 64, 128],
                        help="List of neuron counts for subsampling.")
    parser.add_argument("--sampling_trials", type=int, default=10,
                        help="Number of repetitions for each subsample size.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--fit_intercept", action='store_true', default=True)

    cli_args = parser.parse_args()
    main(cli_args)