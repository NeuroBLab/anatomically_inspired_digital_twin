"""
Performs classification analysis on simulated neural responses to rotated images,
replicating the methodology of Hoeller et al. (2024).

This script loads simulated neural data and model metadata, filters neurons
based on brain area and performance, and trains a linear SVM to decode image
identity. It uses a custom cross-validation strategy based on holding out
consecutive image rotations. The analysis is run across multiple sessions and
random seeds, with results aggregated and saved to a JSON file.
"""

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


def filter_and_select_neurons(data, model_data, area, performance_threshold,
                              sessions, metric='corr_to_ave', layer=None):
    """
    Filters neurons by area and performance for a given set of sessions.

    Args:
        data (dict): Simulated neural responses, keyed by image class.
        model_data (dict): Model metadata with neuron info (area, performance).
        area (str): The visual area to select neurons from (e.g., 'V1').
        performance_threshold (float): Minimum performance for neuron inclusion.
        sessions (list): List of session IDs to process.
        metric (str): The performance metric used for filtering.
        layer (str, optional): Cortical layer to select neurons from.

    Returns:
        dict: A dictionary mapping image class to a 2D NumPy array of
            filtered neural responses (time x neurons), concatenated across
            the specified sessions.
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
        valid_mask = performances >= performance_threshold
        selected_neurons[session] = area_indices[valid_mask]

    filtered_data = {}
    for obj_class in data:
        all_sessions_data = []
        for session in data[obj_class]:
            if session in selected_neurons and len(selected_neurons[session]) > 0:
                session_data = data[obj_class][session][:, selected_neurons[session]]
                all_sessions_data.append(session_data)

        if all_sessions_data:
            # Concatenate neurons from all specified sessions.
            filtered_data[obj_class] = np.concatenate(all_sessions_data, axis=1)
        else:
            filtered_data[obj_class] = np.empty((0, 0))

    return filtered_data


def create_dataframe_from_dict(data_dict):
    """
    Converts a dictionary of neural data into a Pandas DataFrame.

    Args:
        data_dict (dict): Maps class labels to 2D NumPy arrays of features.

    Returns:
        pd.DataFrame: A DataFrame with features and a 'label' column.
    """
    if not data_dict or all(v.size == 0 for v in data_dict.values()):
        return pd.DataFrame({'label': []})

    features_list = [features for features in data_dict.values()]
    labels_list = [
        np.full(features.shape[0], fill_value=label)
        for label, features in data_dict.items()
    ]

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    df = pd.DataFrame(features)
    df['label'] = labels
    return df


def run_classification_analysis(datasets, remove_gray=True, random_seed=0,
                                use_pca=True, test_all_held_out=False, n_jobs=-1):
    """
    Performs the core image classification analysis from Hoeller et al. (2024).

    This function implements a custom cross-validation scheme where image
    rotations are held out for testing. It preprocesses the data, optionally
    applies PCA, and trains a linear SVM classifier.

    Args:
        datasets (dict): Maps area names to DataFrames of neural responses.
        remove_gray (bool): If True, subtracts the mean response to a gray
            screen as a baseline correction.
        random_seed (int): Seed for reproducibility.
        use_pca (bool): If True, projects data onto 100 principal components
            before classification.
        test_all_held_out (bool): If True, iterates through all classes as the
            test set within each CV fold. If False, uses a fixed random pair
            of classes for testing.
        n_jobs (int): Number of parallel jobs for cross-validation.

    Returns:
        dict: A dictionary mapping each area to its mean accuracy, SEM,
            and the list of classes used for testing.
    """
    np.random.seed(random_seed)
    results = {}

    # Use the first area's data to define a common set of image classes.
    first_area = list(datasets.keys())[0]
    ref_df = datasets[first_area]
    ref_features = ref_df.iloc[:, :-1].values
    ref_labels = ref_df.iloc[:, -1].values

    if remove_gray:
        gray_mask = (ref_labels == 'gray')
        if np.any(gray_mask):
            gray_mean = ref_features[gray_mask].mean(axis=0)
            ref_features -= gray_mean

    valid_mask = (ref_labels != 'gray')
    unique_classes = np.unique(ref_labels[valid_mask])
    if len(unique_classes) < 3:
        raise ValueError("Not enough unique classes for train/test split.")

    # Define a fixed train/test split of classes for one of the analysis modes.
    test_classes = np.random.choice(unique_classes, size=2, replace=False)
    train_classes = np.setdiff1d(unique_classes, test_classes)

    # Define a custom cross-validation generator based on holding out rotations.
    n_rotations = 15
    cv_splits = []
    for i in range(n_rotations):
        held_out = [(i + j) % n_rotations for j in range(3)]
        test_rot_idx = held_out[1]
        cv_splits.append((held_out, test_rot_idx))

    for area, df in datasets.items():
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values

        if remove_gray:
            gray_mask = (labels == 'gray')
            if np.any(gray_mask):
                gray_mean = features[gray_mask].mean(axis=0)
                features -= gray_mean

        valid_mask = (labels != 'gray')
        features_valid = features[valid_mask]
        labels_valid = labels[valid_mask]

        # Group data by class, assuming each class has `n_rotations` samples.
        data_by_class = {
            label: features_valid[labels_valid == label]
            for label in unique_classes
        }

        def process_cv_split(held_out_indices, test_rotation_index):
            """Processes a single fold of the cross-validation."""
            if not test_all_held_out:
                # --- Mode 1: Fixed test classes ---
                train_indices = [i for i in range(n_rotations) if i not in held_out_indices]
                X_train = np.vstack([data[train_indices] for data in data_by_class.values()])
                y_train = np.array([lbl for lbl in data_by_class for _ in train_indices])

                X_test = np.vstack([data_by_class[lbl][[test_rotation_index]] for lbl in test_classes])
                y_test = test_classes

                # Normalize and mean-center based on the full training set.
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                if use_pca:
                    pca = PCA(n_components=100).fit(X_train)
                    X_train = pca.transform(X_train)
                    X_test = pca.transform(X_test)

                clf = LinearSVC(C=1.0, max_iter=10000, random_state=random_seed)
                clf.fit(X_train, y_train)
                return accuracy_score(y_test, clf.predict(X_test))

            else:
                # --- Mode 2: Iterate through all classes as test sets ---
                accuracies = []
                for held_out_class in unique_classes:
                    train_indices = [i for i in range(n_rotations) if i not in held_out_indices]
                    current_train_classes = [c for c in unique_classes if c != held_out_class]

                    X_train = np.vstack([data_by_class[lbl][train_indices] for lbl in current_train_classes])
                    y_train = np.array([lbl for lbl in current_train_classes for _ in train_indices])

                    X_test = data_by_class[held_out_class][[test_rotation_index]]
                    y_test = np.array([held_out_class])

                    scaler = StandardScaler().fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)

                    if use_pca:
                        pca = PCA(n_components=100).fit(X_train)
                        X_train = pca.transform(X_train)
                        X_test = pca.transform(X_test)

                    clf = LinearSVC(C=1.0, max_iter=10000, random_state=random_seed)
                    clf.fit(X_train, y_train)
                    accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
                return np.mean(accuracies)

        # Run cross-validation in parallel.
        cv_accuracies = Parallel(n_jobs=n_jobs)(
            delayed(process_cv_split)(held_out, test_rot)
            for held_out, test_rot in cv_splits
        )
        cv_accuracies = [acc for acc in cv_accuracies if acc is not None]

        mean_acc = np.mean(cv_accuracies)
        sem_acc = np.std(cv_accuracies, ddof=1) / np.sqrt(len(cv_accuracies))
        results[area] = {
            'mean_accuracy': mean_acc,
            'sem_accuracy': sem_acc,
            'test_classes': test_classes.tolist() if not test_all_held_out else 'all'
        }
    return results


def aggregate_over_seeds(datasets, n_iterations=10, **kwargs):
    """
    Runs the classification analysis multiple times with different seeds.

    Args:
        datasets (dict): Maps area names to DataFrames of neural responses.
        n_iterations (int): Number of random seeds to run.
        **kwargs: Arguments passed to `run_classification_analysis`.

    Returns:
        dict: Aggregated results with mean, std, and SEM of accuracy
            across iterations for each area.
    """
    all_results = {area: [] for area in datasets}
    base_seed = kwargs.get("random_seed", 42)

    for i in tqdm(range(n_iterations), desc="Aggregating over seeds"):
        kwargs["random_seed"] = base_seed + i
        seed_results = run_classification_analysis(datasets, **kwargs)
        for area, result in seed_results.items():
            all_results[area].append(result['mean_accuracy'])

    aggregated_results = {}
    for area, accuracies in all_results.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        sem_acc = std_acc / np.sqrt(n_iterations)
        aggregated_results[area] = {
            'aggregate_mean_accuracy': float(mean_acc),
            'aggregate_std_accuracy': float(std_acc),
            'aggregate_sem_accuracy': float(sem_acc),
            'all_seed_results': [float(a) for a in accuracies]
        }
    return aggregated_results


def main(args):
    """
    Main function to orchestrate the multi-session analysis pipeline.
    """
    path_model = os.path.join(args.base_path, f"save/results/{args.model}/summary.pkl")
    with open(path_model, 'rb') as f:
        model_data = pickle.load(f)

    path_input = os.path.join(args.base_path, f"save/data/hoeller/{args.model}_{args.input_data}.pkl")
    with open(path_input, 'rb') as f:
        data = pickle.load(f)

    session_results = {}
    for session_id in tqdm(args.microns_sessions, desc="Processing sessions"):
        datasets = {}
        for area in args.visual_areas:
            filtered_data = filter_and_select_neurons(
                data, model_data, area,
                performance_threshold=args.corr_threshold,
                sessions=[session_id],
                metric='corr_to_ave'
            )
            datasets[area] = create_dataframe_from_dict(filtered_data)

        # Ensure all areas have enough neurons before proceeding.
        if any(df.shape[1] < 2 for df in datasets.values()):
            print(f"Skipping session {session_id}: insufficient neurons in at least one area.")
            continue

        agg_res = aggregate_over_seeds(
            datasets,
            n_iterations=args.n_iterations,
            remove_gray=args.remove_gray,
            random_seed=args.random_seed,
            test_all_held_out=args.test_all_held_out,
            use_pca=args.use_pca,
            n_jobs=args.n_jobs
        )
        session_results[session_id] = agg_res

    # --- Save final aggregated results ---
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w") as f:
            json.dump(session_results, f, indent=4)
        print(f"\nMulti-session results saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-session classification analysis for Hoeller et al. (2024) replication."
    )
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--microns_sessions", nargs='+', type=str, required=True)
    parser.add_argument("--corr_threshold", type=float, default=0.35)
    parser.add_argument("--visual_areas", nargs='+', default=['AL', 'LM', 'V1', 'RL'])
    parser.add_argument("--input_data", type=str, default="object_rotations")
    parser.add_argument("--n_iterations", type=int, default=50,
                        help="Number of random seeds for aggregation.")
    parser.add_argument("--remove_gray", action="store_true")
    parser.add_argument("--random_seed", type=int, default=1088)
    parser.add_argument("--test_all_held_out", action="store_true",
                        help="Use all-class hold-out testing instead of fixed test classes.")
    parser.add_argument("--no_pca", dest='use_pca', action="store_false")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the final JSON results.")
    cli_args = parser.parse_args()
    main(cli_args)