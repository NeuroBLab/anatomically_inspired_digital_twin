import os
import pickle
import json
import numpy as np
from collections import defaultdict
from demo.figures.fig_params import MODEL_MAPPINGS

SKIP_SESSIONS = ['7_4', 'aggregated_across_sessions']  # problematic session
CORRECT_ORDERING_ARF = {'V1': 1, 'RL': 2, 'LM': 3, 'AL': 4}
CORRECT_ORDERING_HOELLER = {'V1': 1, 'RL': 3, 'LM': 2, 'AL': 4}
CORRECT_ORDERING_FROUDARAKIS = {'V1': 2, 'RL': 1, 'LM': 3, 'AL': 4}

# DATA LOADING

## SUMMARY
def load_summary(selected_names, metric="CCnorm", agg="median"):
    summary = {
        model_name: {} for model_name in MODEL_MAPPINGS.values() if model_name in selected_names
    }
    for model, model_name in MODEL_MAPPINGS.items():
        filename = "summary.pkl"
        if model == 'towards_wang':
            filename = 'summary_behavior.pkl'
        if model_name not in selected_names:
            continue
        else:
            print(f'Loading {model} summary...')
        path_summary = os.path.join('save', 'results', model, filename)
        with open(path_summary, 'rb') as f:
            data = pickle.load(f)

        for session in data['results']['val']:
            if session in SKIP_SESSIONS:  # Exclude this problematic session
                continue

            metric_data = data['results']['val'][session][metric]
            brain_area = data['model'][session]['brain_area']
            for ba in np.unique(brain_area):
                if ba not in summary[model_name]:
                    summary[model_name][ba] = []
                if agg == 'mean':
                    results = np.nanmean(metric_data[brain_area == ba])
                elif agg == 'median':
                    results = np.nanmedian(metric_data[brain_area == ba])
                summary[model_name][ba].append(results)
        del data
    return summary

## ARF
def load_arf(model_names):
    arf = {
        model_name: {} for model_name in MODEL_MAPPINGS.values() if model_name in model_names
    }
    for model, model_name in MODEL_MAPPINGS.items():
        if model_name not in model_names:
            continue
        path_arf = os.path.join('save', 'results', model, 'aRF.pkl')
        if not os.path.exists(path_arf):
            continue
        print(f'Loading {model} aRF...')
        try:
            with open(path_arf, 'rb') as f:
                data = pickle.load(f)
            arf[model_name][model] = data
            print('low_pass_arf' in data)
            print(data['low_pass_arf'].keys())
        except Exception as e:
            print(f"no data")
            continue
    return arf

## HOELLER
def load_hoeller(model_names):
    hoeller = {
        model_name: {} for model_name in MODEL_MAPPINGS.values() if model_name in model_names
    }
    for model, model_name in MODEL_MAPPINGS.items():
        if model_name not in model_names:
            continue
        path_hoeller = os.path.join('save', 'results', model, 'hoeller.json')
        with open(path_hoeller, 'rb') as f:
            data = json.load(f)
        hoeller[model_name][model] = data
    return hoeller

## FROUDARAKIS
def load_froudarakis(model_names):
    froudarakis = {
        model_name: {} for model_name in MODEL_MAPPINGS.values() if model_name in model_names
    }
    for model, model_name in MODEL_MAPPINGS.items():
        if model_name not in model_names:
            continue
        path_froudarakis = os.path.join('save', 'results', model, 'froudarakis.json')
        with open(path_froudarakis, 'rb') as f:
            data = json.load(f)
        froudarakis[model_name][model] = data
    return froudarakis


# ARF PROCESSING
from scipy.stats import kendalltau

def compute_gaussian_area(params_dict, k=1):
    out = {}
    for session, areas in params_dict.items():
        if session in SKIP_SESSIONS:
            continue
        out[session] = {}
        for ba, params in areas.items():
            session_areas = []
            for vals in params:
                sigma_x, sigma_y = vals[3], vals[4]
                area = np.pi * k**2 * sigma_x * sigma_y * 11.2896
                session_areas.append(area)
            out[session][ba] = np.array(session_areas)
    return out # session -> area -> area of gaussian

def compute_area_medians(area_data):
    """
    Input: {session: {area: [vals]}} — vals are area estimates per neuron.
    Output: {session: {area: median_value}}
    """
    medians = {}
    for session, areas in area_data.items():
        if session in SKIP_SESSIONS:
            continue
        medians[session] = {
            area: np.median([v for v in vals if v is not None])
            for area, vals in areas.items() if len(vals) > 0
        }
    return medians  # session -> area -> median

def evaluate_kendalls_tau_from_medians(median_data, correct_ordering):
    """
    Input:
      median_data: {session: {area: median_value}}
      correct_ordering: {area: rank}

    Output:
      {session: tau}
    """
    session_taus = {}
    for session, area_medians in median_data.items():
        if session in SKIP_SESSIONS:
            continue
        try:
            model_ranking = [area_medians[a] for a in correct_ordering]
            true_ranking = [correct_ordering[a] for a in correct_ordering]
            tau, _ = kendalltau(model_ranking, true_ranking)
            session_taus[session] = 0.0 if np.isnan(tau) else tau
        except Exception as e:
            print(f"Skipping session {session} due to error: {e}")
            session_taus[session] = np.nan
    return session_taus # session -> tau
  
def compute_global_kendalls_tau_mean_sem_from_sessions(session_taus):
    valid = [v for v in session_taus.values() if not np.isnan(v)]
    arr = np.array(valid)
    return np.mean(arr), np.std(arr, ddof=1) / np.sqrt(len(arr)) # mean, sem

def filter_neurons(arf, params, summary, threshold=0.3):
    filtered_arf = {}
    filtered_params = {}
    for session, value in arf.items():
        if session in ['4_7', '5_6', '6_4', '7_5']:
            filtered_arf[session] = {}
            filtered_params[session] = {}
            brain_areas = summary['model'][session]['brain_area']
            corr_to_ave = summary['results']['val'][session]['corr_to_ave']
            performance_idxs = np.where(corr_to_ave >= threshold)[0]
            for brain_area in np.unique(brain_areas):
                idxs = np.intersect1d(performance_idxs, np.where(brain_areas == brain_area)[0])
                if len(idxs) == 0:
                    filtered_arf[session][brain_area] = None
                    filtered_params[session][brain_area] = None
                else:
                    filtered_arf[session][brain_area] = [value[i] for i in idxs]
                    filtered_params[session][brain_area] = [params[session][i] for i in idxs]
    return filtered_arf, filtered_params

def filter_gaussian_parameters(arf, params, sigma_percentile=95, threshold_sigma_x=None, threshold_sigma_y=None): 
    """
    Filters 2D Gaussian fits with separate thresholds for σx and σy:
      - x0 ∈ [0, 64]
      - y0 ∈ [0, 36]
      - σx ≤ threshold_x (based on percentile)
      - σy ≤ threshold_y (based on percentile)
      - None entries are skipped and counted as removed

    Args:
        data: dict of shape {session: {area: [array or None, ...]}}
        sigma_percentile: percentile used for filtering σx and σy

    Returns:
        filtered_data: dict with the same structure
    """
    # Collect σx and σy separately
    sigmas_x = []
    sigmas_y = []
    for session in params.values():
        if session in SKIP_SESSIONS:
            continue
        for arrays in session.values():
            for arr in arrays:
                if arr is not None:
                    sigmas_x.append(arr[3])  # σx
                    sigmas_y.append(arr[4])  # σy

    if threshold_sigma_x is None:
        threshold_sigma_x = np.percentile(sigmas_x, sigma_percentile)
    if threshold_sigma_y is None:
        threshold_sigma_y = np.percentile(sigmas_y, sigma_percentile)

    filtered_arf = {}
    filtered_params = {}
    for session, areas in params.items():
        if session in SKIP_SESSIONS:
            continue
        filtered_arf[session] = {}
        filtered_params[session] = {}
        for area, arr_list in areas.items():
            filtered_arf[session][area] = []
            filtered = []
            removed_count = 0
            for i, arr in enumerate(arr_list):
                if arr is None:
                    removed_count += 1
                    continue
                x0, y0 = arr[1], arr[2]
                sigma_x, sigma_y = arr[3], arr[4]
                if (0 <= x0 <= 64) and (0 <= y0 <= 36) and \
                   (0 < sigma_x <= threshold_sigma_x) and (0 < sigma_y <= threshold_sigma_y):
                    filtered.append(arr)
                    filtered_arf[session][area].append(arf[session][area][i])
                else:
                    removed_count += 1
            filtered_params[session][area] = filtered

    return filtered_arf, filtered_params

def filter_arf(arf_data, summary, threshold=0.3, sigma_percentile=95, threshold_sigma_x=None, threshold_sigma_y=None):
    arf, params = arf_data['low_pass_arf'], arf_data['low_pass_params']
    filtered_arf, filtered_params = filter_neurons(
        arf, 
        params, 
        summary, 
        threshold
    )
    filtered_arf, filtered_params = filter_gaussian_parameters(
        filtered_arf, 
        filtered_params, 
        sigma_percentile, 
        threshold_sigma_x, 
        threshold_sigma_y
    )
    return filtered_arf, filtered_params

def evaluate_models_receptive_fields(
    model_names, 
    correct_ordering,
    threshold_x=None, 
    threshold_y=None,
    return_mean=True
):
    output = {}
    model_means = []
    model_sems = []
    model_labels = []
    for model in model_names:
        print(f'Processing {model}...')
        path_summary = os.path.join('save', 'results', model, 'summary.pkl')
        path_arf = os.path.join('save', 'results', model, 'aRF.pkl')

        with open(path_summary, 'rb') as f:
            summary = pickle.load(f)
        with open(path_arf, 'rb') as f:
            arf_data = pickle.load(f)

        _, low_pass_params_filtered = filter_arf(
            arf_data,
            summary,
            threshold=0.35,
            sigma_percentile=95,
            threshold_sigma_x=threshold_x,
            threshold_sigma_y=threshold_y
        )

        gaussian_areas = compute_gaussian_area(low_pass_params_filtered)
        median_data = compute_area_medians(gaussian_areas)
        session_taus = evaluate_kendalls_tau_from_medians(median_data, correct_ordering)
        if return_mean:
            mean_tau, sem_tau = compute_global_kendalls_tau_mean_sem_from_sessions(session_taus)
        else:
            mean_tau = [v for v in session_taus.values() if not np.isnan(v)]
            mean_tau = np.array(mean_tau)
            sem_tau = mean_tau
        model_labels.append(model)
        model_means.append(mean_tau)
        model_sems.append(sem_tau)

        # Compute median for each area across sessions
        
        area_medians = {}
        for session, areas in median_data.items():
            if session in SKIP_SESSIONS:
                continue
            for area, median_value in areas.items():
                if area not in area_medians:
                    area_medians[area] = []
                area_medians[area].append(median_value)
        output[model] = area_medians
        del summary
    return output, model_labels, model_means, model_sems  # {model: {area: [values]}}}

# HOELLER PROCESSING
def gather_area_mean_accuracy_hoeller(data):
    """
    Converts nested {model: {seed: {session: {area: {aggregate_mean_accuracy}}}}}
    into {model: {area: [mean_accuracy_values]}}, aggregating over seeds and sessions.

    Returns:
        dict[model][area] -> list of float
    """
    output = defaultdict(lambda: defaultdict(list))  # model -> area -> [mean values]

    for model, seed_data in data.items():
        for seed, sessions in seed_data.items():
            for session, areas in sessions.items():
                if session in SKIP_SESSIONS:
                    continue
                for area, metrics in areas.items():
                    mean = metrics['aggregate_mean_accuracy']
                    output[model][area].append(mean)

    return {model: dict(area_dict) for model, area_dict in output.items()}

def rank_areas_hoeller(data, correct_ordering, use_sem=True):
    seed_session_rankings = {}
    seed_session_taus = {}

    for seed, sessions in data.items():
        session_rankings = {}
        session_taus = {}

        for session, areas in sessions.items():
            if session in SKIP_SESSIONS:
                continue
            stats = []
            for area, metrics in areas.items():
                mean = metrics['aggregate_mean_accuracy']
                sem = metrics['aggregate_sem_accuracy']
                stats.append((area, mean, sem))

            stats.sort(key=lambda x: x[1])
            area_ranks = {}
            rank = 1
            for i, (area, mean, sem) in enumerate(stats):
                if i == 0:
                    area_ranks[area] = rank
                    continue
                prev_mean, prev_sem = stats[i - 1][1], stats[i - 1][2]
                overlap = (mean + sem) >= (prev_mean - prev_sem) if use_sem else mean == prev_mean
                if overlap:
                    area_ranks[area] = rank
                else:
                    rank += 1
                    area_ranks[area] = rank

            session_rankings[session] = area_ranks
            common_areas = list(correct_ordering.keys())
            model_ranking = [area_ranks[a] for a in common_areas]
            true_ranking = [correct_ordering[a] for a in common_areas]
            tau, _ = kendalltau(model_ranking, true_ranking)
            session_taus[session] = 0.0 if np.isnan(tau) else tau

        seed_session_rankings[seed] = session_rankings
        seed_session_taus[seed] = session_taus

    return seed_session_rankings, seed_session_taus

def compute_global_kendalls_tau_mean_sem(taus):
    all_taus = []
    for seed_taus in taus.values():
        for tau in seed_taus.values():
            if not np.isnan(tau):
                all_taus.append(tau)
    all_taus = np.array(all_taus)
    mean_tau = np.mean(all_taus)
    sem_tau = np.std(all_taus, ddof=1) / np.sqrt(len(all_taus))
    return mean_tau, sem_tau

def compute_hoeller_kendalls(
    data, 
    correct_ordering, 
    use_sem=False,
    return_mean=True
):
    x_labels = []
    y_values = []
    y_sems = []
    for model_name, model_data in data.items():
        x_labels.append(model_name)
        _, seed_session_taus = rank_areas_hoeller(
            model_data,
            correct_ordering,
            use_sem=use_sem
        )
        if return_mean:
            mean_tau, sem_tau = compute_global_kendalls_tau_mean_sem(seed_session_taus)
        else:
            mean_tau = []
            for seed_taus in seed_session_taus.values():
                for tau in seed_taus.values():
                    if not np.isnan(tau):
                        mean_tau.append(tau)
            mean_tau = np.array(mean_tau)
            sem_tau = mean_tau
        y_values.append(mean_tau)
        y_sems.append(sem_tau)
    return x_labels, y_values, y_sems

# FROUDARAKIS PROCESSING
def gather_area_metric_values(data, metric='test_mi'):
    """
    Converts nested {seed: {session: {area: [dicts]}}} structure to
    {model: {area: [metric_values]}}, aggregating over seeds and sessions.

    Args:
        data: dict[seed][session][area] = {'128': list[dict with metric]}
        metric: metric key to extract from each sample dict

    Returns:
        dict[model][area] -> list of float (len = seeds * sessions)
    """

    output = defaultdict(lambda: defaultdict(list))  # model -> area -> [values]
    for model, seed_data in data.items():
        for seed, sessions in seed_data.items():
            for session, areas in sessions.items():
                if session in SKIP_SESSIONS:
                    continue
                for area, sample_data in areas.items():
                    sample_dicts = sample_data['128']
                    values = np.mean([d[metric] for d in sample_dicts])
                    output[model][area].append(values)

    return {model: dict(area_dict) for model, area_dict in output.items()}

def rank_areas_froudarakis(data, correct_ordering, metric='test_mi', use_sem=False):
    seed_session_rankings = {}
    seed_session_taus = {}

    for seed, sessions in data.items():
        session_rankings = {}
        session_taus = {}

        for session, areas in sessions.items():
            if session in SKIP_SESSIONS:
                continue
            stats = []
            for area, samples in areas.items():
                sample_dicts = samples['128']
                values = [d[metric] for d in sample_dicts]
                mean = np.mean(values)
                sem = np.std(values, ddof=1) / np.sqrt(len(values))
                stats.append((area, mean, sem))

            # Sort descending by mean (higher metric is better)
            stats.sort(key=lambda x: x[1])

            area_ranks = {}
            rank = 1
            for i, (area, mean, sem) in enumerate(stats):
                if i == 0:
                    area_ranks[area] = rank
                else:
                    prev_mean, prev_sem = stats[i - 1][1], stats[i - 1][2]
                    overlap = (mean + sem) >= (prev_mean - prev_sem) if use_sem else mean == prev_mean
                    if overlap:
                        area_ranks[area] = rank
                    else:
                        rank += 1
                        area_ranks[area] = rank

            session_rankings[session] = area_ranks

            model_ranks = [area_ranks[a] for a in correct_ordering]
            true_ranks = [correct_ordering[a] for a in correct_ordering]
            tau, _ = kendalltau(model_ranks, true_ranks)
            session_taus[session] = 0.0 if np.isnan(tau) else tau

        seed_session_rankings[seed] = session_rankings
        seed_session_taus[seed] = session_taus

    return seed_session_rankings, seed_session_taus # {seed: {session: {area: rank}}}, {seed: {session: tau}}

def compute_froudarakis_kendalls(
    data, 
    correct_ordering, 
    metric='test_mi', 
    use_sem=False,
    return_mean=True
):
    x_labels = []
    y_values = []
    y_sems = []
    for model_name, model_data in data.items():
        x_labels.append(model_name)
        _, seed_session_taus = rank_areas_froudarakis(
            model_data,
            correct_ordering,
            metric=metric,
            use_sem=use_sem
        )
        if return_mean:
            mean_tau, sem_tau = compute_global_kendalls_tau_mean_sem(seed_session_taus)
        else:
            mean_tau = []
            for seed_taus in seed_session_taus.values():
                for tau in seed_taus.values():
                    if not np.isnan(tau):
                        mean_tau.append(tau)
            mean_tau = np.array(mean_tau)
            sem_tau = mean_tau
        y_values.append(mean_tau)
        y_sems.append(sem_tau)
    return x_labels, y_values, y_sems

# MODEL RANKINGS
def evaluate_model_name_vs_true_ranking(model_rankings, true_ordering):
    result = {}
    for name, model_ordering in model_rankings.items():
        areas = list(true_ordering.keys())
        model_ranks = [model_ordering[a] for a in areas]
        true_ranks = [true_ordering[a] for a in areas]

        tau, _ = kendalltau(model_ranks, true_ranks)
        tau = 0.0 if np.isnan(tau) else tau
        result[name] = tau
    return result  # {clean_model_name: tau}
