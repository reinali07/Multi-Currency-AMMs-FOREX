import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

import seaborn as sns

import time

from cost_function import calculate_cost

#get the correlations matrix
corr = pd.read_csv('data/partition_data/quarterly_correlations.csv',index_col=0)

#and list of currencies
CCY_CTRY_CODES = pd.read_csv("data/raw_data/non_pegged.csv")
CURRENCIES = CCY_CTRY_CODES['currency'].to_list()

#convert list of lists into a series where the label is the group number
def get_group_labels(groups: list[list[str]]) -> pd.Series:
    """
    Converts a list-of-groups to a currency → integer label mapping.
    Currencies not in any group get label 44.
    """
    labels = {c: 44 for c in CURRENCIES}
    for i, group in enumerate(groups):
        for c in group:
            if c in labels:
                labels[c] = i
    return pd.Series(labels, name="group")

#get the cost function from the other file
cost_fn = calculate_cost()
def cost_function(groups):
    g = [group for group in groups if len(group) > 1]
    return cost_fn(g)

ratio_fn = calculate_cost(mode="ratio")

###################3

def find_optimal_correlation_blocks(
    corr_matrix,
    cost_fn,
    method="average",
    thresholds=None,
    use_abs=False,
    plot=True,
):
    if isinstance(corr_matrix, np.ndarray):
        #in case it's an ndarray instead of a df
        corr_matrix = pd.DataFrame(corr_matrix)

    #thresholds=None -> search all thresholds from 0 to 1 in increments of 0.01
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    start = time.time() #timer

    # Distance matrix
    if use_abs:
        dist = 1 - np.abs(corr_matrix) #this doesn't capture sign so we don't use this
    else:
        dist = np.sqrt(0.5*(1-corr_matrix.clip(-1,1))) #we use this which transforms to something like cosine similarity distance
    condensed_dist = squareform(dist.values, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed_dist, method=method) #average is avg distance btwn all pairs of currencies crossing the two clusters

    results = []

    best_cost = np.inf
    best_result = None

    for threshold in thresholds:

        labels = fcluster(
            Z,
            t= threshold,
            criterion="distance"
        )

        cluster_labels = pd.Series(
            labels,
            index=corr_matrix.index
        )

        # build list-of-lists clusters for use with cost func
        blocks = (
            cluster_labels
            .groupby(cluster_labels)
            .apply(lambda x: list(x.index))
            .tolist()
        )

        if len(blocks) == len(CURRENCIES): #this basically means everything is singleton
            continue

        # evaluate expected system cost
        cost = cost_fn(blocks)
        # print(blocks)
        # print(cost)

        results.append({
            "threshold": threshold,
            "cost": cost,
            "blocks": blocks,
            "labels": labels,
            'cluster_labels': cluster_labels,
        })

        #update best so far
        if cost < best_cost:
            best_cost = cost
            best_result = results[-1]

    best_labels = best_result["labels"]
    # best_labels_series = best_result['cluster_labels']
    
    end = time.time()

    print('time',end-start)

    # Reorder matrix for visualization
    pools = [g for g in best_result['blocks'] if len(g) >= 2]
    pools = [sorted(g) for g in pools]
    pools = sorted(pools)
    print(pools)

    best_labels_series = get_group_labels(pools)
    # print(best_labels_series)

    order = best_labels_series.sort_index().sort_values(kind='mergesort').index.tolist()
    # order = leaves_list(Z)

    clustered_corr = corr_matrix.loc[order, order]

    cluster_labels = pd.Series(
        best_labels,
        index=corr_matrix.index,
        name="cluster"
    )

    if plot:

        mask = np.eye(len(clustered_corr), dtype=bool)

        # Ignore diagonal when computing vmin and vmax
        nonzero_vals = clustered_corr.values[~mask]

        vmin = nonzero_vals.min()
        vmax = nonzero_vals.max()

        clustered_corr[mask] = vmax

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(
            clustered_corr,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={
                "label": "Pairwise probability"
            },
        )
        plt.tight_layout()
        plt.savefig("output/hac_clustering.png", dpi=300, bbox_inches="tight")

        plt.show()

    return {
        "clustered_corr": clustered_corr,
        "cluster_labels": cluster_labels,
        "blocks": best_result["blocks"],
        "best_threshold": best_result["threshold"],
        "best_cost": best_result["cost"],
        "results": pd.DataFrame(results),
    }

result = find_optimal_correlation_blocks(
    corr,
    cost_fn=cost_function
)

# print(result["blocks"])
print([block for block in result['blocks'] if len(block)>1])
print(result["best_threshold"])
print(result['best_cost'])



#vs. greedy algorithm:

import itertools as it

def form_groups():
    """
    Pair currencies minimizing:
       3-asset cost function
    """

    start = time.time()

    combos = list(it.chain(
        it.combinations(CURRENCIES, 2),
        it.combinations(CURRENCIES, 3),
        it.combinations(CURRENCIES, 4),
        it.combinations(CURRENCIES, 5),
        it.combinations(CURRENCIES, 6),
    ))
    triples = []
    for combo in combos:
        ratio = ratio_fn(list(combo)) #multi-asset pool cost / status quo cost

        triples.append([list(combo),ratio])

    end = time.time()

    print('elapsed time:',end-start)
    # print(sorted(triples, key=lambda x: x[1]))
    all_triples = sorted(triples, key=lambda x: x[1])
    greedy_triples = []
    selected_currencies = []
    while True:
        repeat = False
        for c in all_triples[0][0]:
            if c in selected_currencies:
                repeat = True
        if not repeat:
            greedy_triples.append(all_triples[0][0])
            selected_currencies.extend(all_triples[0][0])
        all_triples.pop(0)
        if len(all_triples) == 0:
            break
    return greedy_triples


brute_force = form_groups()
print(brute_force)
brute_force = ", ".join(
    "[" + ", ".join(group) + "]" for group in brute_force
)

print(brute_force)
