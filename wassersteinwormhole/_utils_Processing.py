import numpy as np
import scipy.spatial
from tqdm import tqdm
import numpy as np
import scipy.spatial
from tqdm import tqdm

def get_max_dist_statistic(point_clouds, weights, num_rand=100, dist_func_enc='S2', reduction='mean', sample_size=2048):
    """
    Calculates a statistic (mean or max) of the maximum distances between random pairs of point clouds.

    For performance, if a point cloud contains more points than `sample_size`, a random subset of
    `sample_size` points is used for the calculation.

    :param point_clouds: (list) A list of point cloud coordinate arrays.
    :param weights: (list) A list of corresponding weight arrays for each point cloud.
    :param num_rand: (int) The number of random pairs to sample.
    :param dist_func_enc: (str) The distance function encoding. 'S2', 'W2', 'GS', 'GW' use
                          squared Euclidean distance. Others use L1 (Cityblock) distance.
    :param reduction: (str) The reduction to apply to the list of max distances.
                      Can be either 'mean' or 'max'.
    :param sample_size: (int) The number of points to sample from larger point clouds.
    :return: (float) The calculated statistic (either the mean or the overall max) of the max distances.
    """
    # 1. Validate the reduction parameter
    if reduction not in ['mean', 'max']:
        raise ValueError("The `reduction` parameter must be either 'mean' or 'max'.")

    # 2. Set metric parameters based on the specified distance function
    if dist_func_enc in ["S2", "W2", "GS", "GW"]:
        metric_name = "squared Euclidean"
        cdist_metric = 'euclidean'
        is_squared = True
    else:  # Assumes L1 distance for other cases
        metric_name = "L1 (Cityblock)"
        cdist_metric = 'cityblock'
        is_squared = False

    max_distances = []
    
    # 3. Create a single, informative description for the progress bar
    progress_desc = f"Calculating {reduction} of max distances (metric: {metric_name})"

    # 4. Loop through random pairs with the improved progress bar
    for _ in tqdm(range(num_rand), desc=progress_desc):
        # Select two different point clouds at random
        i, j = np.random.choice(len(point_clouds), 2, replace=False)
        x, a = point_clouds[i], weights[i]
        y, b = point_clouds[j], weights[j]

        # --- OPTIMIZATION: Sample large point clouds to speed up calculation ---
        if len(x) > sample_size:
            indices_x = np.random.choice(len(x), sample_size, replace=False)
            x = x[indices_x]
            a = a[indices_x]
        
        if len(y) > sample_size:
            indices_y = np.random.choice(len(y), sample_size, replace=False)
            y = y[indices_y]
            b = b[indices_y]
        # --------------------------------------------------------------------

        # Calculate the distance matrix
        dist_matrix = scipy.spatial.distance.cdist(x, y, metric=cdist_metric)
        if is_squared:
            dist_matrix **= 2

        # Apply a boolean mask based on weights and find the maximum distance for the pair
        # This considers the distance only if both points have a weight > 0
        weighted_dist_matrix = dist_matrix * (np.outer(a, b) > 0)
        max_distances.append(np.max(weighted_dist_matrix))
    
    # 5. Apply the specified reduction and return the result
    if reduction == 'mean':
        return np.mean(max_distances)
    else: # reduction == 'max'
        return np.max(max_distances)

def MaxMinScale(arr):
    """
    :meta private:
    """

    arr = (
        2
        * (arr - arr.min(axis=0, keepdims=True))
        / (arr.max(axis=0, keepdims=True) - arr.min(axis=0, keepdims=True))
        - 1
    )
    return arr


def pad_pointclouds(point_clouds, weights, max_shape=-1):
    """
    :meta private:
    """

    if max_shape == -1:
        max_shape = np.max([pc.shape[0] for pc in point_clouds]) + 1
    else:
        max_shape = max_shape + 1


    weights_pad = np.asarray(
        [
            np.concatenate((weight, np.zeros(max_shape - pc.shape[0])), axis=0)
            for pc, weight in zip(point_clouds, weights)
        ]
    )
    point_clouds_pad = np.asarray(
        [
            np.concatenate(
                [pc, np.zeros([max_shape - pc.shape[0], pc.shape[-1]])], axis=0
            )
            for pc in point_clouds
        ]
    )

    weights_pad = weights_pad / weights_pad.sum(axis=1, keepdims=True)

    return (
        point_clouds_pad[:, :-1].astype("float32"),
        weights_pad[:, :-1].astype("float32"),
    )