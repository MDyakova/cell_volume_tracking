""""
Functions for segmentation.py code
"""

from collections import defaultdict
from itertools import combinations
import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from PIL import Image
import tifffile as tiff
from tqdm import tqdm
from scipy import stats
from aicsimageio import AICSImage
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
from skimage.measure import block_reduce


# Function to join mask pairs
def join_pairs(pairs):
    """
    Function to find intersected labels
    """
    # Create a dictionary to map each element to its group
    groups = defaultdict(set)

    # Iterate over pairs and add them to the corresponding group
    for pair in pairs:
        a, b = pair
        # Merge the groups of a and b
        group_a = groups[a]
        group_b = groups[b]
        combined_group = group_a | group_b | {a, b}
        # Update both a and b's groups with the combined group
        for elem in combined_group:
            groups[elem] = combined_group

    # Remove duplicates and return the result
    unique_groups = set(frozenset(group) for group in groups.values())
    return [list(group) for group in unique_groups]


def find_similar_contours_fast(image_for_masks):
    """
    Function to join intersected labels
    """
    mask_id, numbers = np.unique(image_for_masks.reshape(-1), return_counts=True)
    mask_numbers = dict(zip(mask_id, numbers))
    groups_short = [
        image_for_masks[i, j, :]
        for i in range(image_for_masks.shape[0])
        for j in range(image_for_masks.shape[1])
    ]
    groups_short = [list(group[group > 0]) for group in groups_short]
    groups_short = [group for group in groups_short if len(group) > 1]

    # Dictionary to store pair counts
    pair_counts = defaultdict(int)
    # Loop through each sublist and generate pairs
    for sublist in groups_short:
        for pair in combinations(
            sorted(sublist), 2
        ):  # Generate sorted pairs to avoid (1, 2) and (2, 1) being counted separately
            pair_counts[pair] += 1
    # Convert the defaultdict to a regular dictionary for better readability (optional)
    pair_counts = dict(pair_counts)

    pairs = []
    for key, value in pair_counts.items():
        value_0 = mask_numbers[key[0]]
        value_1 = mask_numbers[key[1]]
        k = value / min(value_0, value_1)
        if k > 0.6:
            pairs.append(key)

    merged_pairs = join_pairs(pairs)

    for group in merged_pairs:
        main_id = group[0]
        for similar_id in group[1:]:
            image_for_masks[image_for_masks == similar_id] = main_id
    image_for_masks_max = image_for_masks.max(axis=2)
    return image_for_masks_max


def iou_overlap_matrix(lab0: np.ndarray, lab1: np.ndarray):
    """
    Compute overlapping of masks for each frame
    """
    if lab0.shape != lab1.shape:
        raise ValueError("lab0 and lab1 must have the same shape")

    labels0 = np.unique(lab0)
    labels0 = labels0[labels0 != 0]
    labels1 = np.unique(lab1)
    labels1 = labels1[labels1 != 0]

    n0, n1 = len(labels0), len(labels1)
    if n0 == 0 or n1 == 0:
        return np.zeros((n0, n1), dtype=np.float32), labels0, labels1

    map0 = np.zeros(lab0.max() + 1, dtype=np.int32)
    map1 = np.zeros(lab1.max() + 1, dtype=np.int32)
    map0[labels0] = np.arange(1, n0 + 1)
    map1[labels1] = np.arange(1, n1 + 1)

    a = map0[lab0]  # 0..n0
    b = map1[lab1]  # 0..n1

    idx = a.ravel() * (n1 + 1) + b.ravel()
    inter = np.bincount(idx, minlength=(n0 + 1) * (n1 + 1)).reshape((n0 + 1, n1 + 1))
    inter = inter[1:, 1:]  # (n0, n1)

    area0 = np.bincount(a.ravel(), minlength=n0 + 1)[1:]
    area1 = np.bincount(b.ravel(), minlength=n1 + 1)[1:]

    union = area0[:, None] + area1[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou, labels0, labels1


def track_masks_iou(masks, iou_thr=0.1, memory=0):
    """
    Function tracks segmented masks
    and return list of dicts with frame, label, track_id
    """
    assignments = []
    records = []

    next_track_id = 1

    active = {}

    # frame 0: initialize tracks
    lab0 = masks[0]
    labels0 = np.unique(lab0)
    labels0 = labels0[labels0 != 0]
    map0 = {}
    for lbl in labels0:
        tid = next_track_id
        next_track_id += 1
        map0[int(lbl)] = tid
        active[tid] = {"frame": 0, "label": int(lbl)}
        records.append({"frame": 0, "label": int(lbl), "track_id": tid})
    assignments.append(map0)

    # subsequent frames
    for t in range(1, len(masks)):
        prev = masks[t - 1]
        cur = masks[t]

        # drop expired tracks (not seen for > memory frames)
        expired = [tid for tid, st in active.items() if (t - st["frame"] - 1) > memory]
        for tid in expired:
            del active[tid]

        # labels in prev and cur
        iou, prev_labels, cur_labels = iou_overlap_matrix(prev, cur)

        # Hungarian on -IoU
        if iou.size > 0:
            r, c = linear_sum_assignment(-iou)
        else:
            r, c = np.array([], dtype=int), np.array([], dtype=int)

        # build mapping for this frame
        cur_map = {}

        # which current labels already assigned
        used_cur = set()

        # link matched pairs (prev_label -> cur_label) if IoU passes threshold
        for i, j in zip(r, c):
            if iou[i, j] < iou_thr:
                continue
            pl = int(prev_labels[i])
            cl = int(cur_labels[j])

            # find track_id of prev label from previous assignments
            prev_tid = assignments[-1].get(pl, None)
            if prev_tid is None:
                continue

            cur_map[cl] = prev_tid
            used_cur.add(cl)
            active[prev_tid] = {"frame": t, "label": cl}
            records.append({"frame": t, "label": cl, "track_id": prev_tid})

        # create new tracks for unmatched current labels
        for cl in cur_labels:
            cl = int(cl)
            if cl in used_cur:
                continue
            tid = next_track_id
            next_track_id += 1
            cur_map[cl] = tid
            active[tid] = {"frame": t, "label": cl}
            records.append({"frame": t, "label": cl, "track_id": tid})

        assignments.append(cur_map)

    return assignments, records


def make_segmentation(
    model,
    image,
    output_directory,
    file_name,
    file_name_save,
    processed_files,
    tile_size,
    background_threshold,
    cell_diameter_min,
    cell_diameter_max,
    i=0,
    j=0,
):
    """
    Segment and track cells
    """

    results = []
    if tile_size == -1:
        tile_size = np.max(image.shape)
    if file_name not in processed_files:
        all_masks_save = []
        # Segmentation for each layer in a XYZ tile
        for step in tqdm(range(image.shape[0])):
            image[step] = np.where(image[step] > background_threshold, image[step], 0)
            tile_crops = []
            # Make segmentation for each tile for rotation on 0, 90, 180 and 270 degrees
            t = 0
            for k in [0]:
                for diameter in range(cell_diameter_min, cell_diameter_max, 5):
                    # input image for model
                    image_save = Image.fromarray(
                        np.rot90(
                            (image[step] / image[step].max() * 255),
                            k=k,
                        ).astype("uint8")
                    )

                    # Make prediction
                    masks_pred, _, _ = model.eval(
                        [np.array(image_save)],
                        niter=1000,
                        cellprob_threshold=0,
                        diameter=diameter,
                    )

                    # Back rotation
                    pred = np.rot90(masks_pred[0], k=-1 * k)
                    pred = np.where(pred > 0, pred + 1000 * t, 0)
                    tile_crops.append(pred)
                    t += 1

            # Join segmentation for all rotated files
            tile_crops_all = np.stack(tile_crops, axis=2)
            all_masks = find_similar_contours_fast(tile_crops_all)
            all_masks_save.append(all_masks)

        # Combine all frames
        if np.max(all_masks_save[0] > 0) > 0:
            # Tracking
            _, records = track_masks_iou(
                all_masks_save, iou_thr=0.2, memory=2
            )

            # Change lable id to track id
            tracking = np.zeros(
                (
                    len(all_masks_save),
                    all_masks_save[0].shape[0],
                    all_masks_save[0].shape[1],
                )
            )
            for record in records:
                frame = record["frame"]
                label = record["label"]
                track_id = record["track_id"]
                tracking[frame] = np.where(
                    all_masks_save[frame] == label, track_id, tracking[frame]
                )

            # Filter incomplite tracks
            z, y, x = np.nonzero(tracking)
            tracking_df = tracking[z, y, x]
            tracking_df = pd.DataFrame({"label": tracking_df, "z": z, "y": y, "x": x})
            tracking_df = tracking_df.groupby(by=["label", "z"], as_index=False).max()
            tracking_df = tracking_df.groupby(by=["label"], as_index=False).count()
            untracked_labels = list(tracking_df[tracking_df["z"] < len(image)]["label"])

            mask_found = np.isin(tracking, untracked_labels)
            tracking[mask_found] = 0

            # Save results
            tiff.imwrite(
                os.path.join(
                    output_directory,
                    f"{file_name}_labels.tif",
                ),
                tracking.astype(np.uint16),
            )
            if tile_size < np.max(image.shape):
                video = image[:, i : i + tile_size, j : j + tile_size]
                tiff.imwrite(
                    os.path.join(
                        output_directory,
                        f"{file_name}_image.tif",
                    ),
                    video.astype(np.uint16),
                )
            # Compute integrated density
            for label_id in np.unique(tracking):
                if label_id > 0:
                    for step in range(image.shape[0]):
                        integrated_density = np.sum(
                            image[step][np.where(tracking[step] == label_id)]
                        )
                        cell_area = len(np.where(tracking[step] == label_id)[0])
                        y, x = np.where(tracking[step] == label_id)
                        results.append(
                            [
                                file_name_save,
                                label_id,
                                step,
                                integrated_density,
                                cell_area,
                                x.min(),
                                x.max(),
                                y.min(),
                                y.max(),
                                image.shape[1],
                                image.shape[2],
                            ]
                        )

    return results

def make_3d_segmentation(
                image_directory,
                output_directory,
                folder,
                resize_factor,
                cell_diameter,
                params_dict
            ):

    # Load parameters
    dx = params_dict['dx']
    dy = params_dict['dy']
    dz = params_dict['dz']
    edge_pos = params_dict['edge_pos']
    iou_thr = params_dict['iou_thr']
    memory = params_dict['memory']
    min_size = params_dict['min_size']
    max_size = params_dict['max_size']
    min_value = params_dict['min_value']
    max_value = params_dict['max_value']
    # w_3d = params_dict['w_3d']
    # w_size = params_dict['w_size']

    # New voxel sizes
    dx_new = dx * resize_factor
    dy_new = dy * resize_factor
    dz_new = dz

    anisotropy = dx_new/dz_new

    output_folder = os.path.join(output_directory, folder)
    os.makedirs(output_folder, exist_ok=True)

    # Cell segmentation
    all_files = os.listdir(os.path.join(image_directory, folder))
    all_files = list(filter(lambda p: ('nd2' in p) & ('Zone' not in p), all_files))
    all_files = sorted(
        all_files,
        key=lambda x: int(re.search(r'(\d+)(?=\.nd2$)', x).group())
    )
    for file_name in all_files:
        # Convert image to tif
        image = nd2_to_tiff(os.path.join(image_directory, folder, file_name))
        # Name to save files
        file_name_save = file_name.split(".nd2")[0]
        print(folder, file_name_save)

        # arr_resized = resize(
        #     image, 
        #     (image.shape[0], image.shape[1]//resize_factor, image.shape[2]//resize_factor), 
        #     order=1,           # bilinear
        #     preserve_range=True,
        #     anti_aliasing=True
        # ).astype(image.dtype)

        arr_resized = block_reduce(
            image,
            block_size=(1, resize_factor, resize_factor),
            func=np.mean
        )

        # print(arr_resized.shape, arr_sum.shape)

        # remove signal outliers
        if max_value is not None:
            arr_resized = np.where(arr_resized<=max_value, arr_resized, max_value)
        arr_resized = np.where(arr_resized>=min_value, arr_resized, 0)

        crop_path = os.path.join(output_folder, f"{file_name_save}_res.tif")
        tiff.imwrite(crop_path, arr_resized.astype(np.uint16))

        cmd = [
            "python3.10", "-m", "cellpose",
            "--image_path", crop_path,
            "--do_3D",
            "--save_tif",
            "--use_gpu",
            "--diameter", str(cell_diameter),
            # "--anisotropy", "0.75"
        ]
           
        subprocess.run(cmd, check=True)

    # Find all labels and clean
    all_images = []
    all_labels = []
    all_labels_2d = []
    all_names = []
    all_df = []
    for file_name in all_files:
        file_name_save = file_name.split(".nd2")[0]
        image_path = os.path.join(output_folder, f"{file_name_save}_res.tif")
        labels_path = os.path.join(output_folder, f"{file_name_save}_res_cp_masks.tif")

        image = tiff.imread(image_path)
        labels = tiff.imread(labels_path)
        # remove outliers
        labels = remove_outliers(labels, min_size=min_size, max_size=max_size)
        # compute label sizes
        df = compute_sizes(image, labels, dx_new, dy_new, dz, resize_factor)

        # remove z outliers
        labels, df = remove_outliers_pos(labels, df, edge_pos=edge_pos)

        # make 2D label images
        labels_2d = make_2d_labels(labels)

        # save to lists
        all_labels.append(labels)
        all_labels_2d.append(np.array(labels_2d))
        print(file_name, labels.shape)
        all_names.append(file_name)
        all_df.append(df)
        # all_images.append(image)

    # # 3D tracking
    # assignments, records = track_masks_3d_projection(
    #     all_labels,
    #     score_thr=score_thr,
    #     memory=memory,
    #     max_xy_dist=max_xy_dist,
    #     w_xy=w_xy,
    #     w_3d=w_3d,
    #     w_size=w_size,
    # )
    # 3D tracking
    aligned_labels_2d, shifts = align_all_labels_2d(all_labels_2d, reference_idx=0)
    all_labels_2d = np.stack(aligned_labels_2d, axis=0)
    all_labels_2d = [
                    np.nan_to_num(x, nan=0, posinf=0, neginf=0).astype(np.int32)
                    for x in all_labels_2d]
    _, records = track_masks_iou(
        all_labels_2d, iou_thr=iou_thr, memory=memory
    )

    all_tracked_files = []
    for frame, labels in enumerate(all_labels):
        # Change label id to track id
        tracking = np.zeros_like(labels)
        for record in records:
            frame_i = record["frame"]
            label_i = record["label"]
            track_id_i = record["track_id"]
            if frame == frame_i:
                tracking = np.where(
                    labels == label_i, track_id_i, tracking
                )
        all_tracked_files.append(tracking)

    # tracked_labels = relabel_masks_by_tracks(all_labels, assignments)

    # Change labels for all tracks 
    new_df = []
    for frame, (df, name) in enumerate(zip(all_df, all_names)):
        df['folder'] = folder
        df['track_id'] = 0
        df['file_name'] = name
        for record in records:
            if record['frame'] == frame:
                label_i = record['label']
                track_id_i = record['track_id']
                df.loc[df['label'] == label_i, 'track_id'] = track_id_i
        new_df.append(df[['folder', 'file_name', 'track_id', 'size', 'volume', 'integrated_density', 'x', 'y', 'z']])
    new_df = pd.concat(new_df)
    new_df['file_name'] = new_df['file_name'].apply(lambda p: p.replace('.nd2', ''))
    # new_df = new_df.pivot_table(columns='file_name', 
    #                             index=['folder', 'track_id'], 
    #                             values=['volume', 'integrated_density'], 
    #                             aggfunc='max').reset_index(drop=False)
    new_df = (
        new_df.pivot_table(
            columns='file_name',
            index=['folder', 'track_id'],
            values=['volume', 'integrated_density'],
            aggfunc='max'
        )
        .reset_index()
    )

    # flatten columns
    new_df.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0]
        for col in new_df.columns
    ]

    # new_columns = ['folder', 'track_id']
    new_columns = []
    all_cols = [col.replace('.nd2', '') for col in new_df.columns]
    new_columns.extend(all_cols)
    new_df = new_df[new_columns]
    new_df = new_df[~new_df[all_cols].isna().any(axis=1)]

    base_cols = ['folder', 'track_id']
    volume_cols = sorted(
        [c for c in new_df.columns if c.endswith('_volume')]
    )
    density_cols = sorted(
        [c for c in new_df.columns if c.endswith('_integrated_density')]
    )
    new_df = new_df[
        base_cols + volume_cols + density_cols
    ]

    new_df.to_csv(os.path.join(output_folder, f'results_{folder}.csv'), index=None)

    all_track_ids = pd.unique(new_df['track_id'])
    for frame, (labels, name) in enumerate(zip(all_tracked_files, all_names)):
    # Remove incorrect labels
        mask_found = ~np.isin(labels, all_track_ids)
        labels[mask_found]=0
        labels_path = os.path.join(output_folder, name.replace('.nd2', '_labels.tif'))
        tiff.imwrite(
            labels_path,
            labels.astype(np.uint16),
        )
    # Delete temporary files
    all_files = os.listdir(os.path.join(output_folder))
    all_files = list(filter(lambda p: ('.npy' in p), all_files))
    for file in all_files:
        path = os.path.join(output_folder, file)
        if os.path.exists(path):
            os.remove(path)

def nd2_to_tiff(entire_file_name):
    # Convert image to tif
    img = AICSImage(entire_file_name)
    data = img.get_image_data("CZYX", T=0)
    # tiff.imwrite(os.path.join('Nuclear_volume_with_tracking', folder, file_name.replace('.nd2', '.tif')), data[0])
    return data[0]

def remove_outliers(labels, min_size=1000, max_size=None):
    '''
    Clean very small and big labels
    '''
    # Get all non-zero coordinates and their labels
    z, y, x = np.nonzero(labels)
    labels_df = labels[z, y, x]
    # Create DataFrame
    df = pd.DataFrame({'label': labels_df, 'z': z, 'y': y, 'x': x})
    df['size'] = 1
    df = df.groupby(by=['label'], as_index=False).agg({'z':'mean', 'y':'mean', 'x':'mean', 'size':'sum'})
    df = df[df['size']>=min_size]
    if max_size is not None:
        df = df[df['size']<max_size]
    # Remove incorrect labels
    good_label = list(df['label'])
    mask_found = ~np.isin(labels, good_label)
    labels[mask_found]=0
    return labels

def compute_sizes(image, labels, dx, dy, dz, resize_factor):
    '''
    Clean very small and big labels
    '''
    # Get all non-zero coordinates and their labels
    z, y, x = np.nonzero(labels)
    labels_df = labels[z, y, x]
    values = image[z, y, x]
    # Create DataFrame
    df = pd.DataFrame({'label': labels_df, 'z': z, 'y': y, 'x': x, 'values':values})
    df['size'] = 1
    df = df.groupby(by=['label'], as_index=False).agg({'z':'mean', 
                                                       'y':'mean', 
                                                       'x':'mean', 
                                                       'size':'sum', 
                                                       'values':'sum'})
    # Compute volume
    df['volume']=df['size']*dx*dy*dz
    # Compute integrated density
    df['integrated_density']=df['values']*(resize_factor**2)
    
    return df

import numpy as np


def binary_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def size_similarity(v1, v2):
    mx = max(v1, v2)
    return min(v1, v2) / mx if mx > 0 else 0.0


def compute_match_score(
    feat_prev,
    feat_cur,
    w_xy=0.85,
    w_3d=0.05,
    w_size=0.10,
    max_xy_dist=None,
):
    """
    Matching score for 3D objects when Z size may differ across frames.
    """
    if max_xy_dist is not None:
        dist = np.linalg.norm(feat_prev["centroid_xy"] - feat_cur["centroid_xy"])
        if dist > max_xy_dist:
            return 0.0

    # always safe because proj_xy shape is (Y, X)
    iou_xy = binary_iou(feat_prev["proj_xy"], feat_cur["proj_xy"])

    # only compare 3D masks if same shape
    if feat_prev["mask3d"].shape == feat_cur["mask3d"].shape:
        iou_3d = binary_iou(feat_prev["mask3d"], feat_cur["mask3d"])
    else:
        iou_3d = 0.0

    size_score = size_similarity(feat_prev["volume"], feat_cur["volume"])

    return w_xy * iou_xy + w_3d * iou_3d + w_size * size_score

# def binary_iou(a, b):
#     """
#     IoU for two binary masks.
#     """
#     inter = np.logical_and(a, b).sum()
#     union = np.logical_or(a, b).sum()
#     return inter / union if union > 0 else 0.0


# def size_similarity(v1, v2):
#     """
#     Similarity of two sizes in [0, 1].
#     1 = same size, smaller if sizes differ.
#     """
#     mx = max(v1, v2)
#     return min(v1, v2) / mx if mx > 0 else 0.0


def extract_3d_object_features(lbl_img):
    """
    Extract features for each nonzero label in a 3D label image.

    Parameters
    ----------
    lbl_img : np.ndarray
        3D labeled image, shape (Z, Y, X), background=0

    Returns
    -------
    features : dict
        features[label] = {
            'mask3d': bool mask,
            'proj_xy': 2D bool projection,
            'centroid_3d': np.array([z, y, x]),
            'centroid_xy': np.array([y, x]),
            'volume': int,
            'area_xy': int,
        }
    """
    labels = np.unique(lbl_img)
    labels = labels[labels != 0]

    features = {}
    for lbl in labels:
        lbl = int(lbl)
        mask3d = (lbl_img == lbl)
        coords = np.argwhere(mask3d)

        centroid_3d = coords.mean(axis=0)  # z, y, x
        proj_xy = mask3d.any(axis=0)       # shape (Y, X)

        features[lbl] = {
            "mask3d": mask3d,
            "proj_xy": proj_xy,
            "centroid_3d": centroid_3d,
            "centroid_xy": centroid_3d[1:],
            "volume": int(mask3d.sum()),
            "area_xy": int(proj_xy.sum()),
        }

    return features


# def compute_match_score(
#     feat_prev,
#     feat_cur,
#     w_xy=0.8,
#     w_3d=0.1,
#     w_size=0.1,
#     max_xy_dist=None,
# ):
#     """
#     Compute matching score between two 3D objects.

#     Main score is based on XY projection IoU.
#     Optional weak 3D IoU and size similarity are added.

#     Returns
#     -------
#     score : float in [0, 1]
#     """
#     if max_xy_dist is not None:
#         dist = np.linalg.norm(feat_prev["centroid_xy"] - feat_cur["centroid_xy"])
#         if dist > max_xy_dist:
#             return 0.0

#     iou_xy = binary_iou(feat_prev["proj_xy"], feat_cur["proj_xy"])
#     iou_3d = binary_iou(feat_prev["mask3d"], feat_cur["mask3d"])
#     size_score = size_similarity(feat_prev["volume"], feat_cur["volume"])

#     return w_xy * iou_xy + w_3d * iou_3d + w_size * size_score


def build_score_matrix(
    prev_img,
    cur_img,
    prev_labels=None,
    cur_labels=None,
    prev_features=None,
    cur_features=None,
    w_xy=0.8,
    w_3d=0.1,
    w_size=0.1,
    max_xy_dist=None,
):
    """
    Build score matrix between labels in previous and current 3D label images.
    """
    if prev_features is None:
        prev_features = extract_3d_object_features(prev_img)
    if cur_features is None:
        cur_features = extract_3d_object_features(cur_img)

    if prev_labels is None:
        prev_labels = np.array(sorted(prev_features.keys()), dtype=int)
    if cur_labels is None:
        cur_labels = np.array(sorted(cur_features.keys()), dtype=int)

    score = np.zeros((len(prev_labels), len(cur_labels)), dtype=np.float32)

    for i, pl in enumerate(prev_labels):
        for j, cl in enumerate(cur_labels):
            score[i, j] = compute_match_score(
                prev_features[int(pl)],
                cur_features[int(cl)],
                w_xy=w_xy,
                w_3d=w_3d,
                w_size=w_size,
                max_xy_dist=max_xy_dist,
            )

    return score, prev_labels, cur_labels, prev_features, cur_features


def track_masks_3d_projection(
    masks,
    score_thr=0.2,
    memory=0,
    max_xy_dist=None,
    w_xy=0.8,
    w_3d=0.1,
    w_size=0.1,
):
    """
    Track 3D objects across timepoints using XY projection-based matching.

    Parameters
    ----------
    masks : list of np.ndarray
        List of 3D label images, each shape (Z, Y, X), background=0
    score_thr : float
        Minimum matching score to keep the same track
    memory : int
        How many missing frames a track can survive
    max_xy_dist : float or None
        Maximum centroid XY distance allowed for matching
    w_xy, w_3d, w_size : float
        Weights for score computation

    Returns
    -------
    assignments : list of dict
        assignments[t][label_in_frame_t] = track_id
    records : list of dict
        [{'frame': t, 'label': lbl, 'track_id': tid}, ...]
    """
    if len(masks) == 0:
        return [], []

    assignments = []
    records = []
    next_track_id = 1

    # active tracks:
    # active[track_id] = {
    #     'last_frame': int,
    #     'last_label': int,
    #     'features': object_features_dict_for_that_label
    # }
    active = {}

    # ---------- frame 0 ----------
    first = masks[0]
    first_features = extract_3d_object_features(first)

    map0 = {}
    for lbl in sorted(first_features.keys()):
        tid = next_track_id
        next_track_id += 1

        map0[int(lbl)] = tid
        active[tid] = {
            "last_frame": 0,
            "last_label": int(lbl),
            "features": first_features[int(lbl)],
        }
        records.append({"frame": 0, "label": int(lbl), "track_id": tid})

    assignments.append(map0)

    # ---------- next frames ----------
    for t in range(1, len(masks)):
        cur = masks[t]
        cur_features = extract_3d_object_features(cur)
        cur_labels = np.array(sorted(cur_features.keys()), dtype=int)

        # remove expired tracks
        expired = [
            tid for tid, st in active.items()
            if (t - st["last_frame"] - 1) > memory
        ]
        for tid in expired:
            del active[tid]

        cur_map = {}
        used_cur = set()

        active_tids = sorted(active.keys())

        if len(active_tids) > 0 and len(cur_labels) > 0:
            # build score matrix: rows = active tracks, cols = current labels
            score = np.zeros((len(active_tids), len(cur_labels)), dtype=np.float32)

            for i, tid in enumerate(active_tids):
                feat_prev = active[tid]["features"]
                for j, cl in enumerate(cur_labels):
                    feat_cur = cur_features[int(cl)]
                    score[i, j] = compute_match_score(
                        feat_prev,
                        feat_cur,
                        w_xy=w_xy,
                        w_3d=w_3d,
                        w_size=w_size,
                        max_xy_dist=max_xy_dist,
                    )

            if score.size > 0:
                r, c = linear_sum_assignment(-score)
            else:
                r, c = np.array([], dtype=int), np.array([], dtype=int)

            # accept good matches
            matched_tracks = set()
            for i, j in zip(r, c):
                if score[i, j] < score_thr:
                    continue

                tid = int(active_tids[i])
                cl = int(cur_labels[j])

                cur_map[cl] = tid
                used_cur.add(cl)
                matched_tracks.add(tid)

                active[tid] = {
                    "last_frame": t,
                    "last_label": cl,
                    "features": cur_features[cl],
                }
                records.append({"frame": t, "label": cl, "track_id": tid})

        # new tracks for unmatched current labels
        for cl in cur_labels:
            cl = int(cl)
            if cl in used_cur:
                continue

            tid = next_track_id
            next_track_id += 1

            cur_map[cl] = tid
            active[tid] = {
                "last_frame": t,
                "last_label": cl,
                "features": cur_features[cl],
            }
            records.append({"frame": t, "label": cl, "track_id": tid})

        assignments.append(cur_map)

    return assignments, records


def tracks_to_dataframe(records):
    """
    Optional helper if you want pandas DataFrame.
    """
    import pandas as pd
    return pd.DataFrame(records)


def relabel_masks_by_tracks(masks, assignments):
    """
    Convert original label images into track-id label images.

    Parameters
    ----------
    masks : list of np.ndarray
        Original 3D label images
    assignments : list of dict
        Output from track_masks_3d_projection

    Returns
    -------
    tracked_masks : list of np.ndarray
        Same shape as masks, but labels are replaced by track_id
    """
    tracked_masks = []

    for img, amap in zip(masks, assignments):
        out = np.zeros_like(img, dtype=np.int32)
        for lbl, tid in amap.items():
            out[img == lbl] = tid
        tracked_masks.append(out)

    return tracked_masks

def remove_outliers_pos(labels, df, edge_pos=30):
    '''
    Clean labels with incorrect Z position
    '''
    # # Compute statistics
    x = df['z']
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['z']>=lower_bound) & (df['z']<=upper_bound)]

    # Remove ids closed to edges
    max_x = labels.shape[2] - edge_pos
    max_y = labels.shape[1] - edge_pos
    df = df[(df['x']>edge_pos) & (df['x']<max_x) 
                & (df['y']>edge_pos) & (df['y']<max_y)]
    # Remove incorrect labels
    good_label = list(df['label'])
    mask_found = ~np.isin(labels, good_label)
    labels[mask_found]=0
    return labels, df

def make_2d_labels(labels):
    # Replace zeros with NaN so they are ignored
    arr_nonzero = np.where(labels == 0, np.nan, labels)
    # Compute mode along axis=0 (per pixel)
    mode_result = stats.mode(arr_nonzero, axis=0, nan_policy='omit', keepdims=False)
    mode_image = mode_result.mode
    return mode_image

def align_all_labels_2d(all_labels_2d, reference_idx=0, binary_for_shift=True):
    """
    Align all 2D arrays in all_labels_2d to one reference image.

    Parameters
    ----------
    all_labels_2d : list of np.ndarray
        List of 2D arrays with одинаковый shape (H, W)
    reference_idx : int
        Which image to use as reference
    binary_for_shift : bool
        If True, compute shift on (img > 0), which is usually better for label masks

    Returns
    -------
    aligned : list of np.ndarray
        Shifted images
    shifts : list of tuple
        (dy, dx) for each image
    """
    ref = all_labels_2d[reference_idx]

    if binary_for_shift:
        ref_for_shift = (ref > 0).astype(np.float32)
    else:
        ref_for_shift = ref.astype(np.float32)

    aligned = []
    shifts = []

    for img in all_labels_2d:
        if binary_for_shift:
            img_for_shift = (img > 0).astype(np.float32)
        else:
            img_for_shift = img.astype(np.float32)

        shift, error, _ = phase_cross_correlation(
            ref_for_shift,
            img_for_shift,
            upsample_factor=10
        )

        # apply shift to original image
        img_shifted = ndi_shift(
            img,
            shift=shift,   # (dy, dx)
            order=0,       # preserve labels
            mode="constant",
            cval=0,
            prefilter=False
        )

        aligned.append(img_shifted)
        shifts.append(tuple(shift))

    return aligned, shifts