""""
Functions for segmentation.py code
"""

from collections import defaultdict
from itertools import combinations
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from PIL import Image
import tifffile as tiff
from tqdm import tqdm


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
