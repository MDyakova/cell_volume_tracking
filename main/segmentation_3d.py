"""
Code for 3D cell segmentation and tracking
"""

import os
import glob
import argparse
import ast
from cellpose import models, io
import tifffile as tiff
import pandas as pd
from utilities import make_3d_segmentation

io.logger_setup()  # run this to get printing of progress


def main():
    """
    3D Cell segmentation and tracking
    """

    # Load input parameters
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument("--image_directory", type=str, help="Directory with tiff files")
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument("--resize_factor", type=int, default=5, help="Decreases size of tile")
    parser.add_argument(
        "--cell_diameter", type=int, default=30, help="Cell diameter"
    )
    parser.add_argument(
        "--name_filter", type=str, default="", help="Part of file name for filtration"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="GPU utilizing"
    )
    parser.add_argument(
        "--dx", type=float, default=0.1083, help="dx size"
    )
    parser.add_argument(
        "--dy", type=float, default=0.1083, help="dy size"
    )
    parser.add_argument(
        "--dz", type=float, default=0.2, help="dz size"
    )
    parser.add_argument(
        "--iou_thr", type=float, default=0.25, help="Minimum score to accept a match"
    )
    parser.add_argument(
        "--memory", type=float, default=1, help="How long a track can disappear and still be matched later"
    )
    parser.add_argument(
        "--min_size", type=int, default=5000, help="Min cell size"
    )
    parser.add_argument(
        "--max_size", type=int, default=None, help="Max cell size"
    )
    parser.add_argument(
        "--min_value", type=int, default=0, help="Min signal"
    )
    parser.add_argument(
        "--max_value", type=int, default=None, help="Max signal"
    )
    parser.add_argument(
        "--edge_pos", type=float, default=30, help="Distance from boarders"
    )
    parser.add_argument(
        "--anisotropy", type=float, default=1.0, help="Corrects Z spacing"
    )
    parser.add_argument(
        "--flow_threshold", type=float, default=0.4, help="Higher = fewer objects"
    )
    parser.add_argument(
        "--cellprob_threshold", type=float, default=0.0, help="Lower = more sensitive"
    )
    parser.add_argument(
        "--channels_for_volume", type=str, default=[0], help="Choose channels to compute volume"
    )
    parser.add_argument(
        "--channels_for_intens", type=str, default=[0], help="Choose channels to compute intensity"
    )
    parser.add_argument(
        "--channels_names", type=str, default=['ch1'], help="Choose channels names"
    )

    args = parser.parse_args()

    image_directory = os.path.abspath(args.image_directory)
    output_directory = os.path.abspath(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)
    resize_factor = int(args.resize_factor)
    cell_diameter = int(args.cell_diameter)
    name_filter = args.name_filter
    use_gpu = args.use_gpu
    dx = float(args.dx)
    dy = float(args.dy)
    dz = float(args.dz)
    edge_pos = int(args.edge_pos)
    min_size = int(args.min_size)
    if args.max_size is not None:
        max_size = int(args.max_size)
    else:
        max_size = None

    min_value = int(args.min_value)
    if args.max_value is not None:
        max_value = int(args.max_value)
    else:
        max_value = None

    # score = w_xy * IoU_xy + w_3d * IoU_3d + w_size * size_similarity
    iou_thr = float(args.iou_thr)
    memory = int(args.memory)
    anisotropy = float(args.anisotropy)
    flow_threshold = float(args.flow_threshold)
    cellprob_threshold = float(args.cellprob_threshold)

    channels_for_volume = str(args.channels_for_volume)
    channels_for_volume = channels_for_volume[1:-1].split(',')
    channels_for_volume = [int(x) for x in channels_for_volume]
    channels_for_intens = str(args.channels_for_intens)
    channels_for_intens = channels_for_intens[1:-1].split(',')
    channels_for_intens = [int(x) for x in channels_for_intens]
    channels_names = str(args.channels_names)
    channels_names = channels_names[1:-1].split(',')

    params_dict = {
        'dx':dx,
        'dy':dy,
        'dz':dz,
        'iou_thr':iou_thr,
        'memory':memory,
        'edge_pos':edge_pos,
        'min_size':min_size,
        'max_size':max_size,
        'min_value':min_value,
        'max_value':max_value,
        'anisotropy':anisotropy,
        'flow_threshold':flow_threshold,
        'cellprob_threshold':cellprob_threshold,
        'channels_for_volume':channels_for_volume,
        'channels_for_intens':channels_for_intens,
        'channels_names':channels_names
    }


    # Load list with processed files
    if os.path.exists(os.path.join(output_directory, "processed_files.txt")):
        with open(
            os.path.join(output_directory, "processed_files.txt"), "r", encoding="utf-8"
        ) as f:
            processed_files = f.read()
        processed_files = processed_files.split("\n")
    else:
        processed_files = []

    # Go throw all subdirectories in the image_directory
    all_folders = os.listdir(image_directory)
    for folder in all_folders:
        if (name_filter in folder) & (folder not in processed_files):
            try:
                make_3d_segmentation(
                            image_directory,
                            output_directory,
                            folder,
                            resize_factor,
                            cell_diameter,
                            params_dict)
                with open(
                    os.path.join(output_directory, "processed_files.txt"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(f"{folder}" + "\n")
            except Exception as e:
                print(folder, e)



if __name__ == "__main__":
    main()
