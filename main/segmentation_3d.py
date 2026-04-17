"""
Code for 3D cell segmentation and tracking
"""

import os
import glob
import argparse
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
    # parser.add_argument(
    #     "--w_3d", type=float, default=0.0, help="Weight of full 3D IoU"
    # )
    # parser.add_argument(
    #     "--w_size", type=float, default=0.1, help="Weight of size similarity"
    # )
    parser.add_argument(
        "--edge_pos", type=float, default=0.1, help="Distance from boarders"
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

    # score = w_xy * IoU_xy + w_3d * IoU_3d + w_size * size_similarity
    iou_thr = float(args.iou_thr)
    memory = int(args.memory)
    # max_xy_dist = int(args.max_xy_dist)
    # w_xy = float(args.w_xy)
    # w_3d = float(args.w_3d)
    # w_size = float(args.w_size)

    params_dict = {
        'dx':dx,
        'dy':dy,
        'dz':dz,
        'iou_thr':iou_thr,
        'memory':memory,
        # 'max_xy_dist':max_xy_dist,
        # 'w_xy':w_xy,
        # 'w_3d':w_3d,
        # 'w_size':w_size,
        'edge_pos':edge_pos,
        'min_size':min_size,
        'max_size':max_size
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



if __name__ == "__main__":
    main()
