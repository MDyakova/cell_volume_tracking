"""
Code for cell segmentation and tracking
"""

import os
import glob
import argparse
from cellpose import models, io
import tifffile as tiff
import pandas as pd
from utilities import make_segmentation

io.logger_setup()  # run this to get printing of progress


def main():
    """
    Cell segmentation and tracking
    """

    # Load input parameters
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument("--image_directory", type=str, help="Directory with tiff files")
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument("--tile_size", type=int, default=-1, help="Size of one tile")
    parser.add_argument(
        "--background_threshold", type=int, default=0, help="Background treshold"
    )
    parser.add_argument(
        "--cell_diameter_min", type=int, default=30, help="Minimal cell diameter"
    )
    parser.add_argument(
        "--cell_diameter_max", type=int, default=100, help="Maximal cell diameter"
    )
    parser.add_argument(
        "--name_filter", type=str, default="", help="Part of file name for filtration"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=True, help="GPU utilizing"
    )
    args = parser.parse_args()

    image_directory = os.path.abspath(args.image_directory)
    output_directory = os.path.abspath(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)
    tile_size = int(args.tile_size)
    background_threshold = int(args.background_threshold)
    cell_diameter_min = int(args.cell_diameter_min)
    cell_diameter_max = int(args.cell_diameter_max)
    name_filter = args.name_filter
    use_gpu = args.use_gpu

    # Load model
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="omnipose_cyto")

    # Load list with processed files
    if os.path.exists(os.path.join(output_directory, "processed_files.txt")):
        with open(
            os.path.join(output_directory, "processed_files.txt"), "r", encoding="utf-8"
        ) as f:
            processed_files = f.read()
        processed_files = processed_files.split("\n")
    else:
        processed_files = []
    print(processed_files)

    # Go throw all subdirectories in the image_directory
    tiff_files = glob.glob(os.path.join(image_directory, "**", "*.tif"), recursive=True)
    for entire_file_name in tiff_files:
        if name_filter in entire_file_name:
            # Read image
            image = tiff.imread(entire_file_name)
            # Name to save files
            file_name_save = entire_file_name.split("/")[-1].split(".tif")[0]

            if tile_size == -1:
                file_name = file_name_save
                results = make_segmentation(
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
                )
                with open(
                    os.path.join(output_directory, "processed_files.txt"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(f"{file_name}.tif" + "\n")
            else:
                results = []
                # Split image to XY tiles
                for i in range(0, image.shape[1], tile_size - tile_size // 2):
                    for j in range(0, image.shape[2], tile_size - tile_size // 2):
                        file_name = f"{file_name_save}_x_{str(j)}_y_{str(i)}"

                        results_tile = make_segmentation(
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
                            i=i,
                            j=j,
                        )
                        results.extend(results_tile)
                        with open(
                            os.path.join(output_directory, "processed_files.txt"),
                            "a",
                            encoding="utf-8",
                        ) as f:
                            f.write(f"{file_name}.tif" + "\n")

            results = pd.DataFrame(
                results,
                columns=(
                    "file_name_save",
                    "label_id",
                    "step",
                    "integrated_density",
                    "cell_area",
                    "x_min",
                    "x_max",
                    "y_min",
                    "y_max",
                    "height",
                    "width",
                ),
            )
            # Filter masks are closed to edges
            results["x_delta"] = results["width"] - results["x_max"]
            results["y_delta"] = results["height"] - results["y_max"]
            results_group = results.groupby(
                by=["file_name_save", "label_id"], as_index=False
            ).min()
            mask = results_group[["x_min", "y_min", "x_delta", "y_delta"]].min(axis=1) > 10
            correct_labels = results_group.loc[mask, "label_id"].tolist()

            results = results[results["label_id"].isin(correct_labels)]

            results = results[
                [
                    "file_name_save",
                    "label_id",
                    "step",
                    "integrated_density",
                    "cell_area",
                ]
            ]
            results.to_csv(
                os.path.join(output_directory, f"results_{file_name}.csv"), index=None
            )


if __name__ == "__main__":
    main()
