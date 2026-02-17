# cell_volume_tracking

segmentation.py --image_directory "../Avik/Eukaryotic_cell_QPM_Tracking_and_Segmentation" --output_directory "/../Avik/Eukaryotic_cell_QPM_results_new" --tile_size 640 --name_filter axl

docker run --rm --gpus all -v "C:\work_dir\cell_tracking_files:/cell_tracking_files" mdyakova/cell_tracking:v1 python segmentation.py --image_directory "/cell_tracking_files/data" --output_directory "/cell_tracking_files/tracking_results" --name_filter Xenopus --cell_diameter_min 30 --cell_diameter_max 100

