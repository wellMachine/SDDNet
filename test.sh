CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "path/to/checkpoint.pth" \
--test_image_path "path/to/test/images" \
--test_gt_path "path/to/test/ground_truth" \
--save_path "path/to/save/predictions"