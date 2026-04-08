CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "path/to/hiera_checkpoint.pt" \
--train_image_path "path/to/train/images" \
--train_mask_path "path/to/train/masks" \
--save_path "path/to/save/checkpoints" \
--epoch 400 \
--lr 0.001 \
--batch_size 8