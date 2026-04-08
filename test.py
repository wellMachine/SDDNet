import os
import argparse
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SDDNet import SDDNet
from dataset import TestDataset

def main():
    parser = argparse.ArgumentParser(description="Test SDDNet Model with FPS Measurement")
    parser.add_argument("--checkpoint", type=str, default="path/to/checkpoint.pth",
                        help="Path to the checkpoint of SDDNet",
                        default='')
    parser.add_argument("--test_image_path", type=str,
                        help="Path to the image files for testing",
                        default="")
    parser.add_argument("--test_gt_path", type=str,
                        help="Path to the mask files for testing",
                        default="")
    parser.add_argument("--save_path", type=str,
                        help="Path to save the predicted masks",
                        default="")
    args = parser.parse_args()

    print("Arguments:")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Image Path: {args.test_image_path}")
    print(f"Test GT Path: {args.test_gt_path}")
    print(f"Save Path: {args.save_path}\n")

    # Verify paths
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file does not exist at {args.checkpoint}")
        return
    if not os.path.isdir(args.test_image_path):
        print(f"Error: Test image path does not exist or is not a directory: {args.test_image_path}")
        return
    if not os.path.isdir(args.test_gt_path):
        print(f"Error: Test GT path does not exist or is not a directory: {args.test_gt_path}")
        return
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Save path is set to: {args.save_path}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Initialize Test Dataset
    try:
        test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352)
        print(f"Test dataset initialized with size: {test_loader.size}")
    except Exception as e:
        print(f"Error initializing TestDataset: {e}")
        return

    if test_loader.size == 0:
        print("Error: Test dataset is empty. Please check the test_image_path and test_gt_path.")
        return

    # Initialize Model
    try:
        model = SDDNet().to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 建立结果保存目录
    os.makedirs(args.save_path, exist_ok=True)
    sub_save_path = os.path.join(args.save_path)
    os.makedirs(sub_save_path, exist_ok=True)

    # FPS 统计变量
    total_forward_time = 0.0
    num_images = 0

    # Processing Loop
    for i in range(test_loader.size):
        print(f"Processing image {i+1}/{test_loader.size}")
        try:
            with torch.no_grad():
                image, gt, name = test_loader.load_data()
                if image is None or gt is None or name is None:
                    print(f"Warning: Skipping index {i} due to missing data.")
                    continue

                gt = np.asarray(gt, np.float32)
                image = image.to(device)

                # 使用 torch.cuda.Event 测量 forward 时间
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                res, _, _ = model(image)
                end_event.record()
                torch.cuda.synchronize()
                forward_time = start_event.elapsed_time(end_event) / 1000.0  # 单位秒
                total_forward_time += forward_time
                num_images += 1

                # Upsample 和后处理
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)

                # Save the result
                save_filename = os.path.splitext(name)[0] + ".png"
                save_full_path = os.path.join(sub_save_path, save_filename)
                imageio.imsave(save_full_path, res)
                print(f"Saved predicted mask to {save_full_path}\n")
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")

    # 计算并打印平均 FPS
    avg_fps = num_images / total_forward_time if total_forward_time > 0 else 0
    print(f"Processed {num_images} images in {total_forward_time:.2f}s, Average FPS: {avg_fps:.2f}")
    print("Processing completed.")

if __name__ == "__main__":
    main()
