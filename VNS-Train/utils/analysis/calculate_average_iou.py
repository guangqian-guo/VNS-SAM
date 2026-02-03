"""commond
python  utils/analysis/calculate_average_iou.py  --scores_file  work-dir/sam-l/coco/scores.txt   --threshold  0.8  --image_dir /mnt/nvme1n1/Guo/dataset/RobustSeg/test/COCO/images  --mask_dir /mnt/nvme1n1/Guo/dataset/RobustSeg/test/COCO/masks  --output_image_dir data/VNS-COCO/images  --output_mask_dir  data/VNS-COCO/masks 
"""

import argparse
import os
import shutil

def read_scores(scores_file):
    vns_scores = []
    ious = []
    images = []
    bious = []
    with open(scores_file, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            try:
                image = parts[0].split(': ')[1]
                vns_score = float(parts[1].split(': ')[1])
                iou = float(parts[2].split(': ')[1])
                biou = float(parts[3].split(': ')[1])
                images.append(image)
                vns_scores.append(vns_score)
                ious.append(iou)
                bious.append(biou)
                   
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
            
    return images, vns_scores, ious, bious

def calculate_average_iou(scores_file, threshold, image_dir, mask_dir, output_image_dir, output_mask_dir):
    images, vns_scores, ious, bious = read_scores(scores_file)
    
    filtered_ious = [iou for vns_score, iou in zip(vns_scores, ious) if vns_score <= threshold]
    filtered_bious = [biou for vns_score, biou in zip(vns_scores, bious) if vns_score <= threshold]
    
    filtered_images = [image for image, vns_score in zip(images, vns_scores) if vns_score <= threshold]
    
    if not filtered_ious:
        print(f"No scores above the threshold {threshold}")
        return
    
    average_iou = sum(filtered_ious) / len(filtered_ious)
    average_biou = sum(filtered_bious) / len(filtered_bious)
    
    print(f"Number of VNS scores >= {threshold}: {len(filtered_ious)}")
    print(f"Average IoU/BIoU for VNS scores >= {threshold}: {average_iou:.4f}/{average_biou:.4f}")

    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    for image_name in filtered_images:
        mask_path = os.path.join(mask_dir, image_name)
        image_path = os.path.join(image_dir, image_name.replace('.png', '.jpg'))
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            shutil.copy(image_path, output_image_dir)
            shutil.copy(mask_path, output_mask_dir)
        else:
            print(f"Warning: Image or mask not found for {image_path}")




def main():
    parser = argparse.ArgumentParser(description="Calculate average IoU for VNS scores above a threshold and save selected images and masks")
    parser.add_argument("--scores_file", type=str, required=True, help="Path to the scores.txt file")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for VNS scores")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--output_image_dir", type=str, required=True, help="Directory to save selected images")
    parser.add_argument("--output_mask_dir", type=str, required=True, help="Directory to save selected masks")
    args = parser.parse_args()
    
    calculate_average_iou(args.scores_file, args.threshold, args.image_dir, args.mask_dir, args.output_image_dir, args.output_mask_dir)

if __name__ == "__main__":
    main()
