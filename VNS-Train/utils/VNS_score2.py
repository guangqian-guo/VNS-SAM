import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import math


def safe_dilate(mask, max_kernel_size=15):
    # 检测目标是否靠近边界
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # 获取目标外接矩形坐标
    x, y, bw, bh = cv2.boundingRect(contours[0])
    border_threshold = 5  # 距离边界的容忍像素
    
    # 计算允许的最大核尺寸
    left_space = x
    right_space = w - (x + bw)
    top_space = y
    bottom_space = h - (y + bh)
    safe_kernel_size = min(
        max_kernel_size,
        left_space, right_space, top_space, bottom_space
    )
    safe_kernel_size = max(1, safe_kernel_size)  # 至少为1
    
    # 使用安全核大小膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_kernel_size, safe_kernel_size))
    return cv2.dilate(mask, kernel, borderType=cv2.BORDER_ISOLATED)

def get_local_background_mask(mask, max_kernel_size=15):
    dilated_mask = safe_dilate(mask, max_kernel_size)
    local_bg_mask = dilated_mask - mask
    return local_bg_mask.astype(np.uint8)


def compute_foreground_background_contrast(image, mask):
    
    local_bg_mask = get_local_background_mask(mask)
    
    if local_bg_mask.max() == 0:
        local_bg_mask = 1 - mask
        
    # 分离前景和背景
    try:
        fg = cv2.bitwise_and(image, image, mask=mask)
        bg = cv2.bitwise_and(image, image, mask=local_bg_mask)
    except:
        print("Error in separating foreground and background.")
        return 0.5
        
    # 计算颜色差异（LAB空间）
    lab_fg = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)
    mean_fg = lab_fg[mask == 1].mean(axis=0)
    mean_bg = lab_bg[local_bg_mask == 1].mean(axis=0)
    color_diff = np.linalg.norm(mean_fg - mean_bg)  # LAB欧氏距离

    # 计算纹理差异（GLCM对比度）
    def glcm_contrast(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        return graycoprops(glcm, 'contrast')[0, 0]
    
    texture_fg = glcm_contrast(fg) if np.any(mask) else 0
    texture_bg = glcm_contrast(bg)
    texture_diff = np.abs(texture_fg - texture_bg)

    # 归一化（假设最大颜色差异为100，纹理差异为1000）
    color_norm = color_diff / 100.0
    texture_norm = texture_diff / 1000.0
    c_fb = (color_norm + texture_norm) / 2.0

    return c_fb  # 低对比度场景得分高


# def compute_foreground_background_contrast(image, mask):
#     # 分离前景和背景
#     fg = cv2.bitwise_and(image, image, mask=mask)
#     bg = cv2.bitwise_and(image, image, mask=1 - mask)
    
#     # 计算前景与背景LAB空间的直方图相似度（低对比度→相似度高）
#     lab_fg = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
#     lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)
    
#     # 计算L通道直方图相似性（使用Bhattacharyya系数）
#     hist_fg = cv2.calcHist([lab_fg], [0], None, [256], [0, 256])
#     hist_bg = cv2.calcHist([lab_bg], [0], None, [256], [0, 256])
#     similarity = cv2.compareHist(hist_fg, hist_bg, cv2.HISTCMP_BHATTACHARYYA)
    
#     # 低对比度→相似度高→C_fb值低→(1 - C_fb)值高
#     return similarity  # 值域[0,1]，0表示完全一致

def compute_blur_degree(image, mask, ksize=5):
    # 提取边缘ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    edge_roi = dilated - eroded
    
    # 计算边缘区域的梯度幅值均值（模糊→均值低→归一化后得分高）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    try:
        edge_grad_mean = grad_mag[edge_roi > 0].mean()
    except:
        print("Error in computing edge gradient mean.")
        return 0.5
    blur_score = 1.0 - (edge_grad_mean / 255.0)  # 假设最大梯度幅值为255
    # print(blur_score)
    return blur_score


# def compute_edge_sparsity(image, mask):
#     # 自适应Canny边缘检测
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     high_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
#     low_thresh = 0.5 * high_thresh
#     edges = cv2.Canny(gray, low_thresh, high_thresh)
    
#     # 提取目标区域的边缘连通域
#     target_edges = cv2.bitwise_and(edges, edges, mask=mask)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(target_edges)
    
#     # 计算稀疏性：连通域多且平均长度短→得分高
#     if num_labels <= 1:
#         return 0.0
#     lengths = stats[1:, cv2.CC_STAT_WIDTH]
#     avg_length = lengths.mean()
#     sparsity = (num_labels - 1) / (avg_length + 1e-6)  # 连通域数/平均长度
#     return min(sparsity / 10.0, 1.0)  # 归一化到[0,1]

# def compute_edge_sparsity(image, mask):
#     # 提取目标区域的边缘连通域
#     edges = cv2.Canny(image, 50, 150)
#     target_edges = cv2.bitwise_and(edges, edges, mask=mask)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(target_edges)
    
#     if num_labels <= 1:
#         return 0.0

#     # 计算总边缘长度和连通域数量
#     total_length = np.sum(stats[1:, cv2.CC_STAT_WIDTH])
#     sparsity = (num_labels - 1) / (total_length + 1e-6)  # 连通域多且总长短→得分高
#     return min(sparsity * 0.1, 1.0)  # 调整系数确保归一化


def compute_vns_score(image_path, mask_path, alpha=1/2, beta=1/2, gamma=1/3):
    # 读取图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 128).astype(np.uint8)
    
    # 计算各特征项
    c_fb = compute_foreground_background_contrast(image, mask)
    b_blur = compute_blur_degree(image, mask)
    # e_sparsity = compute_edge_sparsity(image, mask)
    
    # 综合得分
    vns_score = alpha * (1 - c_fb) + beta * b_blur 
    return c_fb, b_blur, 0, vns_score


def main(args):
    # os.makedirs(args.output_dir, exist_ok=True)
    scores = []
    total_contrast_term = 0
    total_blur_degree = 0
    total_edge_sparsity = 0
    total_vns_score = 0
    num_images = 0

    for i, img_name in enumerate(tqdm(os.listdir(args.image_dir))):
        if not img_name.endswith('.jpg'):
            continue
        image_path = os.path.join(args.image_dir, img_name)
        mask_path = os.path.join(args.mask_dir, img_name.replace('.jpg', '.png'))
        contrast_term, blur_degree, edge_sparsity, vns_score = compute_vns_score(image_path, mask_path)
       
        if vns_score < 0 or vns_score > 1 or math.isnan(vns_score):
            continue
        
        scores.append((img_name, vns_score))
        
        total_contrast_term += contrast_term
        total_blur_degree += blur_degree
        total_edge_sparsity += edge_sparsity
        total_vns_score += vns_score
        num_images += 1
        
        # Visualization of the score on the image
        image = cv2.imread(image_path)
        text = f"Score: {vns_score:.2f}"
        font_scale = 1
        thickness = 2
        color = (0, 255, 0)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        position = (10, text_height + 10)  # Adjust position to ensure visibility
        
        # Ensure the text is within the image boundaries
        if position[1] + text_height > image.shape[0]:
            position = (10, image.shape[0] - 10)
        
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Save the image with the score to the output directory
        # output_path = os.path.join(args.output_dir, img_name)
        # cv2.imwrite(output_path, image)
    print(total_vns_score, num_images)
    
    # Calculate and print average values
    avg_contrast_term = total_contrast_term / num_images
    avg_blur_degree = total_blur_degree / num_images
    avg_edge_sparsity = total_edge_sparsity / num_images
    avg_vns_score = total_vns_score / num_images

    print(f"Average Contrast Term: {avg_contrast_term:.4f}")
    print(f"Average Blur Degree: {avg_blur_degree:.4f}")
    print(f"Average Edge Sparsity: {avg_edge_sparsity:.4f}")
    print(f"Average VNS Score: {avg_vns_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute VNS Score and visualize results")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to save output images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images")
    args = parser.parse_args()
    main(args)
    
    
    