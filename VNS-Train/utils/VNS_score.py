import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
import os


def compute_foreground_background_contrast(image_path, mask_path):
    # 读取图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    mask = (mask > 128).astype(np.uint8)  # 二值化mask

    # 分离前景和背景
    fg = cv2.bitwise_and(image, image, mask=mask)
    bg = cv2.bitwise_and(image, image, mask=1 - mask)

    # 计算颜色差异（LAB空间）
    lab_fg = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)
    mean_fg = lab_fg[mask == 1].mean(axis=0)
    mean_bg = lab_bg[mask == 0].mean(axis=0)
    color_diff = np.linalg.norm(mean_fg - mean_bg)  # LAB欧氏距离

    # 计算纹理差异（GLCM对比度）
    def glcm_contrast(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        return greycoprops(glcm, 'contrast')[0, 0]
    
    texture_fg = glcm_contrast(fg) if np.any(mask) else 0
    texture_bg = glcm_contrast(bg)
    texture_diff = np.abs(texture_fg - texture_bg)

    # 归一化（假设最大颜色差异为100，纹理差异为1000）
    color_norm = color_diff / 100.0
    texture_norm = texture_diff / 1000.0
    c_fb = (color_norm + texture_norm) / 2.0

    return 1 - c_fb  # 低对比度场景得分高

# 更新后的VNS Score计算函数
def compute_vns_score(image_path, mask_path, alpha=1.0, beta=1.0, gamma=1.0, block_size=16):
    # 计算对比度项
    contrast_term = compute_foreground_background_contrast(image_path, mask_path)
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算边缘密度和局部熵（原代码部分）
    
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)  # Canny边缘检测
    e_density = np.mean(edges / 255.0)  # 边缘像素密度
    
    # 计算局部信息熵项 (S_local)
    h, w = gray.shape
    entropy_list = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size == 0:
                continue
            entropy_block = shannon_entropy(block)
            entropy_list.append(entropy_block)
    local_entropy = np.mean(entropy_list) / 8.0  # 归一化到[0,1] (8-bit最大熵为8)
    
    # 综合计算
    vns_score = alpha * contrast_term + beta * e_density + gamma * local_entropy
    return vns_score


def plot_vns_components(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    entr = entropy(gray.astype(np.uint8), disk(5))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(141), plt.imshow(image), plt.title("Original")
    plt.subplot(142), plt.imshow(gray, cmap='gray'), plt.title("Contrast")
    plt.subplot(143), plt.imshow(edges, cmap='gray'), plt.title("Edges")
    plt.subplot(144), plt.imshow(entr, cmap='jet'), plt.title("Local Entropy")
    plt.show()

# plot_vns_components("example.jpg")

image_dir = ""
scores = []
for img_name in os.listdir(image_dir):
    path = os.path.join(image_dir, img_name)
    score = compute_vns_score(path)
    
    
    scores.append((img_name, score))


