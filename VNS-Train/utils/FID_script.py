# import os
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from scipy.linalg import sqrtm

# # 配置参数
# parser = argparse.ArgumentParser(description='Calculate FID between two datasets')
# parser.add_argument('--data1', type=str, required=True, help='Path to dataset 1')
# parser.add_argument('--data2', type=str, required=True, help='Path to dataset 2')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')
# parser.add_argument('--image_size', type=int, default=299, help='Input image size for Inception model')
# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
#                     help='Device to use (cuda or cpu)')
# args = parser.parse_args()


# class ImageDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform
#         self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('jpg', 'JPG'))]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img

# def get_inception_model(device):
#     # Load the pre-trained Inception v3 model
#     model = models.inception_v3(pretrained=True, transform_input=False)
#     model.eval()
#     model.to(device)
#     # Remove the fully connected layers and keep up to the last convolutional layer
#     return nn.Sequential(*list(model.children())[:-2])  # Keep layers up to the last convolutional block


# def extract_features(dataloader, model, device):
#     features = []
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(device)
#             # Inception v3默认输入尺寸为299x299
#             batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
#             # 预处理：Inception v3要求输入归一化到[-1, 1]
#             batch = (batch - 0.5) * 2.0
#             print(batch.shape)
#             # 提取特征（输出形状：batch_size x 2048 x 8 x 8）
#             feat = model(batch)
#             # 全局平均池化
#             feat = torch.mean(feat, dim=[2, 3])  # batch_size x 2048
#             features.append(feat.cpu().numpy())
#     return np.concatenate(features, axis=0)


# def calculate_fid(mu1, sigma1, mu2, sigma2):
#     # Ensure covariance matrices are positive semi-definite
#     eps = 1e-6
#     sigma1 += np.eye(sigma1.shape[0]) * eps
#     sigma2 += np.eye(sigma2.shape[0]) * eps

#     # Compute sqrtm and handle numerical instability
#     covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real

#     mu_diff = mu1 - mu2
#     trace = np.trace(sigma1 + sigma2 - 2 * covmean)
#     return np.dot(mu_diff, mu_diff) + trace


# def main():
#     # Data preprocessing with normalization for Inception v3
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # 加载数据集
#     dataset1 = ImageDataset(args.data1, transform=transform)
#     dataset2 = ImageDataset(args.data2, transform=transform)
#     dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=False)
#     dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=False)

#     # Check for empty datasets
#     if len(dataset1) == 0 or len(dataset2) == 0:
#         raise ValueError("One or both datasets are empty. Please provide valid datasets.")

#     # 加载Inception模型
#     model = get_inception_model(args.device)

#     # 提取特征
#     print("Extracting features from dataset 1...")
#     features1 = extract_features(dataloader1, model, args.device)
#     print("Extracting features from dataset 2...")
#     features2 = extract_features(dataloader2, model, args.device)

#     # 计算统计量
#     mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
#     mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

#     # 计算FID
#     fid = calculate_fid(mu1, sigma1, mu2, sigma2)
#     print(f"FID between {args.data1} and {args.data2}: {fid:.4f}")


# if __name__ == '__main__':
#     main()


import os
import torch
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.lower().endswith(('jpg', 'JPG'))]
        
        if "LISU" in img_dir:
            self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.lower().endswith(('png'))]
        
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_inception_model():
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.eval()  # 设置为评估模式
    inception.fc = torch.nn.Identity()  # 移除最后的分类层，输出2048维特征
    return inception

def compute_activations(dataloader, model, device):
    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            pred = model(batch)
            activations.append(pred.cpu().numpy())
    return np.concatenate(activations, axis=0)

def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    
        # Ensure covariance matrices are positive semi-definite
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    # Compute sqrtm and handle numerical instability
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    mu_diff = mu1 - mu2
    trace = np.trace(sigma1 + sigma2 - 2 * covmean)
    return np.dot(mu_diff, mu_diff) + trace

    # ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # if np.iscomplexobj(covmean):
    #     covmean = covmean.real
        
    # fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    # return fid

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculate FID between two datasets')
    parser.add_argument('--data1', type=str, required=True, help='Path to first image dataset')
    parser.add_argument('--data2', type=str, required=True, help='Path to second image dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    args = parser.parse_args()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 图像预处理 (适配InceptionV3)
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset1 = ImageDataset(args.data1, transform)
    dataset2 = ImageDataset(args.data2, transform)
    
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)

    # 加载Inception模型
    inception = get_inception_model().to(device)

    # 计算特征激活
    act1 = compute_activations(dataloader1, inception, device)
    act2 = compute_activations(dataloader2, inception, device)

    # 计算FID
    fid_value = calculate_fid(act1, act2)
    print(f'FID between {args.data1} and {args.data2}: {fid_value:.4f}')