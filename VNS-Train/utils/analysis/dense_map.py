import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def read_scores(scores_file):
    vns_scores = []
    ious = []
    
    with open(scores_file, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            try:
                vns_score = float(parts[1].split(': ')[1])
                iou = float(parts[2].split(': ')[1])
                # Only append if vns_score is non-negative
                if vns_score >= 0:
                    vns_scores.append(vns_score)
                    ious.append(iou)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
            
    return np.array(vns_scores), np.array(ious)

def draw_densemap(x, y):
    # 设置 Seaborn 风格
    sns.set(style="whitegrid", font_scale=1.5, rc={"axes.facecolor": (0, 0, 0, 0)})

    # 转换为 DataFrame 以便于使用 Seaborn
    data = pd.DataFrame({'x': x, 'y': y})

    # 创建 JointGrid，调整 space 参数贴合边缘分布
    g = sns.JointGrid(data=data, x="x", y="y", height=8, space=0)  # 调整 space 参数以贴合边缘图

    # 绘制密度图（中心）和边缘密度分布图
    g.plot_joint(
        sns.kdeplot, 
        cmap="Blues",  
        fill=True, 
        thresh=0.05, 
        levels=20,   # 适当调高等高线密度
        alpha=1    # 稍微降低透明度
    )

    # 在边缘绘制直方图，使用适合的颜色和对比度
    g.plot_marginals(
        sns.histplot, 
        kde=True, 
        color="#87CEEB", 
        edgecolor="black", 
        linewidth=2,
        alpha=1, 
        bins=30
    )

    # g.ax_joint.set_xlim(-0.05, 0.25)  # 将横坐标的范围限制在 [-3, 3]


    # 调整标题和标签字体
    g.set_axis_labels("Image Quality", "IoU", fontsize=16, labelpad=15)
    # g.fig.suptitle("Density and Marginal Probability Distributions", fontsize=18, y=1.02, weight="bold")


    g.ax_joint.grid(False)  # 去除主密度图的网格线
    g.ax_marg_x.grid(False)  # 去除顶部边缘图的网格线
    g.ax_marg_y.grid(False)  # 去除右侧边缘图的网格线
    sns.despine(ax=g.ax_joint, left=False, bottom=False)  # 为主密度图添加边框

    # # 去除边框并美化网格
    # g.ax_joint.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)  # 稍微降低网格线的透明度
    # sns.despine(left=True, bottom=True)

    # 保存图像为高分辨率
    plt.savefig('GenSAM2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

def main():
    parser = argparse.ArgumentParser(description="Draw density map for VNS scores and IoU values")
    parser.add_argument("--scores_file", type=str, required=True, 
                       help="Path to scores.txt containing VNS scores and IoU values")
    parser.add_argument("--output", type=str, default="density_map.png",
                       help="Output image path")
    args = parser.parse_args()

    # Read scores from file
    vns_scores, ious = read_scores(args.scores_file)
    
    # Draw density map
    draw_densemap(vns_scores, ious)
    plt.savefig(args.output, dpi=300, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    main()


