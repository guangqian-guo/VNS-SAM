import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
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
                # Only append if values are valid (not nan or inf) and vns_score >= 0
                if np.isfinite(vns_score) and np.isfinite(iou) and vns_score >= 0:
                    vns_scores.append(vns_score)
                    ious.append(iou)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid line: {line.strip()}")
                continue
    
    print(f"Total samples after filtering: {len(vns_scores)} (filtered out negative VNS scores)")        
    return np.array(vns_scores), np.array(ious)

def analyze_correlation(vns_scores, ious, output_dir):
    # Remove any remaining nan or inf values
    mask = np.logical_and(np.isfinite(vns_scores), np.isfinite(ious))
    vns_scores = vns_scores[mask]
    ious = ious[mask]
    
    if len(vns_scores) == 0 or len(ious) == 0:
        print("Error: No valid data points after filtering")
        return None, None
    
    # Calculate correlation coefficient and p-value
    try:
        correlation, p_value = stats.pearsonr(vns_scores, ious)
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return None, None
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(vns_scores, ious, alpha=0.5)
    plt.xlabel('VNS Score')
    plt.ylabel('IoU')
    plt.title(f'VNS Score vs IoU (Correlation: {correlation:.3f}, p-value: {p_value:.3e})')
    
    # Add trend line
    # z = np.polyfit(vns_scores, ious, 1)
    # p = np.poly1d(z)
    # plt.plot(vns_scores, p(vns_scores), "r--", alpha=0.8)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vns_iou_correlation.png'))
    plt.close()
    
    # Print statistics
    print(f"\nCorrelation analysis results:")
    print(f"Number of valid samples: {len(vns_scores)}")
    print(f"Pearson correlation coefficient: {correlation:.3f}")
    print(f"P-value: {p_value:.3e}")
    print("\nDescriptive statistics:")
    print("\nVNS Scores:")
    print(f"Mean: {np.mean(vns_scores):.3f}")
    print(f"Std: {np.std(vns_scores):.3f}")
    print(f"Min: {np.min(vns_scores):.3f}")
    print(f"Max: {np.max(vns_scores):.3f}")
    print("\nIoU Scores:")
    print(f"Mean: {np.mean(ious):.3f}")
    print(f"Std: {np.std(ious):.3f}")
    print(f"Min: {np.min(ious):.3f}")
    print(f"Max: {np.max(ious):.3f}")
    
    # Save detailed results
    save_detailed_results(vns_scores, ious, output_dir, correlation, p_value)
    
    return correlation, p_value

def save_detailed_results(vns_scores, ious, output_dir, correlation, p_value):
    with open(os.path.join(output_dir, 'correlation_analysis.txt'), 'w') as f:
        f.write(f"Correlation Analysis Results\n")
        f.write(f"===========================\n")
        f.write(f"Number of valid samples: {len(vns_scores)}\n")
        f.write(f"Pearson correlation coefficient: {correlation:.3f}\n")
        f.write(f"P-value: {p_value:.3e}\n")
        f.write(f"\nVNS Score statistics:\n")
        f.write(f"Mean: {np.mean(vns_scores):.3f}\n")
        f.write(f"Std: {np.std(vns_scores):.3f}\n")
        f.write(f"Min: {np.min(vns_scores):.3f}\n")
        f.write(f"Max: {np.max(vns_scores):.3f}\n")
        f.write(f"\nIoU statistics:\n")
        f.write(f"Mean: {np.mean(ious):.3f}\n")
        f.write(f"Std: {np.std(ious):.3f}\n")
        f.write(f"Min: {np.min(ious):.3f}\n")
        f.write(f"Max: {np.max(ious):.3f}\n")
        
        # Save raw data points
        f.write("\nRaw data points (VNS Score, IoU):\n")
        for vns, iou in zip(vns_scores, ious):
            f.write(f"{vns:.4f}, {iou:.4f}\n")

def get_args_parser():
    parser = argparse.ArgumentParser('Analyze Scores', add_help=False)
    parser.add_argument("--scores_file", type=str, required=True,
                        help="Path to the scores.txt file containing VNS scores and IoU")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save analysis results") 
    return parser.parse_args()

def main():
    args = get_args_parser()
    
    # Read scores from file
    vns_scores, ious = read_scores(args.scores_file)
    
    # Analyze correlation
    correlation, p_value = analyze_correlation(vns_scores, ious, args.output_dir)
    
    # Save results to text file
    with open(os.path.join(args.output_dir, 'correlation_analysis.txt'), 'w') as f:
        pass

if __name__ == "__main__":
    main()
