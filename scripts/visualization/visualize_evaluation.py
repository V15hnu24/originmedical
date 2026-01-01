"""
Visualize model evaluation results with comprehensive plots and analysis

Usage:
    python visualize_evaluation.py --results results/coordinate_evaluation
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))


def create_visualization_dashboard(results_dir):
    """Create a comprehensive visualization dashboard of model performance."""
    
    results_dir = Path(results_dir)
    
    # Load results
    results_csv = results_dir / "evaluation_results.csv"
    summary_csv = results_dir / "evaluation_summary.csv"
    
    if not results_csv.exists():
        print(f"Error: Results file not found at {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} evaluation results")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. MRE Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['mre'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(df['mre'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["mre"].mean():.2f}')
    ax1.axvline(df['mre'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["mre"].median():.2f}')
    ax1.set_xlabel('Mean Radial Error (pixels)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Distribution of MRE', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Per-Landmark Error Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    landmark_errors = [df['landmark_0_error'], df['landmark_1_error'], 
                       df['landmark_2_error'], df['landmark_3_error']]
    bp = ax2.boxplot(landmark_errors, labels=['LM0', 'LM1', 'LM2', 'LM3'],
                     patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightblue', 'lightcoral', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Error (pixels)', fontsize=10)
    ax2.set_title('Per-Landmark Error Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. BPD vs OFD Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    bpd_ofd_data = [df['bpd_mre'], df['ofd_mre']]
    bp2 = ax3.boxplot(bpd_ofd_data, labels=['BPD (LM0,1)', 'OFD (LM2,3)'],
                      patch_artist=True, showmeans=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Mean Radial Error (pixels)', fontsize=10)
    ax3.set_title('BPD vs OFD Landmark Errors', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Cumulative Error Distribution
    ax4 = fig.add_subplot(gs[0, 3])
    sorted_mre = np.sort(df['mre'])
    cumulative = np.arange(1, len(sorted_mre) + 1) / len(sorted_mre) * 100
    ax4.plot(sorted_mre, cumulative, linewidth=2, color='darkblue')
    ax4.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax4.axhline(75, color='orange', linestyle='--', alpha=0.5, label='75%')
    ax4.axhline(90, color='green', linestyle='--', alpha=0.5, label='90%')
    ax4.set_xlabel('Error (pixels)', fontsize=10)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=10)
    ax4.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Error Heatmap (correlation between landmarks)
    ax5 = fig.add_subplot(gs[1, 0])
    landmark_corr = df[['landmark_0_error', 'landmark_1_error', 
                        'landmark_2_error', 'landmark_3_error']].corr()
    sns.heatmap(landmark_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax5, cbar_kws={'label': 'Correlation'})
    ax5.set_title('Landmark Error Correlation', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(['LM0', 'LM1', 'LM2', 'LM3'])
    ax5.set_yticklabels(['LM0', 'LM1', 'LM2', 'LM3'])
    
    # 6. Best vs Worst Cases
    ax6 = fig.add_subplot(gs[1, 1])
    n_show = 10
    best_indices = df['mre'].nsmallest(n_show).index
    worst_indices = df['mre'].nlargest(n_show).index
    
    x = np.arange(n_show)
    width = 0.35
    ax6.bar(x - width/2, df.loc[best_indices, 'mre'], width, label='Best 10', color='green', alpha=0.7)
    ax6.bar(x + width/2, df.loc[worst_indices, 'mre'], width, label='Worst 10', color='red', alpha=0.7)
    ax6.set_xlabel('Sample Rank', fontsize=10)
    ax6.set_ylabel('MRE (pixels)', fontsize=10)
    ax6.set_title('Best vs Worst Predictions', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3, axis='y')
    
    # 7. Error Statistics Table
    ax7 = fig.add_subplot(gs[1, 2:])
    ax7.axis('off')
    
    stats_data = [
        ['Metric', 'Value'],
        ['Mean MRE', f'{df["mre"].mean():.2f} ± {df["mre"].std():.2f} px'],
        ['Median MRE', f'{df["mre"].median():.2f} px'],
        ['Min Error', f'{df["mre"].min():.2f} px'],
        ['Max Error', f'{df["mre"].max():.2f} px'],
        ['BPD Mean', f'{df["bpd_mre"].mean():.2f} ± {df["bpd_mre"].std():.2f} px'],
        ['OFD Mean', f'{df["ofd_mre"].mean():.2f} ± {df["ofd_mre"].std():.2f} px'],
        ['LM0 Mean', f'{df["landmark_0_error"].mean():.2f} px'],
        ['LM1 Mean', f'{df["landmark_1_error"].mean():.2f} px'],
        ['LM2 Mean', f'{df["landmark_2_error"].mean():.2f} px'],
        ['LM3 Mean', f'{df["landmark_3_error"].mean():.2f} px'],
    ]
    
    table = ax7.table(cellText=stats_data, cellLoc='left', loc='center',
                      colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax7.set_title('Performance Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # 8. Accuracy at Thresholds
    ax8 = fig.add_subplot(gs[2, 0])
    thresholds = [50, 100, 150, 200, 250, 300]
    accuracies = [(df['mre'] < t).mean() * 100 for t in thresholds]
    
    ax8.plot(thresholds, accuracies, marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax8.fill_between(thresholds, accuracies, alpha=0.3, color='green')
    ax8.set_xlabel('Error Threshold (pixels)', fontsize=10)
    ax8.set_ylabel('Accuracy (%)', fontsize=10)
    ax8.set_title('Accuracy at Different Thresholds', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3)
    ax8.set_ylim([0, 105])
    
    # Add percentage labels
    for t, a in zip(thresholds, accuracies):
        ax8.text(t, a + 2, f'{a:.1f}%', ha='center', fontsize=8)
    
    # 9. Per-landmark comparison
    ax9 = fig.add_subplot(gs[2, 1])
    landmarks = ['LM0\n(OFD1)', 'LM1\n(OFD2)', 'LM2\n(BPD1)', 'LM3\n(BPD2)']
    means = [df['landmark_0_error'].mean(), df['landmark_1_error'].mean(),
             df['landmark_2_error'].mean(), df['landmark_3_error'].mean()]
    stds = [df['landmark_0_error'].std(), df['landmark_1_error'].std(),
            df['landmark_2_error'].std(), df['landmark_3_error'].std()]
    
    colors = ['skyblue', 'lightblue', 'lightcoral', 'salmon']
    bars = ax9.bar(landmarks, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax9.set_ylabel('Mean Error (pixels)', fontsize=10)
    ax9.set_title('Mean Error per Landmark', fontsize=12, fontweight='bold')
    ax9.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 10. Violin plot for detailed distribution
    ax10 = fig.add_subplot(gs[2, 2:])
    data_to_plot = [df['landmark_0_error'], df['landmark_1_error'],
                    df['landmark_2_error'], df['landmark_3_error']]
    parts = ax10.violinplot(data_to_plot, positions=[0, 1, 2, 3], 
                            showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        if i < 2:
            pc.set_facecolor('lightblue')
        else:
            pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    ax10.set_xticks([0, 1, 2, 3])
    ax10.set_xticklabels(['LM0 (OFD1)', 'LM1 (OFD2)', 'LM2 (BPD1)', 'LM3 (BPD2)'])
    ax10.set_ylabel('Error (pixels)', fontsize=10)
    ax10.set_title('Error Distribution by Landmark (Violin Plot)', fontsize=12, fontweight='bold')
    ax10.grid(alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle('Coordinate Regression Model - Validation Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = results_dir / "performance_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Dashboard saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal samples evaluated: {len(df)}")
    print(f"\nMean Radial Error: {df['mre'].mean():.2f} ± {df['mre'].std():.2f} pixels")
    print(f"Median Error: {df['mre'].median():.2f} pixels")
    print(f"Best case: {df['mre'].min():.2f} pixels")
    print(f"Worst case: {df['mre'].max():.2f} pixels")
    
    print(f"\nPer-measurement errors:")
    print(f"  BPD: {df['bpd_mre'].mean():.2f} ± {df['bpd_mre'].std():.2f} pixels")
    print(f"  OFD: {df['ofd_mre'].mean():.2f} ± {df['ofd_mre'].std():.2f} pixels")
    
    print(f"\nAccuracy at thresholds:")
    for threshold in [50, 100, 150, 200]:
        acc = (df['mre'] < threshold).mean() * 100
        print(f"  < {threshold:3d} pixels: {acc:5.1f}%")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    
    parser.add_argument('--results', type=str, 
                       default='results/coordinate_evaluation',
                       help='Path to evaluation results directory')
    
    args = parser.parse_args()
    
    # Check if results exist
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        print("\nAvailable results directories:")
        results_base = Path("results")
        if results_base.exists():
            for d in results_base.iterdir():
                if d.is_dir():
                    print(f"  - {d}")
        return
    
    # Create visualizations
    create_visualization_dashboard(results_dir)


if __name__ == '__main__':
    main()
