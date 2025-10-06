import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_scaling_results():
    # Loading data
    strong_df = pd.read_csv('strong_scaling_data.csv')
    weak_df = pd.read_csv('weak_scaling_data.csv')

    print("*******************STRONG SCALING DATA*******************:")
    print(strong_df.to_string(index=False))
    print("\n")
    print("WEAK SCALING DATA:")
    print(weak_df.to_string(index=False))
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    fig.suptitle('Scaling Analysis of OpenMP Matrix Multiplication\n', fontsize=16, fontweight='bold')
    
    # Strong scaling plots [1st row]
    # S
    axes[0, 0].plot(strong_df['P'], strong_df['S'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of threads')
    axes[0, 0].set_ylabel('Speedup')
    axes[0, 0].set_title('Strong scaling: speedup')
    axes[0, 0].grid(True, alpha=0.3)
    # E
    axes[0, 1].plot(strong_df['P'], strong_df['E'], 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of threads')
    axes[0, 1].set_ylabel('Efficiency')
    axes[0, 1].set_title('Strong scaling: efficiency')
    axes[0, 1].grid(True, alpha=0.3)
    # Tp
    axes[0, 2].plot(strong_df['P'], strong_df['Tp'], 'mo-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Number of threads')
    axes[0, 2].set_ylabel('Parallel time (sec)')
    axes[0, 2].set_title('Strong scaling: parallel time')
    axes[0, 2].grid(True, alpha=0.3)
    # W
    axes[0, 3].plot(strong_df['P'], strong_df['W'], 'co-', linewidth=2, markersize=8, label='Actual')
    axes[0, 3].axhline(y=strong_df['W'].iloc[0], color='r', linestyle='--', label='Theoretical (constant)')
    axes[0, 3].set_xlabel('Number of Threads')
    axes[0, 3].set_ylabel('Work')
    axes[0, 3].set_title('Strong scaling: work')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    axes[0, 3].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Weak scaling plots [2nd row]
    # S
    axes[1, 0].plot(weak_df['P'], weak_df['S'], 'bo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of threads')
    axes[1, 0].set_ylabel('Speedup')
    axes[1, 0].set_title('Weak scaling: speedup')
    axes[1, 0].grid(True, alpha=0.3)
    # E
    axes[1, 1].plot(weak_df['P'], weak_df['E'], 'ko-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of threads')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].set_title('Weak scaling: efficiency')
    axes[1, 1].grid(True, alpha=0.3)    
    # Tp
    axes[1, 2].plot(weak_df['P'], weak_df['Tp'], 'co-', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Number of threads')
    axes[1, 2].set_ylabel('Parallel time (sec)')
    axes[1, 2].set_title('Weak scaling: parallel time')
    axes[1, 2].grid(True, alpha=0.3)
    # W
    W_per_thread = weak_df['W'] / weak_df['P']
    axes[1, 3].plot(weak_df['P'], W_per_thread, 'ro-', linewidth=2, markersize=8, label='Actual W/P')
    ideal_W_per_thread = W_per_thread.iloc[0]
    axes[1, 3].axhline(y=ideal_W_per_thread, color='r', linestyle='--', label='Theoretical (constant)')
    axes[1, 3].set_xlabel('Number of threads')
    axes[1, 3].set_ylabel('W/P (operations per thread)')
    axes[1, 3].set_title('Weak scaling: work per thread (W/P = constant)')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_ylim(bottom=3)

    axes[1, 3].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')

    print("\nAnalytical plot succefully saved!\n")

if __name__ == "__main__":
    plot_scaling_results()