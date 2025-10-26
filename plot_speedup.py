#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the data
speedup_data = pd.read_csv('speedup_results.csv')
stats_data = pd.read_csv('speedup_stats.csv')

# Extract statistics
stats_dict = dict(zip(stats_data['metric'], stats_data['value']))
avg_speedup = stats_dict['avg_speedup']
min_speedup = stats_dict['min_speedup']
max_speedup = stats_dict['max_speedup']
avg_error = stats_dict['avg_error']
min_error = stats_dict['min_error']
max_error = stats_dict['max_error']
num_runs = int(stats_dict['num_runs'])

print(f"Statistics from {num_runs} runs:")
print(f"Average speedup: {avg_speedup:.4f}x")
print(f"Minimum speedup: {min_speedup:.4f}x")
print(f"Maximum speedup: {max_speedup:.4f}x")
print(f"Standard deviation speedup: {speedup_data['speedup'].std():.4f}")
print(f"Average relative error: {avg_error:g}")
print(f"Minimum relative error: {min_error:g}")
print(f"Maximum relative error: {max_error:g}")
print(f"Standard deviation error: {speedup_data['relative_error'].std():g}")

# Create subplots - simplified to 2x2 layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HybridVector vs Regular Vector: Performance and Accuracy Analysis', fontsize=16)

# 1. Time series plot
ax1.plot(speedup_data['run'], speedup_data['speedup'], 'b-', alpha=0.7, linewidth=1)
ax1.axhline(y=avg_speedup, color='r', linestyle='--', label=f'Average: {avg_speedup:.3f}x')
ax1.axhline(y=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
ax1.set_xlabel('Run Number')
ax1.set_ylabel('Speedup (x)')
ax1.set_title('Speedup Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Histogram
ax2.hist(speedup_data['speedup'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(x=avg_speedup, color='r', linestyle='--', label=f'Average: {avg_speedup:.3f}x')
ax2.axvline(x=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
ax2.set_xlabel('Speedup (x)')
ax2.set_ylabel('Frequency')
ax2.set_title('Speedup Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Box plot
box_data = [speedup_data['speedup']]
bp = ax3.boxplot(box_data, tick_labels=['Speedup'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax3.axhline(y=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
ax3.set_ylabel('Speedup (x)')
ax3.set_title('Speedup Box Plot')
ax3.grid(True, alpha=0.3)

# 3. Box plot with statistics
box_data = [speedup_data['speedup']]
bp = ax3.boxplot(box_data, tick_labels=['Speedup'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax3.axhline(y=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
ax3.set_ylabel('Speedup (x)')
ax3.set_title('Speedup Distribution Summary')
ax3.grid(True, alpha=0.3)

# Add comprehensive statistics text
stats_text = f'''SPEEDUP STATS:
Mean: {avg_speedup:.3f}x
Median: {speedup_data["speedup"].median():.3f}x
Std: {speedup_data["speedup"].std():.3f}
Min: {min_speedup:.3f}x
Max: {max_speedup:.3f}x

ERROR STATS:
Mean: {avg_error*100:.4f}%
Median: {speedup_data["relative_error"].median()*100:.4f}%
Std: {speedup_data["relative_error"].std()*100:.4f}%
Min: {min_error*100:.4f}%
Max: {max_error*100:.4f}%'''

ax3.text(1.02, 0.98, stats_text, transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=9)

# 4. Speedup vs Error scatter plot
ax4.scatter(speedup_data['speedup'], speedup_data['relative_error']*100, alpha=0.6, s=30, 
           c=range(len(speedup_data)), cmap='viridis')
ax4.axvline(x=1.0, color='k', linestyle='-', alpha=0.5, label='No speedup (1.0x)')
ax4.axhline(y=avg_error*100, color='r', linestyle='--', alpha=0.7, label=f'Avg error: {avg_error*100:.4f}%')
ax4.set_xlabel('Speedup (x)')
ax4.set_ylabel('Relative Error (%)')
ax4.set_title('Speedup vs Accuracy Trade-off')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add correlation coefficient
correlation = speedup_data['speedup'].corr(speedup_data['relative_error'])
ax4.text(0.02, 0.98, f'Correlation: {correlation:.3f}\nRuns > 1.0x: {(speedup_data["speedup"] > 1.0).sum()}/{num_runs}', 
         transform=ax4.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved as speedup_analysis.png")
# plt.show()  # Commented out since running non-interactively

# Additional statistics
print(f"\nAdditional Statistics:")
print(f"Median speedup: {speedup_data['speedup'].median():.4f}x")
print(f"25th percentile speedup: {speedup_data['speedup'].quantile(0.25):.4f}x")
print(f"75th percentile speedup: {speedup_data['speedup'].quantile(0.75):.4f}x")
print(f"Runs with speedup > 1.0x: {(speedup_data['speedup'] > 1.0).sum()}/{num_runs} ({(speedup_data['speedup'] > 1.0).mean()*100:.1f}%)")
print(f"Runs with speedup > 1.5x: {(speedup_data['speedup'] > 1.5).sum()}/{num_runs} ({(speedup_data['speedup'] > 1.5).mean()*100:.1f}%)")
print(f"\nError Statistics:")
print(f"Median relative error: {speedup_data['relative_error'].median()*100:.6f}%")
print(f"25th percentile error: {speedup_data['relative_error'].quantile(0.25)*100:.6f}%")
print(f"75th percentile error: {speedup_data['relative_error'].quantile(0.75)*100:.6f}%")