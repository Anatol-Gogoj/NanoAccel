#!/usr/bin/env python3
"""
Plot accelerometer data from IMU CSV recording.
Saves to PNG (no GUI required).

Usage: python3 plot_accel.py path/to/imu_data.csv
       python3 plot_accel.py path/to/imu_data.csv output.png
"""

import sys

# Use non-interactive backend (no GUI needed)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 plot_accel.py <csv_file> [output.png]")
    sys.exit(1)

csv_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else csv_file.replace('.csv', '_accel.png')

print(f"Reading {csv_file}...")
df = pd.read_csv(csv_file)

print(f"Plotting {len(df)} samples...")
fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

ax.plot(df['timestamp_s'], df['ax_m_s2'], label='Ax', linewidth=0.3, alpha=0.8)
ax.plot(df['timestamp_s'], df['ay_m_s2'], label='Ay', linewidth=0.3, alpha=0.8)
ax.plot(df['timestamp_s'], df['az_m_s2'], label='Az', linewidth=0.3, alpha=0.8)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration (m/s²)')
ax.set_title(f'IMU Accelerometer Data — {csv_file}')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_file)
print(f"Saved to {output_file}")