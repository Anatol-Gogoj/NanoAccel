#!/usr/bin/env python3
"""
Plot RSS (root-sum-square) acceleration magnitude from IMU CSV recording.
Saves to PNG (no GUI required).

Usage: python3 plot_accel_rss.py path/to/imu_data.csv
       python3 plot_accel_rss.py path/to/imu_data.csv output.png
"""

import sys

# Use non-interactive backend (no GUI needed)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python3 plot_accel_rss.py <csv_file> [output.png]")
    sys.exit(1)

csv_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else csv_file.replace('.csv', '_accel_rss.png')

print(f"Reading {csv_file}...")
df = pd.read_csv(csv_file)

required_cols = ['timestamp_s', 'ax_m_s2', 'ay_m_s2', 'az_m_s2']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"Computing RSS acceleration for {len(df)} samples...")
df['accel_rss_m_s2'] = np.sqrt(
    df['ax_m_s2']**2 +
    df['ay_m_s2']**2 +
    df['az_m_s2']**2
)

print("Plotting RSS magnitude...")
fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

ax.plot(
    df['timestamp_s'],
    df['accel_rss_m_s2'],
    label='Acceleration RSS',
    linewidth=0.4,
    alpha=0.9
)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration RSS (m/s²)')
ax.set_title(f'IMU Acceleration RSS — {csv_file}')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_file)
print(f"Saved to {output_file}")
