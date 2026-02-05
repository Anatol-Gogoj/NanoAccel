#!/usr/bin/env python3
"""
Binary IMU stream receiver for Arduino Nano RP2040 Connect

Decodes the 14-byte binary packets, prints live data, and saves to CSV.

Usage:
    python3 receive_imu.py /dev/ttyACM0 /path/to/output/dir
    python3 receive_imu.py /dev/ttyACM0                      # saves to current dir
    python3 receive_imu.py                                   # default port, current dir
"""

import serial
import struct
import sys
import time
import os
import csv
from datetime import datetime
from collections import deque
from pathlib import Path

# Packet format
SYNC_BYTES = bytes([0xAA, 0x55])
PACKET_SIZE = 14

# Scale factors (from LSM6DSOX datasheet)
ACCEL_SCALE = 0.122e-3 * 9.81  # mg/LSB → m/s² (±4g range)
GYRO_SCALE = 70e-3 * (3.14159 / 180)  # mdps/LSB → rad/s (±2000dps range)

# CSV write buffer size (trades memory for fewer disk writes)
CSV_BUFFER_SIZE = 1000


def find_sync(ser, buffer):
    """Find sync bytes in stream, return aligned buffer."""
    while True:
        if len(buffer) >= 2:
            idx = buffer.find(SYNC_BYTES)
            if idx >= 0:
                return buffer[idx:]
            buffer = buffer[-1:]  # Keep last byte in case it's 0xAA
        buffer += ser.read(max(1, ser.in_waiting))


def main():
    # Parse arguments
    port = '/dev/ttyACM0'
    output_dir = '.'
    
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = output_path / f'imu_data_{timestamp}.csv'
    
    print(f"Opening {port} at 2000000 baud...")
    print(f"Saving to: {csv_filename}")
    
    ser = serial.Serial(port, 2000000, timeout=0.1)
    time.sleep(0.1)
    ser.reset_input_buffer()
    
    buffer = b''
    packet_count = 0
    start_time = time.time()
    rate_window = deque(maxlen=1000)
    csv_buffer = []
    
    # Open CSV file
    csvfile = open(csv_filename, 'w', newline='')
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow([
        'timestamp_s',      # Time since recording start (seconds)
        'gx_rad_s',         # Gyro X (rad/s)
        'gy_rad_s',         # Gyro Y (rad/s)
        'gz_rad_s',         # Gyro Z (rad/s)
        'ax_m_s2',          # Accel X (m/s²)
        'ay_m_s2',          # Accel Y (m/s²)
        'az_m_s2',          # Accel Z (m/s²)
        'gx_raw',           # Raw gyro X (int16)
        'gy_raw',           # Raw gyro Y (int16)
        'gz_raw',           # Raw gyro Z (int16)
        'ax_raw',           # Raw accel X (int16)
        'ay_raw',           # Raw accel Y (int16)
        'az_raw',           # Raw accel Z (int16)
    ])
    
    print("\nStreaming (Ctrl+C to stop)...")
    print("      Gx       Gy       Gz  |      Ax       Ay       Az  | Rate  | Saved")
    print("   rad/s    rad/s    rad/s  |    m/s²    m/s²    m/s²  |  Hz   |")
    print("-" * 80)
    
    try:
        while True:
            # Read available data
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting)
            
            # Process complete packets
            while len(buffer) >= PACKET_SIZE:
                # Check sync
                if buffer[:2] != SYNC_BYTES:
                    buffer = find_sync(ser, buffer)
                    continue
                
                # Extract packet
                packet = buffer[:PACKET_SIZE]
                buffer = buffer[PACKET_SIZE:]
                
                # Timestamp relative to start
                now = time.time()
                t = now - start_time
                
                # Decode (little-endian int16)
                gx, gy, gz, ax, ay, az = struct.unpack('<6h', packet[2:])
                
                # Scale to physical units
                gx_rad = gx * GYRO_SCALE
                gy_rad = gy * GYRO_SCALE
                gz_rad = gz * GYRO_SCALE
                ax_ms2 = ax * ACCEL_SCALE
                ay_ms2 = ay * ACCEL_SCALE
                az_ms2 = az * ACCEL_SCALE
                
                # Buffer for CSV (write in batches for performance)
                csv_buffer.append([
                    f'{t:.6f}',
                    f'{gx_rad:.6f}', f'{gy_rad:.6f}', f'{gz_rad:.6f}',
                    f'{ax_ms2:.6f}', f'{ay_ms2:.6f}', f'{az_ms2:.6f}',
                    gx, gy, gz, ax, ay, az
                ])
                
                # Flush CSV buffer periodically
                if len(csv_buffer) >= CSV_BUFFER_SIZE:
                    writer.writerows(csv_buffer)
                    csv_buffer.clear()
                
                # Track rate
                rate_window.append(now)
                packet_count += 1
                
                # Print every 500 packets (~10 Hz display update at 5kHz)
                if packet_count % 500 == 0:
                    if len(rate_window) > 1:
                        rate = len(rate_window) / (rate_window[-1] - rate_window[0])
                    else:
                        rate = 0
                    print(f"{gx_rad:8.3f} {gy_rad:8.3f} {gz_rad:8.3f}  | "
                          f"{ax_ms2:7.2f} {ay_ms2:7.2f} {az_ms2:7.2f}  | "
                          f"{rate:5.0f} | {packet_count:>7}")
            
            # Small sleep to avoid CPU spin when buffer is empty
            if not ser.in_waiting:
                time.sleep(0.0001)
                
    except KeyboardInterrupt:
        pass
    finally:
        # Flush remaining buffer
        if csv_buffer:
            writer.writerows(csv_buffer)
        csvfile.close()
        ser.close()
        
        elapsed = time.time() - start_time
        print(f"\n\nRecording complete:")
        print(f"  Packets:  {packet_count}")
        print(f"  Duration: {elapsed:.1f} s")
        print(f"  Avg rate: {packet_count/elapsed:.1f} Hz")
        print(f"  Saved to: {csv_filename}")
        print(f"  File size: {os.path.getsize(csv_filename) / 1e6:.2f} MB")


if __name__ == '__main__':
    main()