#!/usr/bin/env python3
"""
Generate:
  1) Welch PSD of RSS acceleration (log-frequency x-axis) -> PNG
  2) "Waterfall PSD" (time vs frequency spectrogram of PSD) -> PNG

Input CSV columns expected:
  timestamp_s, ax_m_s2, ay_m_s2, az_m_s2

Usage:
  python3 plot_accel_psd_waterfall.py path/to/imu_data.csv
  python3 plot_accel_psd_waterfall.py path/to/imu_data.csv output_prefix

Outputs (by default):
  <csv>_psd.png
  <csv>_waterfall_psd.png

Notes:
- Assumes roughly uniform sampling. Sampling rate is estimated from median dt.
- Uses RSS acceleration: sqrt(ax^2 + ay^2 + az^2), mean-removed.
- PSD units: (m/s^2)^2/Hz. Plotted as dB: 10*log10(PSD).
"""

import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def EstimateSamplingRateHz(TimeS: np.ndarray) -> tuple[float, float, float]:
    if TimeS.size < 3:
        raise ValueError("Need at least 3 samples to estimate sampling rate.")
    Dt = np.diff(TimeS)
    Dt = Dt[np.isfinite(Dt)]
    if Dt.size == 0:
        raise ValueError("Non-finite timestamp differences found.")
    DtMed = float(np.median(Dt))
    DtMad = float(np.median(np.abs(Dt - DtMed)))
    if DtMed <= 0:
        raise ValueError("Non-positive median dt; timestamps may be invalid.")
    FsHz = 1.0 / DtMed
    return FsHz, DtMed, DtMad


def NextPow2Leq(N: int) -> int:
    if N < 1:
        return 1
    P = 1
    while (P << 1) <= N:
        P <<= 1
    return P


def ComputeWelchPsdOneSided(
    Signal: np.ndarray,
    FsHz: float,
    NPerSeg: int,
    OverlapFrac: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (FreqHz, Pxx) where:
      - FreqHz is one-sided rFFT frequency vector
      - Pxx is PSD in units of Signal^2 / Hz
    """
    if Signal.ndim != 1:
        raise ValueError("Signal must be 1D.")
    N = Signal.size
    if N < NPerSeg:
        raise ValueError("Signal too short for requested NPerSeg.")

    if not (0.0 <= OverlapFrac < 1.0):
        raise ValueError("OverlapFrac must be in [0, 1).")

    Hop = int(round(NPerSeg * (1.0 - OverlapFrac)))
    Hop = max(Hop, 1)

    Window = np.hanning(NPerSeg)
    WinPow = np.sum(Window * Window)  # sum(w^2)
    if WinPow <= 0:
        raise ValueError("Window power is non-positive.")

    # Window normalization for PSD
    # U = (1/NPerSeg) * sum(w^2)
    U = WinPow / float(NPerSeg)

    FreqHz = np.fft.rfftfreq(NPerSeg, d=1.0 / FsHz)
    PxxAcc = np.zeros(FreqHz.size, dtype=float)
    SegCount = 0

    Start = 0
    while Start + NPerSeg <= N:
        X = Signal[Start:Start + NPerSeg]
        Xw = X * Window
        Xf = np.fft.rfft(Xw)
        # Two-sided periodogram density mapped to one-sided via rFFT:
        # Pxx = (1/(Fs * NPerSeg * U)) * |Xf|^2
        Pxx = (np.abs(Xf) ** 2) / (FsHz * float(NPerSeg) * U)

        # Convert to one-sided PSD: double non-DC and non-Nyquist bins
        if Pxx.size > 2:
            Pxx[1:-1] *= 2.0
        elif Pxx.size == 2:
            # Only DC and Nyquist (rare for very small NPerSeg)
            pass

        PxxAcc += Pxx
        SegCount += 1
        Start += Hop

    if SegCount == 0:
        raise ValueError("No segments processed for Welch PSD.")

    PxxAvg = PxxAcc / float(SegCount)
    return FreqHz, PxxAvg


def ComputeWaterfallPsdDb(
    Signal: np.ndarray,
    TimeS: np.ndarray,
    FsHz: float,
    NPerSeg: int,
    OverlapFrac: float = 0.5,
    FloorDb: float = -200.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (FreqHz, TimeCentersS, PsdDbMatrix)

    PsdDbMatrix shape: (num_times, num_freqs)
    x-axis intended: frequency (log)
    y-axis intended: time (s)
    """
    if Signal.ndim != 1 or TimeS.ndim != 1:
        raise ValueError("Signal and TimeS must be 1D.")
    if Signal.size != TimeS.size:
        raise ValueError("Signal and TimeS must have the same length.")
    N = Signal.size
    if N < NPerSeg:
        raise ValueError("Signal too short for requested NPerSeg.")

    if not (0.0 <= OverlapFrac < 1.0):
        raise ValueError("OverlapFrac must be in [0, 1).")

    Hop = int(round(NPerSeg * (1.0 - OverlapFrac)))
    Hop = max(Hop, 1)

    Window = np.hanning(NPerSeg)
    WinPow = np.sum(Window * Window)
    U = WinPow / float(NPerSeg)

    FreqHz = np.fft.rfftfreq(NPerSeg, d=1.0 / FsHz)

    TimeCenters = []
    PsdRows = []

    Start = 0
    while Start + NPerSeg <= N:
        X = Signal[Start:Start + NPerSeg]
        Xw = X * Window
        Xf = np.fft.rfft(Xw)

        Pxx = (np.abs(Xf) ** 2) / (FsHz * float(NPerSeg) * U)
        if Pxx.size > 2:
            Pxx[1:-1] *= 2.0

        PxxDb = 10.0 * np.log10(np.maximum(Pxx, 1e-300))
        PxxDb = np.maximum(PxxDb, FloorDb)

        TCenter = float(TimeS[Start:Start + NPerSeg].mean())
        TimeCenters.append(TCenter)
        PsdRows.append(PxxDb)

        Start += Hop

    if len(TimeCenters) == 0:
        raise ValueError("No segments processed for waterfall PSD.")

    TimeCentersS = np.array(TimeCenters, dtype=float)
    PsdDbMatrix = np.vstack(PsdRows)
    return FreqHz, TimeCentersS, PsdDbMatrix


def Main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 plot_accel_psd_waterfall.py <csv_file> [output_prefix]")
        return 1

    CsvFile = sys.argv[1]
    OutputPrefix = sys.argv[2] if len(sys.argv) > 2 else CsvFile.replace(".csv", "")

    PsdPng = f"{OutputPrefix}_psd.png"
    WaterfallPng = f"{OutputPrefix}_waterfall_psd.png"

    print(f"Reading {CsvFile}...")
    Df = pd.read_csv(CsvFile)

    RequiredCols = ["timestamp_s", "ax_m_s2", "ay_m_s2", "az_m_s2"]
    Missing = [c for c in RequiredCols if c not in Df.columns]
    if Missing:
        raise ValueError(f"Missing required columns: {Missing}")

    TimeS = Df["timestamp_s"].to_numpy(dtype=float)
    Ax = Df["ax_m_s2"].to_numpy(dtype=float)
    Ay = Df["ay_m_s2"].to_numpy(dtype=float)
    Az = Df["az_m_s2"].to_numpy(dtype=float)

    GoodMask = np.isfinite(TimeS) & np.isfinite(Ax) & np.isfinite(Ay) & np.isfinite(Az)
    TimeS = TimeS[GoodMask]
    Ax = Ax[GoodMask]
    Ay = Ay[GoodMask]
    Az = Az[GoodMask]

    if TimeS.size < 32:
        raise ValueError("Not enough valid samples (need at least ~32).")

    SortIdx = np.argsort(TimeS)
    TimeS = TimeS[SortIdx]
    Ax = Ax[SortIdx]
    Ay = Ay[SortIdx]
    Az = Az[SortIdx]

    FsHz, DtMed, DtMad = EstimateSamplingRateHz(TimeS)
    print(f"Estimated Fs = {FsHz:.3f} Hz (median dt = {DtMed:.6f} s, median abs jitter = {DtMad:.6f} s)")

    if (DtMad / DtMed) > 0.02:
        print("WARNING: Timestamp jitter looks >2% of dt. PSD/spectrogram may be smeared.")
        print("         If this matters, resample to a uniform time grid before FFT/PSD.")

    AccelRss = np.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    AccelRss = AccelRss - float(np.mean(AccelRss))

    # Choose a reasonable NPerSeg based on data length (power of 2, not too big)
    N = AccelRss.size
    Candidate = NextPow2Leq(N)
    # Cap to something reasonable for plots
    NPerSeg = min(Candidate, 8192)
    # Also avoid tiny segments
    NPerSeg = max(NPerSeg, 256)
    if NPerSeg > N:
        NPerSeg = NextPow2Leq(N)

    OverlapFrac = 0.5
    print(f"Using NPerSeg = {NPerSeg}, overlap = {int(OverlapFrac * 100)}%")

    # --- Welch PSD plot (log-x) ---
    FreqHz, Pxx = ComputeWelchPsdOneSided(AccelRss, FsHz, NPerSeg=NPerSeg, OverlapFrac=OverlapFrac)

    # Avoid DC on log axis
    FreqMask = FreqHz > 0.0
    FreqHzPlot = FreqHz[FreqMask]
    PxxPlot = Pxx[FreqMask]
    PxxDb = 10.0 * np.log10(np.maximum(PxxPlot, 1e-300))

    print("Plotting Welch PSD...")
    fig1, ax1 = plt.subplots(figsize=(14, 6), dpi=150)
    ax1.plot(FreqHzPlot, PxxDb, linewidth=0.8, alpha=0.9, label="Welch PSD (RSS)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Frequency (Hz) [log]")
    ax1.set_ylabel("PSD (dB re (m/s^2)^2/Hz)")
    ax1.set_title(f"IMU RSS Welch PSD — {CsvFile}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.savefig(PsdPng)
    print(f"Saved PSD plot to {PsdPng}")

    # --- Waterfall PSD (spectrogram) ---
    print("Computing waterfall PSD...")
    FreqHzW, TimeCentersS, PsdDbMatrix = ComputeWaterfallPsdDb(
        Signal=AccelRss,
        TimeS=TimeS,
        FsHz=FsHz,
        NPerSeg=NPerSeg,
        OverlapFrac=OverlapFrac,
        FloorDb=-200.0
    )

    # Drop DC for log-x
    FreqMaskW = FreqHzW > 0.0
    FreqHzWPlot = FreqHzW[FreqMaskW]
    PsdDbMatrixPlot = PsdDbMatrix[:, FreqMaskW]

    # --- Waterfall PSD (FLIPPED AXES: time on x, frequency on y) ---
    print("Plotting waterfall PSD (time on x, frequency on y)...")

    fig2, ax2 = plt.subplots(figsize=(14, 8), dpi=150)

    # ---- Construct frequency edges (log-spaced-friendly) ----
    FreqEdges = np.zeros(FreqHzWPlot.size + 1, dtype=float)
    FreqEdges[1:-1] = 0.5 * (FreqHzWPlot[1:] + FreqHzWPlot[:-1])

    if FreqHzWPlot.size > 1:
        FreqEdges[0] = FreqHzWPlot[0] * (FreqHzWPlot[0] / FreqHzWPlot[1])
        FreqEdges[-1] = FreqHzWPlot[-1] * (FreqHzWPlot[-1] / FreqHzWPlot[-2])
    else:
        FreqEdges[0] = FreqHzWPlot[0] * 0.5
        FreqEdges[-1] = FreqHzWPlot[0] * 2.0

    # ---- Construct time edges ----
    TimeEdges = np.zeros(TimeCentersS.size + 1, dtype=float)
    if TimeCentersS.size > 1:
        DtCenters = np.diff(TimeCentersS)
        DtC = float(np.median(DtCenters))
    else:
        DtC = float(NPerSeg / FsHz)

    TimeEdges[1:-1] = 0.5 * (TimeCentersS[1:] + TimeCentersS[:-1])
    TimeEdges[0] = TimeCentersS[0] - 0.5 * DtC
    TimeEdges[-1] = TimeCentersS[-1] + 0.5 * DtC

    # ---- NOTE THE TRANSPOSE HERE ----
    # PsdDbMatrixPlot shape: (time, freq)
    Mesh = ax2.pcolormesh(
        TimeEdges,
        FreqEdges,
        PsdDbMatrixPlot.T,
        shading="auto"
    )

    ax2.set_yscale("log")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz) [log]")
    ax2.set_title(f"IMU RSS Waterfall PSD — {CsvFile}")

    # ---- Aspect ratio tuning ----
    # This keeps cells closer to square visually
    ax2.set_aspect("auto")

    ax2.grid(True, which="both", alpha=0.2)

    Cbar = fig2.colorbar(Mesh, ax=ax2)
    Cbar.set_label("PSD (dB re (m/s^2)^2/Hz)")

    plt.tight_layout()
    plt.savefig(WaterfallPng)
    print(f"Saved flipped-axis waterfall PSD plot to {WaterfallPng}")

    return 0


if __name__ == "__main__":
    raise SystemExit(Main())


