import argparse
import os
import subprocess
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

OVERSAMPLE = 16

def frac_delay(signal: np.ndarray, delay: float, taps: int = 81) -> np.ndarray:
    n = np.arange(-(taps//2), taps//2 + 1)
    h = np.sinc(n - delay)
    w = np.hamming(len(h))
    h *= w
    s = np.sum(h)
    if abs(s) < 1e-12:
        return signal.copy()
    h /= s
    return np.convolve(signal, h, mode='same')

def sharp_impulse_response(sps:int=16,
                           main_spike=1.0,
                           ring_amp=0.8,
                           ring_freq=3.5,  
                           ring_tau=0.06, 
                           tail_amp=0.05,
                           tail_tau=0.5,
                           span_symbols=6):
    L = int(span_symbols * sps)
    t = np.arange(0, L) / float(sps)
    spike = np.exp(-((t* sps) ** 2) / (2.0 * (0.6)**2))
    spike /= np.max(spike) if spike.max() != 0 else 1.0
    spike *= main_spike
    ring = ring_amp * np.exp(-t / (ring_tau + 1e-12)) * np.cos(2 * np.pi * ring_freq * t)
    tail = tail_amp * np.exp(-t / (tail_tau + 1e-12))
    h = spike + ring + tail
    h /= (np.sum(np.abs(h)) + 1e-12)
    return h

def memory_poly(x: np.ndarray, b1: float=0.06, b2: float=-0.02):
    x_del = np.concatenate(([0.0], x[:-1]))
    return x + b1 * x * x_del + b2 * x_del**2

def make_case(M:int, K:int, sps:int=OVERSAMPLE,
                    snr_db:float=28.0,
                    spike_strength:float=3.0,
                    ring_amp:float=2.8,
                    ring_freq:float=5.5,
                    ring_tau:float=0.05,
                    tail_amp:float=0.05,
                    pre_emph:float=0.6,
                    asymmetry=0.52,
                    mem_params=(0.045, -0.02),
                    frac_timing_mean:float=0.70,
                    frac_timing_jitter:float=0.02,
                    target_peak:float=0.32,
                    seed=None):
    rng = np.random.default_rng(seed)
    levels = np.array([-3, -1, 1, 3], dtype=int)
    dig1 = rng.choice(levels, size=M)
    dig2 = rng.choice(levels, size=K)
    h = sharp_impulse_response(sps=sps,
                               main_spike=spike_strength,
                               ring_amp=ring_amp,
                               ring_freq=ring_freq,
                               ring_tau=ring_tau,
                               tail_amp=tail_amp,
                               tail_tau=0.35,
                               span_symbols=6)
    def tx_waveform(dig):
        N = len(dig) * sps
        up = np.zeros(N, dtype=float)
        up[::sps] = dig.astype(float) / 3.0
        if pre_emph > 0:
            diffs = np.diff(np.concatenate(([up[0]], up[::sps])))
            for i_sym, d in enumerate(diffs):
                if i_sym*sps < N:
                    idx = i_sym * sps
                    up[idx] += d * pre_emph * 0.9
                    if idx + 1 < N:
                        up[idx+1] += d * pre_emph * 0.4
        tx = np.convolve(up, h, mode='same')
        if asymmetry != 0:
            syms = dig.astype(float) / 3.0
            for i in range(1, len(syms)):
                prev = syms[i-1]; cur = syms[i]
                if cur > prev:
                    start = i * sps
                    end = min(start + int(0.06 * sps) + 1, N)
                    tx[start:end] *= (1.0 + asymmetry)
                elif cur < prev:
                    start = i * sps
                    end = min(start + int(0.06 * sps) + 1, N)
                    tx[start:end] *= (1.0 - asymmetry*0.7)
        return tx
    def through_link(x, frac_mean):
        alpha = 0.96
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * y[i-1] + (1-alpha) * x[i]
        if frac_timing_jitter > 1e-12:
            out = np.zeros_like(y)
            n_sym = len(y) // sps
            delays = frac_mean + rng.normal(0.0, frac_timing_jitter, size=n_sym+2)
            for si in range(n_sym):
                block = y[si*sps:(si+1)*sps]
                d = float(delays[si])
                block_shifted = frac_delay(np.pad(block, (sps, sps)), d, taps=41)
                out[si*sps:(si+1)*sps] = block_shifted[sps:sps+sps]
            y = out
        gain = 1.05
        y_nl = np.tanh(gain * y)
        y_mem = memory_poly(y_nl, b1=mem_params[0], b2=mem_params[1])
        sig_pow = np.mean(y_mem**2) + 1e-12
        snr = 10**(snr_db/10.0)
        noise_pow = sig_pow / snr
        y_mem += rng.normal(0.0, np.sqrt(noise_pow), size=y_mem.shape)
        y_mem = np.clip(y_mem, -1.0, 1.0)
        return y_mem
    tx1 = tx_waveform(dig1)
    tx2 = tx_waveform(dig2)
    ana1 = through_link(tx1, frac_timing_mean)
    ana2 = through_link(tx2, frac_timing_mean)
    peak = np.max(np.abs(ana1)) + 1e-12
    scale = target_peak / peak
    ana1 *= scale
    ana2 *= scale
    ana1 += rng.normal(0.0, 0.0008, size=ana1.shape)
    ana2 += rng.normal(0.0, 0.0008, size=ana2.shape)
    kernel = np.array([0.18, 0.64, 0.18])
    ana1 = np.convolve(ana1, kernel, mode='same')
    ana2 = np.convolve(ana2, kernel, mode='same')
    ana1 = np.clip(ana1, -1.0, 1.0)
    ana2 = np.clip(ana2, -1.0, 1.0)
    return ana1.astype(float), dig1.astype(int), dig2.astype(int), ana2.astype(float)

def format_float_list(xs):
    return " ".join("{:.8f}".format(float(x)) for x in xs)

def write_input_file(path, ana_waveform, dig_data_1, dig_data_2):
    N, M, K = len(ana_waveform), len(dig_data_1), len(dig_data_2)
    with open(path, "w") as f:
        f.write(str(N) + "\n")
        f.write(format_float_list(ana_waveform) + "\n")
        f.write(str(M) + "\n")
        f.write(" ".join(str(int(x)) for x in dig_data_1) + "\n")
        f.write(str(K) + "\n")
        f.write(" ".join(str(int(x)) for x in dig_data_2) + "\n")

def run_cpp_solution(exe_path, input_path, timeout=10):
    with open(input_path, "rb") as fin:
        try:
            proc = subprocess.run([exe_path], stdin=fin, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Execution timed out.") from e
    if proc.returncode != 0:
        raise RuntimeError(f"Solution exited with code {proc.returncode}. Stderr:\n{proc.stderr.decode('utf-8', errors='ignore')}")
    return proc.stdout.decode('utf-8', errors='ignore')

def parse_solution_output(stdout_text):
    tokens = re.findall(r"[+-]?(?:\d+\.\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?|[-+]?\d+", stdout_text)
    if not tokens:
        raise ValueError("No numeric tokens found in solution output.")
    L = int(float(tokens[0]))
    floats = [float(t) for t in tokens[1:1+L]]
    if len(floats) < L:
        raise ValueError(f"Solution output claimed L={L} but only {len(floats)} floats found.")
    return L, np.array(floats, dtype=np.float64)

def rmse(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} vs {b.shape}")
    return np.sqrt(np.mean((a - b) ** 2))

def main():
    parser = argparse.ArgumentParser(description="Generate contest-like tests, run a C++ solution, and visualize results.")
    parser.add_argument("--seed", type=int, help="Random seed (default: random)")
    parser.add_argument("--M", type=int, default=500, help="Length of dig_data_1")
    parser.add_argument("--K", type=int, default=800, help="Length of dig_data_2")
    parser.add_argument("--oversample", type=int, default=OVERSAMPLE, help="Oversample rate")
    parser.add_argument("--snr_db", type=float, default=26.0, help="SNR in dB for AWGN")
    parser.add_argument("--solution", type=str, help="Path to C++ solution file (e.g. solution.cpp)")
    parser.add_argument("--run-solution", type=int, choices=[0,1], default=1, help="Whether to run the solution")
    parser.add_argument("--timeout", type=int, default=20, help="Timeout in seconds for the solution")
    parser.add_argument("--stress-test", type=int, help="Number of stress test runs")
    parser.add_argument("--stress-folder", type=str, default="stress_results", help="Folder to save stress test results")
    parser.add_argument("--expected-output", type=str, help="Path to file containing expected reference waveform")
    parser.add_argument("--input-data", type=str, help="Path to file containing input waveform + digital symbols")
    args = parser.parse_args()

    if args.stress_test:
        os.makedirs(args.stress_folder, exist_ok=True)
        results = []

        exe_path = None
        if args.solution and args.run_solution:
            exe_path = "./solution_exec"
            try:
                compile_cmd = ["g++", "-std=c++17", "-O2", args.solution, "-o", exe_path]
                print("Compiling solution once:", " ".join(compile_cmd))
                subprocess.run(compile_cmd, check=True, capture_output=True)
            except Exception as e:
                print(f"Compilation failed: {e}", file=sys.stderr)
                exe_path = None

        for i in range(args.stress_test):
            seed = np.random.SeedSequence().entropy % (2**32)
            np.random.seed(seed)
            ana1, dig1, dig2, ana2 = make_case(args.M, args.K, sps=args.oversample, snr_db=args.snr_db, seed=seed)
            input_path = os.path.join(args.stress_folder, f"input_{i}.txt")
            write_input_file(input_path, ana1, dig1, dig2)

            sol_waveform = None
            if exe_path:
                try:
                    stdout_text = run_cpp_solution(exe_path, input_path, timeout=args.timeout)
                    _, sol_waveform = parse_solution_output(stdout_text)
                except Exception as e:
                    print(f"Run {i} error (Seed {seed}): {e}", file=sys.stderr)

            expected_L = args.K * args.oversample
            if sol_waveform is None:
                sol_waveform = np.zeros(expected_L)
            if sol_waveform.size != expected_L:
                padded_sol = np.zeros(expected_L)
                common_len = min(len(sol_waveform), expected_L)
                padded_sol[:common_len] = sol_waveform[:common_len]
                sol_waveform = padded_sol

            cur_rmse = rmse(sol_waveform, ana2)
            results.append((cur_rmse, seed))

        results.sort(reverse=True, key=lambda x: x[0])
        print("Stress Test Results (sorted by RMSE):")
        for r, s in results:
            print(f"Seed {s}: RMSE = {r:.6f}")
        return

    if args.input_data and args.expected_output:
        print(f"Using provided input '{args.input_data}' and expected output '{args.expected_output}'")
        with open(args.input_data, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        idx = 0
        N = int(lines[idx]); idx += 1
        ana_waveform_1 = np.array([float(x) for x in lines[idx].split()]); idx += 1
        M = int(lines[idx]); idx += 1
        dig_data_1 = [int(x) for x in lines[idx].split()]; idx += 1
        K = int(lines[idx]); idx += 1
        dig_data_2 = [int(x) for x in lines[idx].split()]; idx += 1
        with open(args.expected_output, "r") as f:
            expected_lines = [l.strip() for l in f if l.strip()]
        L = int(expected_lines[0])
        ana_waveform_2_ref = np.array([float(x) for x in expected_lines[1].split()])
        input_path = "provided_input.txt"
        write_input_file(input_path, ana_waveform_1, dig_data_1, dig_data_2)
        expected_L = len(ana_waveform_2_ref)
    else:
        seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy % (2**32)
        np.random.seed(seed)
        print(f"Using Seed: {seed}")
        ana_waveform_1, dig_data_1, dig_data_2, ana_waveform_2_ref = make_case(args.M, args.K, sps=args.oversample, snr_db=args.snr_db, seed=seed)
        input_path = "gen_input.txt"
        write_input_file(input_path, ana_waveform_1, dig_data_1, dig_data_2)
        print(f"Generated input written to '{input_path}'.")
        expected_L = args.K * args.oversample

    sol_waveform = None
    if args.solution and args.run_solution:
        try:
            exe_path = "./solution_exec"
            compile_cmd = ["g++", "-std=c++17", "-O2", args.solution, "-o", exe_path]
            print("Compiling solution:", " ".join(compile_cmd))
            subprocess.run(compile_cmd, check=True, capture_output=True)
            print("Running solution...")
            stdout_text = run_cpp_solution(exe_path, input_path, timeout=args.timeout)
            _, sol_waveform = parse_solution_output(stdout_text)
            print("Solution finished.")
        except Exception as e:
            print(f"Error during solution compile/run: {e}", file=sys.stderr)

    if sol_waveform is None:
        sol_waveform = np.zeros(expected_L)
    if sol_waveform.size != expected_L:
        padded_sol = np.zeros(expected_L)
        common_len = min(len(sol_waveform), expected_L)
        padded_sol[:common_len] = sol_waveform[:common_len]
        sol_waveform = padded_sol

    cur_rmse = rmse(sol_waveform, ana_waveform_2_ref)
    print(f"Final RMSE = {cur_rmse:.6f}")

    plt.figure(figsize=(15, 5))
    plt.title(f"Signal Comparison (RMSE: {cur_rmse:.6f})")
    x = np.arange(expected_L)
    plot_len = min(expected_L, 800)
    plt.plot(x[:plot_len], ana_waveform_2_ref[:plot_len], label="Reference Waveform", alpha=0.8)
    plt.plot(x[:plot_len], sol_waveform[:plot_len], label="Solution Output", alpha=0.8)
    dig_x = np.arange(len(dig_data_2)) * args.oversample + args.oversample // 2 if not (args.input_data and args.expected_output) else np.arange(len(dig_data_2))
    plt.stem(dig_x, np.array(dig_data_2) / 3.0, linefmt='r:', markerfmt='ro', basefmt=' ', label='Digital Symbols (scaled)')
    plt.xlim(0, plot_len)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    out_png = "comparison_plot.png"
    plt.savefig(out_png)
    print(f"Plot saved to {out_png}")


if __name__ == "__main__":
    main()
