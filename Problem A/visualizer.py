import argparse
import os
import subprocess
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

OVERSAMPLE = 16

def rrc_filter(beta: float, span: int, sps: int) -> np.ndarray:
    N = span * sps
    t = np.arange(-N, N + 1) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(1 - (4 * beta * ti) ** 2) < 1e-8:
            h[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))
        elif abs(ti) < 1e-8:
            h[i] = (1 + beta * (4 / np.pi - 1))
        else:
            num = np.sin(np.pi * (1 - beta) * ti) + 4 * beta * ti * np.cos(np.pi * (1 + beta) * ti)
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h = h / np.sqrt(np.sum(h**2))
    return h

def frac_delay(signal: np.ndarray, delay: float, taps: int = 81) -> np.ndarray:
    n = np.arange(-(taps//2), taps//2 + 1)
    h = np.sinc(n - delay)
    h *= np.hamming(len(h))
    h /= np.sum(h)
    return np.convolve(signal, h, mode='same')

def channel_impulse_response(sps: int) -> np.ndarray:
    sym_fir = np.array([0.95, 0.15, -0.08, 0.04, -0.02], dtype=float)
    up = np.zeros(len(sym_fir) * sps)
    up[::sps] = sym_fir
    rc = rrc_filter(beta=0.25, span=4, sps=sps)
    h = np.convolve(up, rc)
    h /= np.sum(abs(h))
    return h

def memoryless_nl(x: np.ndarray, a3: float, a5: float) -> np.ndarray:
    return x + a3 * x**3 + a5 * x**5

def memory_poly(x: np.ndarray, b1: float, b2: float) -> np.ndarray:
    x_del = np.concatenate([[0.0], x[:-1]])
    return x + b1 * x * x_del + b2 * x_del**2

def make_case(M:int, K:int, sps:int=OVERSAMPLE, snr_db:float=26.0,
              nl_params=(0.06,0.008), mem_params=(0.05,-0.02), frac_timing:float=0.35, seed=None):
    levels = np.array([-3,-1,1,3], dtype=int)
    rng = np.random.default_rng(seed)
    dig1 = rng.choice(levels, size=M, replace=True)
    dig2 = rng.choice(levels, size=K, replace=True)

    tx_rrc = rrc_filter(beta=0.3, span=6, sps=sps)
    def tx_waveform(dig):
        up = np.zeros(len(dig)*sps, dtype=float)
        up[::sps] = dig.astype(float) / 3.0
        x = np.convolve(up, tx_rrc, mode='same')
        return x

    hch = channel_impulse_response(sps)
    a3, a5 = nl_params
    b1, b2 = mem_params

    def through_link(x):
        y = np.convolve(x, hch, mode='same')
        y = frac_delay(y, frac_timing, taps=81)
        y = memoryless_nl(y, a3=a3, a5=a5)
        y = memory_poly(y, b1=b1, b2=b2)
        sig_pow = np.mean(y**2)
        snr = 10**(snr_db/10)
        noise_pow = sig_pow / snr
        y += rng.normal(0.0, np.sqrt(noise_pow), size=y.shape)
        y = np.clip(y, -1.0, 1.0)
        return y

    ana1 = through_link(tx_waveform(dig1))
    ana2 = through_link(tx_waveform(dig2))
    return ana1, dig1, dig2, ana2

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
    plt.plot(x[:plot_len], sol_waveform[:plot_len], label="Solution Output", linestyle='--', alpha=0.8)
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
