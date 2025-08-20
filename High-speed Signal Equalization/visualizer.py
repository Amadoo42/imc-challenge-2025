#!/usr/bin/env python3
"""
PAM4 Signal Recovery Problem Visualizer
Generates test cases, runs C++ solutions, and visualizes results
"""

import argparse
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tempfile
import shutil
from pathlib import Path
import json
import sys

class PAM4Generator:
    def __init__(self, seed=42, noise_level=0.1):
        np.random.seed(seed)
        random.seed(seed)
        self.noise_level = noise_level
        self.pam4_levels = [3, 1, -1, -3]
        self.osr = 4  # Over sampling rate
        self.beta = 0.5  # Roll-off factor for RRC
        
    def generate_digital_sequence(self, length):
        """Generate random PAM4 digital sequence"""
        return np.random.choice(self.pam4_levels, size=length)
    
    def generate_rrc_pulse(self, span=6):
        """Generate Root Raised Cosine pulse"""
        # Number of samples for the pulse
        N = span * self.osr + 1
        t = np.arange(-span/2, span/2 + 1/self.osr, 1/self.osr)
        
        # Avoid division by zero
        pulse = np.zeros(len(t))
        
        for i, time in enumerate(t):
            if time == 0:
                pulse[i] = (1 + self.beta * (4/np.pi - 1))
            elif abs(time) == 1/(4*self.beta):
                pulse[i] = (self.beta/np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*self.beta)) + 
                           (1 - 2/np.pi) * np.cos(np.pi/(4*self.beta)))
            else:
                numerator = np.sin(np.pi * time * (1 - self.beta)) + 4 * self.beta * time * np.cos(np.pi * time * (1 + self.beta))
                denominator = np.pi * time * (1 - (4 * self.beta * time)**2)
                pulse[i] = numerator / denominator
        
        # Normalize pulse energy
        pulse = pulse / np.sqrt(np.sum(pulse**2))
        return pulse
    
    def apply_pulse_shaping(self, digital_data):
        """Apply RRC pulse shaping to digital data"""
        # Generate RRC pulse
        rrc_pulse = self.generate_rrc_pulse()
        
        # Upsample digital data (insert zeros between symbols)
        upsampled = np.zeros(len(digital_data) * self.osr)
        upsampled[::self.osr] = digital_data
        
        # Convolve with RRC pulse
        shaped_signal = np.convolve(upsampled, rrc_pulse, mode='same')
        
        return shaped_signal
    
    def add_channel_effects(self, signal):
        """Add realistic channel impairments"""
        # Add intersymbol interference (ISI) - multipath channel
        isi_filter = np.array([0.05, 0.9, 0.05])  # Simple ISI model
        signal_isi = np.convolve(signal, isi_filter, mode='same')
        
        # Add nonlinearity (AM-AM and AM-PM distortion)
        signal_nonlinear = signal_isi + 0.03 * signal_isi**3 - 0.01 * signal_isi**5
        
        # Add frequency-dependent attenuation
        # Simple first-order low-pass filter (bandwidth limitation)
        alpha = 0.2
        filtered_signal = np.zeros_like(signal_nonlinear)
        filtered_signal[0] = signal_nonlinear[0]
        for i in range(1, len(signal_nonlinear)):
            filtered_signal[i] = alpha * signal_nonlinear[i] + (1 - alpha) * filtered_signal[i-1]
        
        # Add AWGN noise
        noise = np.random.normal(0, self.noise_level, len(filtered_signal))
        signal_noisy = filtered_signal + noise
        
        # Add timing jitter (small random delays)
        jitter_std = 0.1
        jitter = np.random.normal(0, jitter_std, len(signal_noisy))
        jittered_indices = np.arange(len(signal_noisy)) + jitter
        jittered_indices = np.clip(jittered_indices, 0, len(signal_noisy) - 1)
        
        # Interpolate to apply jitter
        signal_jittered = np.interp(np.arange(len(signal_noisy)), jittered_indices, signal_noisy)
        
        # Normalize to [-1, 1] range while preserving relative amplitudes
        max_val = np.max(np.abs(signal_jittered))
        if max_val > 0:
            signal_jittered = 0.8 * signal_jittered / max_val  # Leave some headroom
        
        return signal_jittered
    
    def generate_analog_waveform(self, digital_data):
        """Convert digital PAM4 sequence to analog waveform using RRC pulse shaping"""
        # Apply RRC pulse shaping
        shaped_signal = self.apply_pulse_shaping(digital_data)
        
        # Add channel effects
        analog_waveform = self.add_channel_effects(shaped_signal)
        
        return analog_waveform

class PAM4Visualizer:
    def __init__(self):
        pass
        
    def compile_cpp_solution(self, cpp_file):
        """Compile C++ solution"""
        if not os.path.exists(cpp_file):
            print(f"Error: C++ file {cpp_file} not found")
            return None
            
        executable = cpp_file.replace('.cpp', '')
        try:
            result = subprocess.run(['g++', '-O2', '-o', executable, cpp_file], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Compilation error: {result.stderr}")
                return None
            return executable
        except Exception as e:
            print(f"Compilation failed: {e}")
            return None
    
    def run_solution(self, executable, analog_data):
        """Run compiled solution with input data"""
        try:
            input_str = f"{len(analog_data)}\n"
            input_str += " ".join(map(str, analog_data)) + "\n"
            
            result = subprocess.run([f'./{executable}'], input=input_str, 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Runtime error: {result.stderr}")
                return None
                
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                print("Invalid output format")
                return None
                
            L = int(lines[0])
            reconstructed = list(map(int, lines[1].split()))
            
            if L != len(reconstructed):
                print(f"Output length mismatch: expected {L}, got {len(reconstructed)}")
                return None
                
            return reconstructed
            
        except subprocess.TimeoutExpired:
            print("Solution timed out")
            return None
        except Exception as e:
            print(f"Execution failed: {e}")
            return None
    
    def calculate_ber(self, original, reconstructed):
        """Calculate Bit Error Rate using PAM4 to bit conversion"""
        if len(original) != len(reconstructed):
            return 1.0
        
        # PAM4 to 2-bit mapping (Gray coding typically used)
        pam4_to_bits = {
            3: [0, 0],   # 00
            1: [0, 1],   # 01  
            -1: [1, 1],  # 11
            -3: [1, 0]   # 10
        }
        
        total_bit_errors = 0
        total_bits = len(original) * 2  # Each PAM4 symbol = 2 bits
        
        for orig_symbol, recon_symbol in zip(original, reconstructed):
            if orig_symbol in pam4_to_bits and recon_symbol in pam4_to_bits:
                orig_bits = pam4_to_bits[orig_symbol]
                recon_bits = pam4_to_bits[recon_symbol]
                
                # Count bit errors
                for orig_bit, recon_bit in zip(orig_bits, recon_bits):
                    if orig_bit != recon_bit:
                        total_bit_errors += 1
            else:
                # If invalid symbol, count as 2 bit errors
                total_bit_errors += 2
        
        return total_bit_errors / total_bits
    
    def save_generated_input(self, analog_data, seed, filename="gen_input.txt"):
        """Save the generated input to a file with seed information"""
        with open(filename, 'w') as f:
            f.write(f"# Generated with seed: {seed}\n")
            f.write(f"# To reproduce: --seed {seed}\n")
            f.write(f"{len(analog_data)}\n")
            f.write(" ".join(map(str, analog_data)) + "\n")
        print(f"Generated input saved as {filename}")
    
    def visualize_results(self, analog_data, original_digital, reconstructed_digital, 
                         filename="comparison_plot.png", show_stats=True):
        """Create visualization plot"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Analog waveform
        time_analog = np.arange(len(analog_data))
        ax1.plot(time_analog, analog_data, 'b-', linewidth=0.5, label='Received Analog Signal')
        ax1.set_title('Received Analog Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Original vs Reconstructed Digital Signals (rectangular/step plot)
        time_digital = np.arange(len(original_digital))
        
        # Create step plots for rectangular appearance
        ax2.step(time_digital, original_digital, 'g-', linewidth=2, 
                label='Original Digital Signal', alpha=0.8, where='post')
        ax2.step(time_digital, reconstructed_digital, 'r--', linewidth=2, 
                label='Reconstructed Digital Signal', alpha=0.8, where='post')
        
        # Highlight errors with vertical lines
        errors = [i for i, (a, b) in enumerate(zip(original_digital, reconstructed_digital)) if a != b]
        if errors:
            for error_idx in errors:
                ax2.axvline(x=error_idx, color='red', alpha=0.3, linewidth=3, label='Error' if error_idx == errors[0] else "")
            
            # Also add scatter points for visibility
            ax2.scatter([time_digital[i] for i in errors], 
                       [reconstructed_digital[i] for i in errors], 
                       color='red', s=50, marker='x', zorder=5)
        
        ax2.set_title('Digital Signal Comparison (Rectangular)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Symbol Index')
        ax2.set_ylabel('PAM4 Level')
        ax2.set_yticks([-3, -1, 1, 3])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if show_stats:
            ber = self.calculate_ber(original_digital, reconstructed_digital)
            total_symbols = len(original_digital)
            total_bits = total_symbols * 2  # Each PAM4 symbol = 2 bits
            symbol_errors = sum(1 for a, b in zip(original_digital, reconstructed_digital) if a != b)
            bit_errors = int(ber * total_bits)
            
            stats_text = f'BER: {ber:.6f}\nSymbol Errors: {symbol_errors}/{total_symbols}\nBit Errors: {bit_errors}/{total_bits}'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {filename}")
        
        return fig
    
    def stress_test(self, cpp_file, num_tests, L, noise_level, seed_start=0):
        """Run stress test with multiple random test cases"""
        executable = self.compile_cpp_solution(cpp_file)
        if not executable:
            return
        
        test_results = []
        
        print(f"Running stress test with {num_tests} test cases...")
        
        for i in range(num_tests):
            current_seed = seed_start + i
            print(f"Running test {i+1}/{num_tests} (seed: {current_seed})...", end=' ')
            
            generator = PAM4Generator(seed=current_seed, noise_level=noise_level)
            original_digital = generator.generate_digital_sequence(L)
            analog_data = generator.generate_analog_waveform(original_digital)
            
            reconstructed_digital = self.run_solution(executable, analog_data)
            
            if reconstructed_digital is None:
                print("FAILED")
                continue
            
            ber = self.calculate_ber(original_digital, reconstructed_digital)
            
            test_result = {
                'test_id': i,
                'seed': current_seed,
                'ber': ber,
                'L': L,
                'noise_level': noise_level
            }
            
            test_results.append(test_result)
            print(f"BER: {ber:.6f}")
        
        # Sort by BER (highest first)
        test_results.sort(key=lambda x: x['ber'], reverse=True)
        
        # Display results summary
        print(f"\nTop 3 worst cases (highest BER):")
        for i, result in enumerate(test_results[:3]):
            print(f"{i+1}. Test {result['test_id']} (seed: {result['seed']}): BER = {result['ber']:.6f}")
        
        # Summary statistics
        bers = [r['ber'] for r in test_results]
        print(f"\nStress Test Summary:")
        print(f"Total tests: {len(test_results)}")
        print(f"Average BER: {np.mean(bers):.6f}")
        print(f"Median BER: {np.median(bers):.6f}")
        print(f"Worst BER: {np.max(bers):.6f}")
        print(f"Best BER: {np.min(bers):.6f}")
        
        # Clean up
        if os.path.exists(executable):
            os.remove(executable)

def main():
    parser = argparse.ArgumentParser(description='PAM4 Signal Recovery Visualizer')
    parser.add_argument('cpp_file', help='Path to C++ solution file')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (if not provided, a random seed will be generated)')
    parser.add_argument('--L', type=int, default=1000, help='Length of digital sequence (number of symbols)')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level (0.0 to 1.0)')
    parser.add_argument('--stress-test', type=int, help='Run stress test with N test cases')
    parser.add_argument('--custom-input', help='Path to custom input file (format: first line N, second line analog data)')
    parser.add_argument('--custom-output', help='Path to expected output file (format: first line L, second line digital data)')
    
    args = parser.parse_args()
    
    # Generate random seed if none provided
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"No seed provided, using random seed: {args.seed}")
    else:
        print(f"Using provided seed: {args.seed}")
    
    visualizer = PAM4Visualizer()
    
    if args.stress_test:
        # For stress tests, use provided seed or generate one
        if args.seed is None:
            base_seed = random.randint(0, 2**31 - 1)
            print(f"Stress test using random base seed: {base_seed}")
        else:
            base_seed = args.seed
            print(f"Stress test using provided base seed: {base_seed}")
        
        visualizer.stress_test(args.cpp_file, args.stress_test, args.L, args.noise, 
                              base_seed)
        return
    
    if args.custom_input and args.custom_output:
        # Use custom input/output
        try:
            with open(args.custom_input, 'r') as f:
                lines = f.readlines()
                N = int(lines[0].strip())
                analog_data = list(map(float, lines[1].strip().split()))
            
            with open(args.custom_output, 'r') as f:
                lines = f.readlines()
                L = int(lines[0].strip())
                original_digital = list(map(int, lines[1].strip().split()))
                
            print(f"Using custom input: N={N}, L={L}")
            
        except Exception as e:
            print(f"Error reading custom files: {e}")
            return
    else:
        # Generate test case
        generator = PAM4Generator(seed=args.seed, noise_level=args.noise)
        original_digital = generator.generate_digital_sequence(args.L)
        analog_data = generator.generate_analog_waveform(original_digital)
        
        # Save generated input
        visualizer.save_generated_input(analog_data, args.seed)
        
        print(f"Generated test case: L={args.L}, Noise={args.noise}, Seed={args.seed}")
        print(f"To reproduce this test case, use: --seed {args.seed}")
    
    # Compile and run solution
    executable = visualizer.compile_cpp_solution(args.cpp_file)
    if not executable:
        return
    
    print("Running solution...")
    reconstructed_digital = visualizer.run_solution(executable, analog_data)
    
    if reconstructed_digital is None:
        print("Solution failed to run")
        return
    
    # Calculate and display results
    ber = visualizer.calculate_ber(original_digital, reconstructed_digital)
    symbol_errors = sum(1 for a, b in zip(original_digital, reconstructed_digital) if a != b)
    total_bits = len(original_digital) * 2
    bit_errors = int(ber * total_bits)
    
    print(f"Bit Error Rate (BER): {ber:.6f}")
    print(f"Symbol Errors: {symbol_errors}/{len(original_digital)}")
    print(f"Bit Errors: {bit_errors}/{total_bits}")
    
    # Create visualization
    visualizer.visualize_results(analog_data, original_digital, reconstructed_digital)
    
    # Clean up
    if os.path.exists(executable):
        os.remove(executable)

if __name__ == "__main__":
    main()