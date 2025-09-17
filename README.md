# Huawei IMC Challenge 2025 - Team "Sunflowers shine"

🏅 **38th Place Globally** - This repository contains our team's solutions for the [Huawei IMC (Information, Mathematics & Computing) Challenge 2025](https://www.huawei.com/minisite/imc-challenge/en/). Our team "Sunflowers shine" consists of 4 members who tackled challenging problems in high-speed signal processing and digital communication systems.

## 🏆 Competition Overview

The Huawei IMC Challenge is a global competition focusing on cutting-edge problems in information technology, mathematics, and computing. Our solutions address two major problem areas in modern communication systems:

1. **High-speed Signal Equalization** - PAM4 signal recovery from noisy analog waveforms
2. **High-speed Signal Modeling** - Analog waveform synthesis from digital symbol sequences

## 📁 Repository Structure

```
├── High-speed Signal Equalization/
│   ├── signal_equalization_v1.cpp     # Main equalization solution
│   └── visualizer.py                  # Test case generation and visualization
├── High-speed Signal Modeling/
│   ├── signal_modeling_v1.cpp         # Linear regression approach
│   ├── signal_modeling_v2.cpp         # Robust regression with IRLS
│   ├── signal_modeling_v3.cpp         # Polynomial regression approach
│   ├── visualizer.py                  # Legacy visualizer
│   └── new_visualizer.py              # Enhanced visualizer with stress testing
├── .gitignore
├── .gitattributes
└── LICENSE
```

## 🔧 Problem Solutions

### Problem 1: High-speed Signal Equalization

**Challenge**: Recover original PAM4 digital symbols from received analog waveforms corrupted by channel impairments and noise.

**Our Approach** (`signal_equalization_v1.cpp`):
- **Adaptive Equalization**: Custom adaptive filter implementation with step-size control
- **K-means Clustering**: 1D K-means for PAM4 level detection and decision boundaries
- **Multi-phase Processing**: Process signal in 4 phases to handle timing offsets
- **Normalization**: Dynamic amplitude scaling for robust performance

**Key Features**:
- Handles intersymbol interference (ISI)
- Robust to timing jitter and noise
- Automatic gain control
- Phase-aware symbol recovery

### Problem 2: High-speed Signal Modeling

**Challenge**: Generate realistic analog waveforms from digital symbol sequences, modeling complex channel characteristics.

**Our Solutions**:

#### Version 1: Linear Regression (`signal_modeling_v1.cpp`)
- Ridge regression with extensive feature engineering
- Neighbor-based features with polynomial interactions
- Binomial smoothing for coefficient regularization
- Symbol-specific mean modeling

#### Version 2: Robust Regression (`signal_modeling_v2.cpp`)
- **Iteratively Reweighted Least Squares (IRLS)**
- Huber loss function for outlier robustness
- Adaptive neighbor selection strategy
- Feature normalization and standardization

#### Version 3: Polynomial Modeling (`signal_modeling_v3.cpp`)
- Two-stage approach: Linear + Polynomial refinement
- Cholesky decomposition for efficient matrix operations
- Residual-based correction
- Output clamping for realistic signal bounds

## 🛠️ Technical Implementation

### Key Algorithms Used

1. **Adaptive Filtering**
   - LMS (Least Mean Squares) adaptation
   - Variable step-size control
   - Convergence monitoring

2. **Machine Learning**
   - K-means clustering for symbol detection
   - Ridge regression with L2 regularization
   - Robust regression with iterative reweighting

3. **Signal Processing**
   - Root Raised Cosine (RRC) filtering
   - Fractional delay implementation
   - Memory polynomial modeling
   - Channel impulse response modeling

4. **Numerical Methods**
   - Cholesky decomposition for linear systems
   - Matrix inversion with pivoting
   - Iterative optimization algorithms

### Performance Features

- **Real-time Processing**: Optimized C++ implementation
- **Memory Efficiency**: In-place operations where possible
- **Numerical Stability**: Ridge regularization and careful scaling
- **Robustness**: Handles edge cases and outliers gracefully

## 🎯 Skills Learned

Through this competition, our team developed expertise in:

### Technical Skills
- **Digital Signal Processing**: Filter design, equalization, channel modeling
- **Machine Learning**: Clustering, regression, robust statistics
- **Numerical Computing**: Linear algebra, optimization, matrix decomposition
- **C++ Programming**: High-performance computing, memory optimization
- **Algorithm Design**: Adaptive algorithms, iterative methods

### Mathematical Concepts
- **Communication Theory**: PAM4 modulation, channel impairments, ISI
- **Statistical Methods**: Robust regression, outlier detection, noise modeling
- **Linear Algebra**: Matrix operations, eigenvalue problems, least squares
- **Optimization**: Regularization techniques, convergence analysis

### Engineering Practices
- **Problem Decomposition**: Breaking complex problems into manageable components
- **Performance Tuning**: Algorithmic and implementation optimizations
- **Testing & Validation**: Comprehensive test case generation and validation
- **Code Organization**: Modular design, version control, documentation

### Soft Skills
- **Team Collaboration**: Coordinated development across 4 team members
- **Research Skills**: Literature review, algorithm adaptation
- **Problem Solving**: Creative approaches to challenging technical problems
- **Presentation**: Clear documentation and code organization

## 🚀 Usage

### Compilation
```bash
g++ -std=c++17 -O2 signal_equalization_v1.cpp -o equalization
g++ -std=c++17 -O2 signal_modeling_v3.cpp -o modeling
```

### Testing with Visualizers
```bash
# Signal Equalization
python3 visualizer.py signal_equalization_v1.cpp --L 1000 --noise 0.1

# Signal Modeling
python3 new_visualizer.py --solution signal_modeling_v3.cpp --M 500 --K 800

# Stress Testing
python3 new_visualizer.py --solution signal_modeling_v3.cpp --stress-test 10
```

## 📊 Results

Our solutions achieved **38th place globally** in the Huawei IMC Challenge 2025, demonstrating:
- **High Accuracy**: Low bit error rates for signal equalization
- **Realistic Modeling**: RMSE values competitive with reference implementations
- **Robustness**: Consistent performance across diverse test conditions
- **Efficiency**: Fast execution suitable for real-time applications
- **Global Competitiveness**: Top-tier performance among international participants

## 🤝 Team Members

**Team "Sunflowers shine"** - 4 dedicated members who contributed to algorithm development, implementation, testing, and optimization.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Official Huawei IMC Challenge Page](https://www.huawei.com/minisite/imc-challenge/en/)
- Competition focuses on advanced problems in information technology and mathematical computing

---

*This repository represents our team's journey through complex signal processing challenges, demonstrating the intersection of mathematics, computer science, and engineering in solving real-world communication problems.*
