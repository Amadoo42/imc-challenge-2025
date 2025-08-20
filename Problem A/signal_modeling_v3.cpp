#include <bits/stdc++.h>
using namespace std;

/* ------------------------- */
/* Hyperparameters */
const int NEIGH = 200;
const int POLY_DEG = 6;
const int SMOOTHING_RADIUS = 3;
const double RIDGE_FACTOR = 1e-2;
/* ------------------------- */

const int S = 16, KERNEL_LEN = 2 * NEIGH + 1, INFO = KERNEL_LEN + 1;

inline int idx_of(const int &v) {
    if(v == -3) return 0;
    if(v == -1) return 1;
    if(v == 1) return 2;
    return 3;
}

inline int zero_padding(int i, const vector <int> &data) {
    if(i < 0 || i >= (int)data.size()) return 0;
    return data[i];
}

inline void cholesky_solve(vector <vector <double>> &A, vector <vector <double>> &B) {
    int n = A.size(), m = B[0].size();
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for(int k = 0; k < j; ++k) sum -= A[i][k] * A[j][k];
            if(i == j) {
                assert(sum > 0.0);
                A[i][i] = sqrt(sum);
            }
            else A[i][j] = sum / A[j][j];
        }
    }
    for(int k = 0; k < m; ++k) {
        for(int i = 0; i < n; ++i) {
            double sum = B[i][k];
            for(int j = 0; j < i; ++j) sum -= A[i][j] * B[j][k];
            B[i][k] = sum / A[i][i];
        }
    }
    for(int k = 0; k < m; ++k) {
        for(int i = n - 1; i >= 0; --i) {
            double sum = B[i][k];
            for(int j = i + 1; j < n; ++j) sum -= A[j][i] * B[j][k];
            B[i][k] = sum / A[i][i];
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n; cin >> n;
    vector <double> ana_waveform_1(n);
    for(int i = 0; i < n; ++i) cin >> ana_waveform_1[i];
    int m; cin >> m;
    vector <int> dig_data_1(m);
    for(int i = 0; i < m; ++i) cin >> dig_data_1[i];
    int k; cin >> k;
    vector <int> dig_data_2(k);
    for(int i = 0; i < k; ++i) cin >> dig_data_2[i];
    vector <vector <double>> means(4, vector <double>(S, 0.0));
    vector <int> cnts(4, 0);
    for(int i = 0; i < m; ++i) {
        int id = idx_of(dig_data_1[i]);
        for(int p = 0; p < S; ++p) means[id][p] += ana_waveform_1[i * S + p];
        cnts[id]++;
    }
    for(int id = 0; id < 4; ++id) {
        if(!cnts[id]) continue;
        for(int p = 0; p < S; ++p) means[id][p] /= cnts[id];
    }
    vector <vector <double>> A(INFO, vector <double>(INFO, 0.0));
    vector <vector <double>> B(INFO, vector <double>(S, 0.0));
    vector <double> info(INFO, 0.0);
    for(int i = 0; i < m; ++i) {
        for(int u = 0; u < KERNEL_LEN; ++u) info[u] = zero_padding(i + u - NEIGH, dig_data_1);
        info[KERNEL_LEN] = 1.0;
        for(int u = 0; u < INFO; ++u) for(int v = 0; v <= u; ++v) A[u][v] += info[u] * info[v];
        for(int p = 0; p < S; ++p) for(int u = 0; u < INFO; ++u) B[u][p] += info[u] * (ana_waveform_1[i * S + p] - means[idx_of(dig_data_1[i])][p]);
    }
    for(int i = 0; i < INFO; ++i) for(int j = i + 1; j < INFO; ++j) A[i][j] = A[j][i];
    double diag_sum = 0.0;
    for(int i = 0; i < INFO; ++i) diag_sum += A[i][i];
    double ridge = RIDGE_FACTOR;
    if(diag_sum > 0.0) ridge = RIDGE_FACTOR * (diag_sum / INFO);
    for(int i = 0; i < INFO; ++i) A[i][i] += ridge;
    cholesky_solve(A, B);
    vector <double> lin_learned(n, 0.0);
    for(int i = 0; i < m; ++i) {
        for(int u = 0; u < KERNEL_LEN; ++u) info[u] = zero_padding(i + u - NEIGH, dig_data_1);
        info[KERNEL_LEN] = 1.0;
        for(int p = 0; p < S; ++p) {
            lin_learned[i * S + p] = means[idx_of(dig_data_1[i])][p];
            for(int u = 0; u < INFO; ++u) lin_learned[i * S + p] += info[u] * B[u][p];
        }
    }
    double lin_mean = 0.0, lin_std = 0.0;
    for(int i = 0; i < n; ++i) lin_mean += lin_learned[i];
    lin_mean /= n;
    for(int i = 0; i < n; ++i) lin_std += (lin_learned[i] - lin_mean) * (lin_learned[i] - lin_mean);
    lin_std = sqrt(max(1e-18, lin_std / n));
    if(lin_std < 1e-9) lin_std = 1.0;
    vector <vector <double>> P(POLY_DEG + 1, vector <double>(POLY_DEG + 1, 0.0));
    vector <double> q(POLY_DEG + 1, 0.0);
    for(int i = 0; i <n; ++i) {
        double t = (lin_learned[i] - lin_mean) / lin_std;
        vector <double> poly_pow(POLY_DEG + 1);
        poly_pow[0] = 1.0;
        for(int d = 1; d <= POLY_DEG; ++d) poly_pow[d] = poly_pow[d - 1] * t;
        for(int u = 0; u <= POLY_DEG; ++u) {
            for(int v = 0; v <= POLY_DEG; ++v) P[u][v] += poly_pow[u] * poly_pow[v];
            q[u] += poly_pow[u] * ana_waveform_1[i];
        }
    }
    diag_sum = 0.0;
    for(int i = 0; i <= POLY_DEG; ++i) diag_sum += P[i][i];
    ridge = 1e-8 * max(1.0, diag_sum / (POLY_DEG + 1));
    for(int i = 0; i <= POLY_DEG; ++i) P[i][i] += ridge;
    vector <vector <double>> Q(POLY_DEG + 1, vector <double>(1));
    for(int i = 0; i <= POLY_DEG; ++i) Q[i][0] = q[i];
    cholesky_solve(P, Q);
    vector <double> poly_learned(n, 0.0);
    for(int i = 0; i < n; ++i) {
        double t = (lin_learned[i] - lin_mean) / lin_std;
        double powt = 1.0;
        for(int d = 0; d <= POLY_DEG; ++d) poly_learned[i] += Q[d][0] * powt, powt *= t;
    }
    double num = 0.0, denom = 0.0, pred_mean = 0.0, true_mean = 0.0;
    for(int i = 0; i < n; ++i) {
        num += poly_learned[i] * ana_waveform_1[i];
        denom += poly_learned[i] * poly_learned[i];
        pred_mean += poly_learned[i];
        true_mean += ana_waveform_1[i];
    }
    pred_mean /= n, true_mean /= n;
    double s = 1.0, b = 0.0;
    if(denom > 1e-15) s = num / denom;
    b = true_mean - s * pred_mean;
    for(int i = 0; i < n; ++i) poly_learned[i] = s * poly_learned[i] + b;
    int L = k * S;
    cout << L << '\n';
    vector <double> res;
    for(int i = 0; i < k; ++i) {
        for(int u = 0; u < KERNEL_LEN; ++u) info[u] = zero_padding(i + u - NEIGH, dig_data_2);
        info[KERNEL_LEN] = 1.0;
        for(int p = 0; p < S; ++p) {
            double sum = means[idx_of(dig_data_2[i])][p];
            for(int u = 0; u < INFO; ++u) sum += info[u] * B[u][p];
            res.push_back(sum);
        }
    }
    for(int i = 0; i < L; ++i) {
        double t = (res[i] - lin_mean) / lin_std;
        double sum = 0.0, powt = 1.0;
        for(int d = 0; d <= POLY_DEG; ++d) sum += Q[d][0] * powt, powt *= t;
        res[i] = sum;
    }
    for(int i = 0; i < L; ++i) res[i] = s * res[i] + b;
    vector <vector <double>> residual_mean(4, vector <double>(S, 0));
    vector <vector <int>> residual_cnt(4, vector <int>(S, 0));
    for(int i = 0; i < m; ++i) for(int p = 0; p < S; ++p) residual_mean[idx_of(dig_data_1[i])][p] += ana_waveform_1[i * S + p] - poly_learned[i * S + p], residual_cnt[idx_of(dig_data_1[i])][p]++;
    for(int id = 0; id < 4; ++id) for(int p = 0; p < S; ++p) {
        if(residual_cnt[id][p]) residual_mean[id][p] /= residual_cnt[id][p];
    }
    for(int id = 0; id < 4; ++id) for(int p = 0; p < S; ++p) {
        if(residual_mean[id][p] > 0.2) residual_mean[id][p] = 0.2;
        else if(residual_mean[id][p] < -0.2) residual_mean[id][p] = -0.2;
    }
    for(int i = 0; i < k; ++i) for(int p = 0; p < S; ++p) res[i * S + p] += residual_mean[idx_of(dig_data_2[i])][p];
    vector <double> tmp = res;
    for(int i = 0; i < L; ++i) {
        double a0 = tmp[max(0, i - 1)], b0 = tmp[i], c0 = tmp[min(i + 1, L - 1)];
        res[i] = 0.25 * a0 + 0.5 * b0 + 0.25 * c0;
    }
    double mn = 1e9, mx = -1e9;
    for(int i = 0; i < n; ++i) mn = min(mn, ana_waveform_1[i]), mx = max(mx, ana_waveform_1[i]);
    double margin = 1e-6 + 0.05 * max(1.0, fabs(mx - mn));
    for(int i = 0; i < L; ++i) {
        if(res[i] < mn - margin) res[i] = mn - margin;
        if(res[i] > mx + margin) res[i] = mx + margin;
    }
    for(int i = 0; i < L; ++i) cout << fixed << setprecision(6) << res[i] << " \n"[i == L - 1];
    return 0;
}