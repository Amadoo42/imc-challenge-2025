#include <bits/stdc++.h>
using namespace std;

const int S = 16, NEIGH = 84, SMOOTHING_REPETITIONS = 7, IRLS_REPETITIONS = 30;
const double EPS = 1e-12, RIDGE_FACTOR = 1e-3;

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
            if(i == j) A[i][i] = sqrt(sum);
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
    vector <vector <double>> sums(4, vector <double>(S, 0.0));
    vector <int> cnts(4, 0);
    for(int i = 0; i < m; ++i) {
        int id = idx_of(dig_data_1[i]);
        for(int p = 0; p < S; ++p) sums[id][p] += ana_waveform_1[i * S + p];
        cnts[id]++;
    }
    vector <vector <double>> means(4, vector <double>(S, 0.0));
    for(int i = 0; i < 4; ++i) {
        if(!cnts[i]) continue;
        for(int p = 0; p < S; ++p) means[i][p] = sums[i][p] / cnts[i];
    }
    vector <int> neighbours_taken;
    for(int p = -NEIGH; p <= NEIGH; ++p) {
        int ap = abs(p);
        bool take = 0;
        if(ap <= 10) take = 1;
        else if(ap <= 30 && (ap % 2 == 0)) take = 1;
        else if(ap <= 60 && (ap % 4 == 0)) take = 1;
        else if(ap <= NEIGH && (ap % 8 == 0)) take = 1;
        if(take) neighbours_taken.push_back(p);
    }
    bool took_0 = 0;
    for(auto i : neighbours_taken) took_0 |= (!i);
    if(!took_0) neighbours_taken.push_back(0);
    sort(begin(neighbours_taken), end(neighbours_taken));
    neighbours_taken.erase(unique(begin(neighbours_taken), end(neighbours_taken)), end(neighbours_taken));
    int INFO = 3 + (int)neighbours_taken.size() * 2;
    vector <vector <double>> info(m, vector <double>(INFO, 0.0));
    for(int i = 0; i < m; ++i) {
        int idx = 0;
        info[i][idx++] = 1.0;
        for(auto neigh : neighbours_taken) {
            double val = zero_padding(i + neigh, dig_data_1);
            info[i][idx++] = val;
            info[i][idx++] = val * val * val;
        }
        double prev = zero_padding(i - 1, dig_data_1);
        double cur = dig_data_1[i];
        double nxt = zero_padding(i + 1, dig_data_1);
        info[i][idx++] = prev * cur;
        info[i][idx++] = cur * nxt;
    }
    vector <double> info_mean(INFO, 0.0), info_var(INFO, 0.0), info_std(INFO, 0.0);
    for(int i = 0; i < INFO; ++i) {
        for(int j = 0; j < m; ++j) info_mean[i] += info[j][i];
        info_mean[i] /= m;
    }
    for(int i = 0; i < INFO; ++i) {
        for(int j = 0; j < m; ++j) info_var[i] += (info[j][i] - info_mean[i]) * (info[j][i] - info_mean[i]);
        info_var[i] /= max(1, m - 1);
    }
    for(int i = 0; i < INFO; ++i) info_std[i] = sqrtl(info_var[i]) + EPS;
    info_mean[0] = 0.0;
    info_std[0] = 1.0;
    for(int i = 0; i < m; ++i) for(int j = 0; j < INFO; ++j) info[i][j] = (info[i][j] - info_mean[j]) / info_std[j];
    vector <double> weighting(m, 1.0);
    vector <vector <double>> ans(S, vector <double>(INFO));
    for(int rep = 0; rep < IRLS_REPETITIONS; ++rep) {
        vector <vector <double>> A(INFO, vector <double>(INFO, 0.0));
        vector <vector <double>> B(INFO, vector <double>(S, 0.0));
        for(int i = 0; i < m; ++i) {
            for(int u = 0; u < INFO; ++u) for(int v = 0; v < INFO; ++v) A[u][v] += info[i][u] * info[i][v] * weighting[i];
            for(int p = 0; p < S; ++p) for(int u = 0; u < INFO; ++u) B[u][p] += info[i][u] * (ana_waveform_1[i * S + p] - means[idx_of(dig_data_1[i])][p]) * weighting[i];
        }
        double mean = 0.0;
        for(int i = 0; i < INFO; ++i) mean += A[i][i];
        mean /= INFO;
        double ridge = RIDGE_FACTOR * max(EPS, mean);
        for(int i = 0; i < INFO; ++i) A[i][i] += ridge;
        cholesky_solve(A, B);
        for(int p = 0; p < S; ++p) for(int u = 0; u < INFO; ++u) ans[p][u] = B[u][p];
        vector <double> residuals(m, 0.0);
        for(int i = 0; i < m; ++i) {
            double total = 0.0;
            for(int p = 0; p < S; ++p) {
                double sum = means[idx_of(dig_data_1[i])][p];
                for(int u = 0; u < INFO; ++u) sum += ans[p][u] * info[i][u];
                residuals[i] += fabs(sum - ana_waveform_1[i * S + p]);
            }
            residuals[i] /= S;
        }
        vector <double> tmp = residuals;
        sort(begin(tmp), end(tmp));
        double med = tmp[m / 2];
        vector <double> abs_dev(m);
        for(int i = 0; i < m; ++i) abs_dev[i] = fabs(residuals[i] - med);
        sort(begin(abs_dev), end(abs_dev));
        double huber = 0.8 * ((abs_dev[m / 2] / 0.67449) + EPS);
        if(huber <= 0) huber = 1e-6;
        double max_change = 0.0;
        for(int i = 0; i < m; ++i) {
            double new_weight = 1.0;
            if(residuals[i] > huber) new_weight = huber / residuals[i];
            if(new_weight < 1e-4) new_weight = 1e-4;
            max_change = max(max_change, fabs(new_weight - weighting[i]));
            weighting[i] = new_weight;
        }
        if(max_change < 1e-6) break;
    }
    for(int rep = 0; rep < SMOOTHING_REPETITIONS; ++rep) {
        for(int u = 0; u < INFO; ++u) {
            vector <double> tmp(S);
            for(int p = 0; p < S; ++p) tmp[p] = ans[p][u];
            for(int p = 1; p + 1 < S; ++p) ans[p][u] = 0.25 * tmp[p - 1] + 0.5 * tmp[p] + 0.25 * tmp[p + 1];
        }
    }
    int L = k * S;
    cout << L << '\n';
    vector <double> res;
    for(int i = 0; i < k; ++i) {
        vector <double> cur_info(INFO);
        int idx = 0;
        cur_info[idx++] = 1.0;
        for(auto neigh : neighbours_taken) {
            double val = zero_padding(i + neigh, dig_data_2);
            cur_info[idx++] = val;
            cur_info[idx++] = val * val * val;
        }
        double prev = zero_padding(i - 1, dig_data_2);
        double cur = dig_data_2[i];
        double nxt = zero_padding(i + 1, dig_data_2);
        cur_info[idx++] = prev * cur;
        cur_info[idx++] = cur * nxt;
        for(int u = 0; u < INFO; ++u) cur_info[u] = (cur_info[u] - info_mean[u]) / info_std[u];
        for(int p = 0; p < S; ++p) {
            double sum = means[idx_of(dig_data_2[i])][p];
            for(int u = 0; u < INFO; ++u) sum += ans[p][u] * cur_info[u];
            res.push_back(sum);
        }
    }
    for(int i = 0; i < L; ++i) cout << fixed << setprecision(6) << res[i] << " \n"[i == L - 1];
    return 0;
}