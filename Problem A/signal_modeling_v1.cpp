#include <bits/stdc++.h>
using namespace std;

const int NEIGH = 84, CENTER_RADIUS = 15, SMOOTHING_REPETITIONS = 7, SMOOTHING_RADIUS = 1;
const double RIDGE = 10;
const int S = 16, INFO = 20 + (2 * NEIGH + 1 + CENTER_RADIUS) * 3;

long long C[SMOOTHING_RADIUS << 1 | 1][SMOOTHING_RADIUS << 1 | 1];

void build() {
    C[0][0] = 1;
    for(int i = 1; i <= (SMOOTHING_RADIUS << 1); ++i) {
        C[i][0] = 1;
        for(int j = 1; j <= i; ++j) C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
    }
}

inline void mat_inv(vector <vector <double>> &A) {
    int n = A.size();
    vector <int> col(n);
    vector <vector <double>> tmp(n, vector <double>(n));
    for(int i = 0; i < n; ++i) tmp[i][i] = 1, col[i] = i;
    for(int i = 0; i < n; ++i) {
        int r = i, c = i;
        for(int j = i; j < n; ++j) for(int k = i; k < n; ++k) if(fabs(A[j][k]) > fabs(A[r][c])) r = j, c = k;
        A[i].swap(A[r]); tmp[i].swap(tmp[r]);
        for(int j = 0; j < n; ++j) swap(A[j][i], A[j][c]), swap(tmp[j][i], tmp[j][c]);
        swap(col[i], col[c]);
        double v = A[i][i];
        for(int j = i + 1; j < n; ++j) {
            double f = A[j][i] / v;
            A[j][i] = 0;
            for(int k = i + 1; k < n; ++k) A[j][k] -= f * A[i][k];
            for(int k = 0; k < n; ++k) tmp[j][k] -= f * tmp[i][k];
        }
        for(int j = i + 1; j < n; ++j) A[i][j] /= v;
        for(int j = 0; j < n; ++j) tmp[i][j] /= v;
        A[i][i] = 1;
    }
    for(int i = n - 1; i > 0; --i) for(int j = 0; j < i; ++j) {
        double v = A[j][i];
        for(int k = 0; k < n; ++k) tmp[j][k] -= v * tmp[i][k];
    }
    for(int i = 0; i < n; ++i) for(int j = 0; j < n; ++j) A[col[i]][col[j]] = tmp[i][j];
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n; cin >> n;
    vector <double> ana_waveform_1(n);
    for(int i = 0; i < n; ++i) cin >> ana_waveform_1[i];
    int m; cin >> m;
    vector <double> dig_data_1(m);
    for(int i = 0; i < m; ++i) cin >> dig_data_1[i];
    int k; cin >> k;
    vector <double> dig_data_2(k);
    for(int i = 0; i < k; ++i) cin >> dig_data_2[i];
    auto idx_of = [&](int v) -> int {
        if(v == -3) return 0;
        if(v == -1) return 1;
        if(v == 1) return 2;
        if(v == 3) return 3;
    };
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
    vector <vector <double>> A(INFO, vector <double>(INFO, 0.0));
    vector <vector <double>> B(S, vector <double>(INFO, 0.0));
    for(int i = 0; i < m; ++i) {
        vector <int> neigh(2 * NEIGH + 1);
        for(int p = -NEIGH; p <= NEIGH; ++p) {
            int pos = i + p;
            if(pos < 0 || pos >= m) neigh[p + NEIGH] = 0;
            else neigh[p + NEIGH] = dig_data_1[pos];
        }
        vector <double> info(INFO);
        info[0] = 1.0;
        int pos = 1;
        for(int p = 0; p < 2 * NEIGH + 1; ++p) {
            info[pos++] = (double)neigh[p];
            info[pos++] = (double)neigh[p] * neigh[p];
            info[pos++] = (double)neigh[p] * neigh[p] * neigh[p];
        }
        for(int p = 1; p <= CENTER_RADIUS; ++p) info[pos++] = neigh[NEIGH - p] * neigh[NEIGH], info[pos++] = neigh[NEIGH + p] * neigh[NEIGH];
        info[pos++] = neigh[NEIGH - 5] * neigh[NEIGH - 4];
        info[pos++] = neigh[NEIGH - 4] * neigh[NEIGH - 3];
        info[pos++] = neigh[NEIGH - 3] * neigh[NEIGH - 2];
        info[pos++] = neigh[NEIGH - 2] * neigh[NEIGH - 1];
        info[pos++] = neigh[NEIGH + 1] * neigh[NEIGH + 2];
        info[pos++] = neigh[NEIGH + 2] * neigh[NEIGH + 3];
        info[pos++] = neigh[NEIGH + 3] * neigh[NEIGH + 4];
        info[pos++] = neigh[NEIGH + 4] * neigh[NEIGH + 5];
        info[pos++] = neigh[NEIGH - 1] * neigh[NEIGH + 1];
        info[pos++] = neigh[NEIGH - 2] * neigh[NEIGH + 2];
        info[pos++] = neigh[NEIGH - 3] * neigh[NEIGH + 3];
        info[pos++] = neigh[NEIGH - 4] * neigh[NEIGH + 4];
        info[pos++] = neigh[NEIGH - 5] * neigh[NEIGH + 5];
        info[pos++] = neigh[NEIGH - 4] * neigh[NEIGH - 3] * neigh[NEIGH - 2];
        info[pos++] = neigh[NEIGH - 3] * neigh[NEIGH - 2] * neigh[NEIGH - 1];
        info[pos++] = neigh[NEIGH - 2] * neigh[NEIGH - 1] * neigh[NEIGH];
        info[pos++] = neigh[NEIGH - 1] * neigh[NEIGH] * neigh[NEIGH + 1];
        info[pos++] = neigh[NEIGH] * neigh[NEIGH + 1] * neigh[NEIGH + 2];
        info[pos++] = neigh[NEIGH + 1] * neigh[NEIGH + 2] * neigh[NEIGH + 3];
        info[pos++] = neigh[NEIGH + 2] * neigh[NEIGH + 3] * neigh[NEIGH + 4];
        for(int u = 0; u < INFO; ++u) for(int v = 0; v < INFO; ++v) A[u][v] += info[u] * info[v];
        for(int p = 0; p < S; ++p) for(int u = 0; u < INFO; ++u) B[p][u] += (ana_waveform_1[i * S + p] - means[idx_of(dig_data_1[i])][p]) * info[u];
    }
    for(int i = 0; i < INFO; ++i) A[i][i] += RIDGE;
    vector <vector <double>> A_inv = A;
    mat_inv(A_inv);
    vector <vector <double>> ans(S, vector <double>(INFO, 0.0));
    for(int p = 0; p < S; ++p) for(int i = 0; i < INFO; ++i) for(int j = 0; j < INFO; ++j) ans[p][i] += A_inv[i][j] * B[p][j];
    build();
    for(int rep = 0; rep < SMOOTHING_REPETITIONS; ++rep) {
        vector <double> tmp(S);
        for(int i = 0; i < INFO; ++i) {
            for(int p = 0; p < S; ++p) tmp[p] = ans[p][i];
            for(int p = SMOOTHING_RADIUS; p + SMOOTHING_RADIUS < S; ++p) {
                double weighted_average_sum = 0.0;
                long long denom = (1LL << (SMOOTHING_RADIUS << 1));
                for(int j = 0; j <= (SMOOTHING_RADIUS << 1); ++j) weighted_average_sum += (double)C[SMOOTHING_RADIUS << 1][j] / denom * tmp[p - SMOOTHING_RADIUS + j];
                ans[p][i] = weighted_average_sum;
            }
        }
    }
    int L = k * S;
    cout << L << '\n';
    vector <double> res(L);
    int idx = 0;
    for(int i = 0; i < k; ++i) {
        vector <int> neigh(2 * NEIGH + 1);
        for(int p = -NEIGH; p <= NEIGH; ++p) {
            int pos = i + p;
            if(pos < 0 || pos >= k) neigh[p + NEIGH] = 0;
            else neigh[p + NEIGH] = dig_data_2[pos];
        }
        vector <double> info(INFO);
        info[0] = 1.0;
        int pos = 1;
        for(int p = 0; p < 2 * NEIGH + 1; ++p) {
            info[pos++] = (double)neigh[p];
            info[pos++] = (double)neigh[p] * neigh[p];
            info[pos++] = (double)neigh[p] * neigh[p] * neigh[p];
        }
        for(int p = 1; p <= CENTER_RADIUS; ++p) info[pos++] = neigh[NEIGH - p] * neigh[NEIGH], info[pos++] = neigh[NEIGH + p] * neigh[NEIGH];
        info[pos++] = neigh[NEIGH - 5] * neigh[NEIGH - 4];
        info[pos++] = neigh[NEIGH - 4] * neigh[NEIGH - 3];
        info[pos++] = neigh[NEIGH - 3] * neigh[NEIGH - 2];
        info[pos++] = neigh[NEIGH - 2] * neigh[NEIGH - 1];
        info[pos++] = neigh[NEIGH + 1] * neigh[NEIGH + 2];
        info[pos++] = neigh[NEIGH + 2] * neigh[NEIGH + 3];
        info[pos++] = neigh[NEIGH + 3] * neigh[NEIGH + 4];
        info[pos++] = neigh[NEIGH + 4] * neigh[NEIGH + 5];
        info[pos++] = neigh[NEIGH - 1] * neigh[NEIGH + 1];
        info[pos++] = neigh[NEIGH - 2] * neigh[NEIGH + 2];
        info[pos++] = neigh[NEIGH - 3] * neigh[NEIGH + 3];
        info[pos++] = neigh[NEIGH - 4] * neigh[NEIGH + 4];
        info[pos++] = neigh[NEIGH - 5] * neigh[NEIGH + 5];
        info[pos++] = neigh[NEIGH - 4] * neigh[NEIGH - 3] * neigh[NEIGH - 2];
        info[pos++] = neigh[NEIGH - 3] * neigh[NEIGH - 2] * neigh[NEIGH - 1];
        info[pos++] = neigh[NEIGH - 2] * neigh[NEIGH - 1] * neigh[NEIGH];
        info[pos++] = neigh[NEIGH - 1] * neigh[NEIGH] * neigh[NEIGH + 1];
        info[pos++] = neigh[NEIGH] * neigh[NEIGH + 1] * neigh[NEIGH + 2];
        info[pos++] = neigh[NEIGH + 1] * neigh[NEIGH + 2] * neigh[NEIGH + 3];
        info[pos++] = neigh[NEIGH + 2] * neigh[NEIGH + 3] * neigh[NEIGH + 4];
        for(int p = 0; p < S; ++p) {
            double sum = 0.0;
            for(int j = 0; j < INFO; ++j) sum += ans[p][j] * info[j];
            res[idx++] = sum + means[idx_of(dig_data_2[i])][p];
        }
    }
    for(int i = 0; i < L; ++i) cout << fixed << setprecision(6) << res[i] * 0.999 << " \n"[i == L - 1];
    return 0;
}