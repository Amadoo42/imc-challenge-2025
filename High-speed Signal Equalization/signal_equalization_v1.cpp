#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
const int SMOOTHING_REPETITIONS = 7, SMOOTHING_RADIUS = 4;
long long C[SMOOTHING_RADIUS << 1 | 1][SMOOTHING_RADIUS << 1 | 1];

struct KMeans1D {
    static void run(const vector<ld>& xs,
                    array<ld,4>& centers_out,
                    array<int,4>& label_of_ranked_center) {
        const int K = 4;
        const int n = (int)xs.size();
        ld mn = *min_element(xs.begin(), xs.end());
        ld mx = *max_element(xs.begin(), xs.end());
        array<ld,4> c = {mn + (mx-mn)*0.05,
                             mn + (mx-mn)*0.3,
                             mn + (mx-mn)*0.7,
                             mn + (mx-mn)*0.95};

        array<ld,4> nc; array<int,4> cnt;
        for (int it = 0; it < 57; ++it) {
            nc.fill(0.0); cnt.fill(0);
            for (ld v : xs) {
                int best = 0;
                ld bestd = fabs(v - c[0]);
                for (int k = 1; k < K; ++k) {
                    ld d = fabs(v - c[k]);
                    if (d < bestd) { bestd = d; best = k; }
                }
                nc[best] += v; cnt[best]++;
            }
            ld gmean = accumulate(xs.begin(), xs.end(), 0.0) / max(1, n);
            for (int k = 0; k < K; ++k) {
                if (cnt[k] == 0) c[k] = (c[k] + gmean) * 0.5;
                else c[k] = nc[k] / cnt[k];
            }
        }

        array<pair<ld,int>,4> tmp;
        for (int k = 0; k < K; ++k) tmp[k] = {c[k], k};
        sort(tmp.begin(), tmp.end(),
             [](auto &a, auto &b){ return a.first < b.first; });
        for (int r = 0; r < K; ++r) label_of_ranked_center[r] = tmp[r].second;
        for (int r = 0; r < K; ++r) centers_out[r] = tmp[r].first;
    }
};
map<int,vector<double>>q;
vector<double>y;
vector<double>z;
int Lq=3;
double u=0.4;
double mxu=0.4;
double get_nu(int m)
{
    double nu = max(0.000001,(double)((int)z.size()-m*80)*mxu/double(z.size()));
    return nu;
}
void get_q_m1(int m)
{
    for(int l=-Lq;l<=Lq;l++)
    {
        double ret=1;
        if(m>0)ret=q[l][m-1];
        double ry=1;
        if(m>0)ry=y[m-1];
        double rz=1;
        if(m-l-1>=0 && m-l-1<(int)z.size())rz=z[m-l-1];
        ret-=u*ry*rz*(ry*ry-1.00);
        u=get_nu(m);

        q[l].push_back(ret);
    }
}
double calc_y(int m)
{
    double ry=0;
    for(int k=0;k<Lq;k++)
    {
        double rz = 0;
        if(m-k>=0 && m-k<(int)z.size())rz=z[m-k];
        ry+=q[k][m]*rz;
    }
    y.push_back(ry);
    return ry;
}
vector<double>filter_wave(vector<double>ana_wave)
{
    for(int i=0;i<ana_wave.size();i++)
    {
        get_q_m1(i);
        calc_y(i);
    }
    return y;
}
mt19937 rnd(57);

int rng(int l, int r) {
    return rnd() % (r - l + 1) + l;
}
ld arr[5000010];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<double> ana(n);
    ld mxx=-5,mnn=5;
    for(int i=0;i<n;i++){
        string s;
        cin>>s;
        arr[i]=stod(s);
        mxx=max(mxx,arr[i]);
        mnn=min(mnn,arr[i]);
    }

    for (int i = 0; i < n; i++) {
        ana[i]=arr[i];
    }
    z=ana;
    ana=filter_wave(ana);
    double mx=-1;
    for(auto i:ana)mx=max(mx,abs(i));
    for(auto &i:ana)i/=mx;
    //solve(ana);
    ld mn;
    /*for(int p = SMOOTHING_RADIUS; p + SMOOTHING_RADIUS < n; ++p) {
        double weighted_average_sum = 0.0;
        long long denom = (1LL << (SMOOTHING_RADIUS << 1));
        for(int j = 0; j <= (SMOOTHING_RADIUS << 1); ++j) weighted_average_sum += (double)C[SMOOTHING_RADIUS << 1][j] / denom * ana[p - SMOOTHING_RADIUS + j];
        ana[p] = weighted_average_sum;
    }*/
    vector<int> ans, pam4 = {-3, -1, 1, 3};
    for (int ii = 0; ii < 4; ii++) {
        vector<ld> cur;
        vector<int> bruh;
        for (int i = ii; i < n; i += 4) {
            cur.push_back(ana[i]);
        }
        array<ld, 4> chabd;
        array<int, 4> pam4habd;
        KMeans1D::run(cur, chabd, pam4habd);
        ld opt = 0;
        for (int i = 0; i < n / 4; i++) {
            int id = 0;
            for (int j = 1; j < 4; j++) {
                if (fabs(cur[i] - chabd[j]) < fabs(cur[i] - chabd[id])) {
                    id = j;
                }
            }
            bruh.push_back(pam4[id]);
            opt += ld(cur[i] - chabd[id]) * ld(cur[i] - chabd[id]);
        }
        //opt = sqrt(opt / ld((n / 4)));
        if (mn > opt || ii == 0) {
            mn = opt;
            ans = bruh;
        }
    }
    int L = n / 4;
    cout << L << "\n";
    for (int i = 0; i < L; i++) {
        cout << ans[i] << " \n"[(i + 1 == L)];
    }
}

