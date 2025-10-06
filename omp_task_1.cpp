#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include <omp.h>

using namespace std;

// Successive matrix multiplication
void mtxMultSucc(const vector<vector<int> >& A,
                 const vector<vector<int> >& B, 
                 vector<vector<int> >& C) {
    int M = A.size();
    int N = B.size();
    int K = B[0].size();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Parallel matrix multiplication
void mtxMultPar(const vector<vector<int> >& A,
                const vector<vector<int> >& B, 
                vector<vector<int> >& C,
                int threads) {
    int M = A.size();
    int N = B.size();
    int K = B[0].size();

    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Initialization of matrix with random values
void mtxRandInit(vector<vector<int> >& mtx) {
    int rows = mtx.size();
    int columns = mtx[0].size();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            mtx[i][j] = rand() % 100;
        }
    }
}

// Saving data to .csv then for creating plots in .py
void saveToFile(const string& filename, 
                const vector<int>& M, const vector<int>& N, const vector<int>& K,
                const vector<int>& P, const vector<double>& Tp, 
                const vector<double>& S, const vector<double>& E,
                const vector<unsigned long long>& W, const vector<double>& Cost,
                const vector<double>& T0) {
    ofstream file(filename);    
    // The 1st string is header
    file << "M,N,K,P,Tp,S,E,W,Cost,T0" << endl;
    // Next strings are data
    for (size_t i = 0; i < M.size(); i++) {
        file << M[i] << "," << N[i] << "," << K[i] << "," 
             << P[i] << "," << Tp[i] << "," << S[i] << "," << E[i] << ","
             << W[i] << "," << Cost[i] << "," << T0[i] << endl;
    }
    
    file.close();
    cout << "Successful saving" << endl;
}

int main() {   
    int base_M, base_N, base_K, max_P;
    cout << "Enter base dimensions (M,N,K) and maximum threads value P: ";
    cin >> base_M >> base_N >> base_K >> max_P;
    
    // Vectors for strong scaling
    vector<int> M_strong, N_strong, K_strong, P_strong;
    vector<double> T1_strong, Tp_strong, S_strong, E_strong, Cost_strong, T0_strong;
    vector<unsigned long long> W_strong;
    
    // Vectors for weak scaling
    vector<int> M_weak, N_weak, K_weak, P_weak;
    vector<double> T1_weak, Tp_weak, S_weak, E_weak, Cost_weak, T0_weak;
    vector<unsigned long long> W_weak;

    cout << "****************************** STRONG SCALING RESEARCH ******************************" << endl;
    // STRONG SCALING (W ≈ const)
    int M_fixed = base_M, N_fixed = base_N, K_fixed = base_K;
    unsigned long long W_fixed = M_fixed * N_fixed * K_fixed * 2ULL;
    
    vector<vector<int> > A_strong(M_fixed, vector<int>(N_fixed));
    vector<vector<int> > B_strong(N_fixed, vector<int>(K_fixed));
    vector<vector<int> > C_succ_strong(M_fixed, vector<int>(K_fixed, 0));
    vector<vector<int> > C_par_strong(M_fixed, vector<int>(K_fixed, 0));
    
    srand(time(NULL));
    mtxRandInit(A_strong);
    mtxRandInit(B_strong);
    
    // Time T1 of succesive multiplication
    double timeStartSucc_strong = omp_get_wtime();
    mtxMultSucc(A_strong, B_strong, C_succ_strong);
    double timeEndSucc_strong = omp_get_wtime();
    double T1_fixed = timeEndSucc_strong - timeStartSucc_strong;
    
    cout << "Fixed dimensions: " << M_fixed << " * " << N_fixed << " * " << K_fixed << endl
        << "W = " << W_fixed << " operations, T1 = " << T1_fixed << " seconds" << endl;
    
    // Trying different thread num for strong scaling
    for (int P = 1; P <= max_P; P++) {
        for (int row = 0; row < M_fixed; row++) {
            for (int col = 0; col < K_fixed; col++) {
                C_par_strong[row][col] = 0;
            }
        }
        // Time Tp of parallel multiplication
        double timeStartPar = omp_get_wtime();
        mtxMultPar(A_strong, B_strong, C_par_strong, P);
        double timeEndPar = omp_get_wtime();
        double Tp = timeEndPar - timeStartPar;
        // Calculating metrics
        double S = T1_fixed / Tp;      // Speedup
        double E = S / P;              // Efficiency
        double Cost = P * Tp;          // Cost
        double T0 = P * T1_fixed * Tp; // Total overhead
        
        // Save results
        M_strong.push_back(M_fixed);
        N_strong.push_back(N_fixed);
        K_strong.push_back(K_fixed);
        P_strong.push_back(P);
        T1_strong.push_back(T1_fixed);
        Tp_strong.push_back(Tp);
        S_strong.push_back(S);
        E_strong.push_back(E);
        Cost_strong.push_back(Cost);
        T0_strong.push_back(T0);
        W_strong.push_back(W_fixed);
        
        cout << "P=" << P << ": Tp=" << Tp << "s, S=" << S << ", E=" << E << endl;
    }

    cout << "\n****************************** WEAK SCALING RESEARCH ******************************" << endl;
    // WEAK SCALING (W/p ≈ const)
    for (int P = 1; P <= max_P; P++) {
        // Calculating cube root dimentions
        int M_curr = base_M * pow(P, 1.0/3.0);
        int N_curr = base_N * pow(P, 1.0/3.0);
        int K_curr = base_K * pow(P, 1.0/3.0);
        
        vector<vector<int> > A_weak(M_curr, vector<int>(N_curr));
        vector<vector<int> > B_weak(N_curr, vector<int>(K_curr));
        vector<vector<int> > C_succ_weak(M_curr, vector<int>(K_curr, 0));
        vector<vector<int> > C_par_weak(M_curr, vector<int>(K_curr, 0));
        
        mtxRandInit(A_weak);
        mtxRandInit(B_weak);
        // W
        unsigned long long W_curr = M_curr * N_curr * K_curr * 2ULL;
        // T1
        double timeStartSucc_weak = omp_get_wtime();
        mtxMultSucc(A_weak, B_weak, C_succ_weak);
        double timeEndSucc_weak = omp_get_wtime();
        double T1_curr = timeEndSucc_weak - timeStartSucc_weak;
        // Tp
        double timeStartPar_weak = omp_get_wtime();
        mtxMultPar(A_weak, B_weak, C_par_weak, P);
        double timeEndPar_weak = omp_get_wtime();
        double Tp_curr = timeEndPar_weak - timeStartPar_weak;
        // Metrics
        double S_curr = T1_curr / Tp_curr;
        double E_curr = S_curr / P;
        double Cost_curr = P * Tp_curr;
        double T0_curr = P * Tp_curr * T1_curr;
        // Saving results
        M_weak.push_back(M_curr);
        N_weak.push_back(N_curr);
        K_weak.push_back(K_curr);
        P_weak.push_back(P);
        T1_weak.push_back(T1_curr);
        Tp_weak.push_back(Tp_curr);
        S_weak.push_back(S_curr);
        E_weak.push_back(E_curr);
        Cost_weak.push_back(Cost_curr);
        T0_weak.push_back(T0_curr);
        W_weak.push_back(W_curr);
        
        cout << "P=" << P << ", Size=" << M_curr << "*" << N_curr << "*" << K_curr 
            << ": T1=" << T1_curr << "s, Tp=" << Tp_curr << "s, S=" << S_curr << ", E=" << E_curr << endl;
    }

    // Saving data to .csv
    saveToFile("strong_scaling_data.csv", M_strong, N_strong, K_strong, P_strong, 
               Tp_strong, S_strong, E_strong, W_strong, Cost_strong, T0_strong);
    
    saveToFile("weak_scaling_data.csv", M_weak, N_weak, K_weak, P_weak, 
               Tp_weak, S_weak, E_weak, W_weak, Cost_weak, T0_weak);

    cout << "\n" << "Data is successfully saved to .csv files!" << endl;

    return 0;
}
