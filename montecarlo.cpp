#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>

// -------------------- Parameters --------------------
const int N = 100;              // Lattice size (NxN grid)
const double J = 1.0;           // Interaction strength (ferromagnetic)
const double k_B = 1.0;         // Boltzmann constant
const int STEPS = 500000;       // Total Monte Carlo steps
const int BURNIN = 75000;       // Burn-in period (initial steps discarded)

// Random number generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist(0.0, 1.0);
std::uniform_int_distribution<int> lattice_dist(0, N-1);

// -------------------- Compute Energy --------------------
double compute_energy(const std::vector<std::vector<int>>& lattice, double B) {
    double energy = 0.0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Nearest neighbors with periodic boundary conditions
            int i_up = (i + 1) % N;
            int i_down = (i - 1 + N) % N;
            int j_right = (j + 1) % N;
            int j_left = (j - 1 + N) % N;
            
            // Sum interaction energy (only count each pair once)
            // Only consider right and down neighbors to avoid double counting
            energy += -J * lattice[i][j] * (lattice[i_up][j] + lattice[i][j_right]);
            
            // External field contribution
            energy += -B * lattice[i][j];
        }
    }
    
    return energy;
}

// -------------------- Metropolis Algorithm --------------------
double metropolis(std::vector<std::vector<int>>& lattice, double T, double B, int steps) {
    // Returns mean magnetization after burn-in
    std::vector<double> magnetization;
    int num_accept = 0;
    
    for (int step = 0; step < steps; step++) {
        // Pick random spin
        int i = lattice_dist(gen);
        int j = lattice_dist(gen);
        
        // Calculate energy change for flipping this spin
        double delta_E = 0.0;
        int current_spin = lattice[i][j];
        
        // Check all four neighbors
        int i_up = (i + 1) % N;
        int i_down = (i - 1 + N) % N;
        int j_right = (j + 1) % N;
        int j_left = (j - 1 + N) % N;
        
        // Sum over neighbors - the -2 factor comes from the fact that flipping a spin changes S_i to -S_i
        // so the change in S_i*S_j is -2*S_i*S_j
        delta_E += 2 * J * current_spin * (
            lattice[i_up][j] + 
            lattice[i_down][j] + 
            lattice[i][j_right] + 
            lattice[i][j_left]
        );
        
        // External field contribution - when we flip the spin, B*S_i becomes B*(-S_i)
        // so the change is -2*B*S_i
        delta_E += 2 * B * current_spin;
        
        // Metropolis acceptance rule
        if (delta_E <= 0 || dist(gen) < std::exp(-delta_E / (k_B * T))) {
            lattice[i][j] *= -1;  // Flip spin
            num_accept++;
        }
        
        // Calculate and store magnetization per spin
        if (step >= BURNIN) {
            double m = 0.0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    m += lattice[i][j];
                }
            }
            magnetization.push_back(m / (N * N));
        }
    }
    
    // Calculate acceptance rate
    double acceptance_rate = static_cast<double>(num_accept) / steps;
    std::cout << "Acceptance rate at T=" << T << ", B=" << B << ": " 
              << std::fixed << std::setprecision(4) << acceptance_rate << std::endl;
    
    // Calculate mean magnetization after burn-in
    double mean_mag = 0.0;
    for (const auto& m : magnetization) {
        mean_mag += m;
    }
    mean_mag /= magnetization.size();
    
    return mean_mag;
}

// -------------------- Initialize Lattice --------------------
std::vector<std::vector<int>> initialize_lattice(std::string mode = "ordered") {
    std::vector<std::vector<int>> lattice(N, std::vector<int>(N));
    
    if (mode == "ordered") {
        // All spins up (+1)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                lattice[i][j] = 1;
            }
        }
    } else if (mode == "random") {
        // Random configuration of spins
        std::bernoulli_distribution bd(0.5);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                lattice[i][j] = bd(gen) ? 1 : -1;
            }
        }
    }
    
    return lattice;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Temperature and magnetic field ranges
    const int num_T = 50;
    const int num_B = 50;
    std::vector<double> T_vals(num_T);
    std::vector<double> B_vals(num_B);
    
    double T_min = 1.0, T_max = 6.0;
    double B_min = -2.0, B_max = 2.0;
    
    // Initialize temperature and field arrays
    for (int i = 0; i < num_T; i++) {
        T_vals[i] = T_min + i * (T_max - T_min) / (num_T - 1);
    }
    
    for (int i = 0; i < num_B; i++) {
        B_vals[i] = B_min + i * (B_max - B_min) / (num_B - 1);
    }
    
    // Initialize magnetization array
    std::vector<std::vector<double>> M(num_B, std::vector<double>(num_T, 0.0));
    
    // Perform simulations for each temperature and field
    for (int i = 0; i < num_T; i++) {
        double T = T_vals[i];
        
        for (int j = 0; j < num_B; j++) {
            double B = B_vals[j];
            
            // Initialize with random configuration
            std::vector<std::vector<int>> lattice = initialize_lattice("random");
            
            // Run Metropolis algorithm
            double mean_mag = metropolis(lattice, T, B, STEPS);
            M[j][i] = mean_mag;
            
            std::cout << "T = " << T << ", B = " << B 
                      << ", Mean magnetization = " << std::fixed << std::setprecision(4) 
                      << mean_mag << std::endl;
        }
    }
    
    // Save results to file
    std::ofstream outfile("magnetization_data.txt");
    outfile << "T,B,Magnetization" << std::endl;
    
    for (int i = 0; i < num_T; i++) {
        for (int j = 0; j < num_B; j++) {
            outfile << T_vals[i] << "," << B_vals[j] << "," << M[j][i] << std::endl;
        }
    }
    outfile.close();
    
    // Print execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Execution time: " << duration << " seconds" << std::endl;
    
    std::cout << "Saved magnetization data to magnetization_data.txt" << std::endl;
    std::cout << "To visualize results, use a plotting program or the provided Python script." << std::endl;
    
    return 0;
}