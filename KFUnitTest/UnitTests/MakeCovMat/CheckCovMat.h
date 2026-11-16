#pragma once
#include <array>
#include <iostream>
#include <vector>
R__ADD_INCLUDE_PATH(/ usr / include / eigen3)
#include <Eigen/Dense>

struct CovMatrixCheckStats {
  int negative_diagonal_count       = 0;
  int fixed_negative_diagonal_count = 0;

  int cauchy_schwarz_violations  = 0;
  int fixed_cauchy_schwarz_count = 0;

  unsigned long long psd_checked_count = 0;
  unsigned long long psd_failed_count = 0;

  void PrintSummary(bool fixed = false) const
  {
    std::cout << "\n=== Covariance Matrix Check Summary ===\n";
    std::cout << "→ Negative diagonal elements found: " << negative_diagonal_count;
    if (fixed) { std::cout << " | Fixed: " << fixed_negative_diagonal_count; }
    std::cout << "\n";

    std::cout << "→ Cauchy-Schwarz violations: " << cauchy_schwarz_violations;
    if (fixed) { std::cout << " | Fixed: " << fixed_cauchy_schwarz_count; }
    std::cout << "\n";
    std::cout << "→ Cholesky checks failed (after optional fixes): "
              << psd_failed_count << " / " << psd_checked_count
              << " (" << (psd_checked_count ? 100. * psd_failed_count / psd_checked_count : 0.)
              << "%)\n";
    std::cout << "\n";
    std::cout << "=======================================\n";
  }
};

template<typename T, int l>
bool CheckCovMat(T (&matrix)[l], CovMatrixCheckStats& stats, bool silent_mode = true, bool check_psd = true,
                 bool fix = true)
{
  float n = sqrt(l);
  if (n != floor(n)) {
    if (!silent_mode) {
      std::cout << "Warning! Covariance matrix has invalid format. Its length L = " << l
                << ", hence it cannot be interpreted as a square matrix. A square "
                   "simmetrical NxN matrix must be converted into vector with the "
                   "length L=(N+1)*N/2.\n";
    }
    return false;
  }

  bool res_flag   = true;
  float tolerance = 1;
  float epsilon   = 1e-6;

  size_t n_int = (size_t) n;
  for (size_t i = 0; i < n_int; i++) {
    size_t diag_el_num = (i + 1) * (i + 2) / 2 - 1;
    if (matrix[diag_el_num] < 0) {
      stats.negative_diagonal_count++;
      if (fix) {
        matrix[diag_el_num] = epsilon;
        stats.fixed_negative_diagonal_count++;
        if (!silent_mode) { std::cout << "→ Fixed to " << matrix[diag_el_num] << "\n"; }
      }
      else {
        res_flag = false;
      }
      if (!silent_mode) {
        std::cout << "Warning! Covariance matrix has negative diagonal element #" << diag_el_num << " with the value "
                  << matrix[diag_el_num] << ". Please, check the matrix.\n";
      }
    }
  }

  for (size_t i = 0; i < n_int; i++) {
    for (size_t j = 0; j < i; j++) {
      size_t ij   = i * (i + 1) / 2 + j;
      size_t ii   = (i + 1) * (i + 2) / 2 - 1;
      size_t jj   = (j + 1) * (j + 2) / 2 - 1;
      float c_ij  = matrix[ij];
      float c_ii  = matrix[ii];
      float c_jj  = matrix[jj];
      float limit = tolerance * sqrt(c_ii * c_jj);
      if (abs(c_ij) > tolerance * sqrt(c_ii * c_jj)) {
        stats.cauchy_schwarz_violations++;
        if (!silent_mode) {
          std::cout << "Warning! Covariance matrix has non-diagonal element C" << ij << " = " << c_ij
                    << " does not satisfy the condition abs(Cij) < sqrt(Cii*Cjj) (C" << ii << " = " << c_ii << ", C"
                    << jj << " = " << c_jj << "). Please, check the matrix.\n";
        }

        if (fix) {
          matrix[ij] = (c_ij > 0 ? 1 : -1) * limit;
          stats.fixed_cauchy_schwarz_count++;
          if (!silent_mode) { std::cout << "→ Fixed to " << matrix[ij] << "\n"; }
        }
        else {
          res_flag = false;
        }
      }
    }
  }

  // Full PSD check using Eigen's Cholesky decomposition
  if (check_psd) {
    Eigen::MatrixXf full_cov = Eigen::MatrixXf::Zero(n_int, n_int);

    // Fill upper triangle and mirror to lower
    for (size_t i = 0; i < n_int; i++) {
      for (size_t j = 0; j <= i; j++) {
        size_t index   = i * (i + 1) / 2 + j;
        full_cov(i, j) = full_cov(j, i) = matrix[index];
      }
    }

    ++stats.psd_checked_count;

    // Attempt Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXf> llt(full_cov);
    if (llt.info() != Eigen::Success) {
    ++stats.psd_failed_count;
      if (!silent_mode) {
        std::cout << "Warning! Full covariance matrix is not positive "
                     "semi-definite (PSD). Cholesky decomposition failed.\n";
      }
      // res_flag = false;
    }
    else {
      if (!silent_mode) { std::cout << "Full covariance matrix is positive semi-definite (PSD).\n"; }
    }
  }

  if (res_flag) {
    if (!silent_mode) { std::cout << "Covariance matrix is OK!\n"; }
  }
  return res_flag;
}
