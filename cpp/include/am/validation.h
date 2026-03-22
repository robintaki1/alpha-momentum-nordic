#pragma once

#include <string>
#include <utility>
#include <vector>

namespace am {

double annualized_sharpe(const std::vector<double>& returns, int periods_per_year = 12);
double max_drawdown(const std::vector<double>& returns);
double total_return(const std::vector<double>& returns);

struct DeflatedSharpeMetrics {
  double score = 0.0;
  double probability = 0.0;
  double expected_max_noise_sharpe = 0.0;
};

DeflatedSharpeMetrics deflated_sharpe_metrics(const std::vector<double>& returns, int n_trials);

std::pair<double, double> stationary_bootstrap_sharpe_ci(
    const std::vector<double>& returns,
    int mean_block_length,
    int n_resamples,
    int seed);

std::vector<std::vector<double>> cross_sectional_score_shuffle_runs(
    const std::vector<std::vector<std::pair<double, double>>>& months,
    int top_n,
    int n_runs,
    int seed);

std::vector<std::vector<double>> block_shuffled_return_path_runs(
    const std::vector<double>& returns,
    int block_length,
    int n_runs,
    int seed);

double negative_control_pass_rate(
    const std::vector<std::vector<double>>& return_runs,
    int n_trials,
    double sharpe_threshold,
    int bootstrap_resamples,
    int seed);

}  // namespace am
