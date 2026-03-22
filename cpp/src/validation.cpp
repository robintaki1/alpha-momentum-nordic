#include "am/validation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace am {
namespace {

double mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double stdev(const std::vector<double>& values) {
  if (values.size() < 2) {
    return 0.0;
  }
  double mu = mean(values);
  double accum = 0.0;
  for (double value : values) {
    double diff = value - mu;
    accum += diff * diff;
  }
  return std::sqrt(accum / static_cast<double>(values.size() - 1));
}

double normal_cdf(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double inverse_normal_cdf(double p) {
  // Acklam's approximation.
  if (p <= 0.0) return -INFINITY;
  if (p >= 1.0) return INFINITY;

  static const double a1 = -3.969683028665376e+01;
  static const double a2 = 2.209460984245205e+02;
  static const double a3 = -2.759285104469687e+02;
  static const double a4 = 1.383577518672690e+02;
  static const double a5 = -3.066479806614716e+01;
  static const double a6 = 2.506628277459239e+00;

  static const double b1 = -5.447609879822406e+01;
  static const double b2 = 1.615858368580409e+02;
  static const double b3 = -1.556989798598866e+02;
  static const double b4 = 6.680131188771972e+01;
  static const double b5 = -1.328068155288572e+01;

  static const double c1 = -7.784894002430293e-03;
  static const double c2 = -3.223964580411365e-01;
  static const double c3 = -2.400758277161838e+00;
  static const double c4 = -2.549732539343734e+00;
  static const double c5 = 4.374664141464968e+00;
  static const double c6 = 2.938163982698783e+00;

  static const double d1 = 7.784695709041462e-03;
  static const double d2 = 3.224671290700398e-01;
  static const double d3 = 2.445134137142996e+00;
  static const double d4 = 3.754408661907416e+00;

  double q = 0.0;
  if (p < 0.02425) {
    q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > 1.0 - 0.02425) {
    q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  q = p - 0.5;
  double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double sample_skewness(const std::vector<double>& values) {
  if (values.size() < 3) return 0.0;
  double mu = mean(values);
  double sd = stdev(values);
  if (sd == 0.0) return 0.0;
  double accum = 0.0;
  for (double value : values) {
    accum += std::pow((value - mu) / sd, 3.0);
  }
  return accum / static_cast<double>(values.size());
}

double sample_kurtosis(const std::vector<double>& values) {
  if (values.size() < 4) return 3.0;
  double mu = mean(values);
  double sd = stdev(values);
  if (sd == 0.0) return 3.0;
  double accum = 0.0;
  for (double value : values) {
    accum += std::pow((value - mu) / sd, 4.0);
  }
  return accum / static_cast<double>(values.size());
}

std::vector<double> stationary_bootstrap_sample(
    const std::vector<double>& values,
    int mean_block_length,
    std::mt19937& rng) {
  std::vector<double> sample;
  if (values.empty()) return sample;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::uniform_int_distribution<int> pick(0, static_cast<int>(values.size() - 1));
  double p = 1.0 / std::max(1, mean_block_length);
  int index = pick(rng);
  sample.reserve(values.size());
  for (size_t pos = 0; pos < values.size(); ++pos) {
    if (pos == 0 || uni(rng) < p) {
      index = pick(rng);
    } else {
      index = (index + 1) % static_cast<int>(values.size());
    }
    sample.push_back(values[index]);
  }
  return sample;
}

double quantile(const std::vector<double>& sorted, double probability) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted[0];
  double index = probability * (sorted.size() - 1);
  auto lower = static_cast<size_t>(std::floor(index));
  auto upper = static_cast<size_t>(std::ceil(index));
  if (lower == upper) return sorted[lower];
  double weight = index - lower;
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

}  // namespace

double annualized_sharpe(const std::vector<double>& returns, int periods_per_year) {
  if (returns.size() < 2) return 0.0;
  double sd = stdev(returns);
  if (sd == 0.0) return 0.0;
  return mean(returns) / sd * std::sqrt(static_cast<double>(periods_per_year));
}

double max_drawdown(const std::vector<double>& returns) {
  double equity = 1.0;
  double peak = 1.0;
  double max_dd = 0.0;
  for (double value : returns) {
    equity *= (1.0 + value);
    peak = std::max(peak, equity);
    double drawdown = 1.0 - (equity / peak);
    max_dd = std::max(max_dd, drawdown);
  }
  return max_dd;
}

double total_return(const std::vector<double>& returns) {
  double equity = 1.0;
  for (double value : returns) {
    equity *= (1.0 + value);
  }
  return equity - 1.0;
}

DeflatedSharpeMetrics deflated_sharpe_metrics(const std::vector<double>& returns, int n_trials) {
  DeflatedSharpeMetrics metrics;
  if (returns.size() < 3 || n_trials < 1) {
    return metrics;
  }
  double sd = stdev(returns);
  if (sd == 0.0) return metrics;
  double monthly_sharpe = mean(returns) / sd;
  double skewness = sample_skewness(returns);
  double kurtosis = sample_kurtosis(returns);
  double variance_term = 1.0 - skewness * monthly_sharpe + ((kurtosis - 1.0) / 4.0) * monthly_sharpe * monthly_sharpe;
  variance_term = std::max(variance_term, 1e-12);
  double sharpe_std = std::sqrt(variance_term / std::max<int>(static_cast<int>(returns.size()) - 1, 1));
  double expected_max_noise_sharpe = 0.0;
  if (n_trials > 1) {
    double gamma = 0.5772156649;
    double first = inverse_normal_cdf(1.0 - 1.0 / n_trials);
    double second = inverse_normal_cdf(1.0 - 1.0 / (n_trials * std::exp(1.0)));
    expected_max_noise_sharpe = sharpe_std * ((1.0 - gamma) * first + gamma * second);
  }
  metrics.score = monthly_sharpe - expected_max_noise_sharpe;
  metrics.expected_max_noise_sharpe = expected_max_noise_sharpe;
  metrics.probability = sharpe_std > 0.0 ? normal_cdf(metrics.score / sharpe_std) : 0.0;
  return metrics;
}

std::pair<double, double> stationary_bootstrap_sharpe_ci(
    const std::vector<double>& returns,
    int mean_block_length,
    int n_resamples,
    int seed) {
  if (returns.size() < 2) {
    return {0.0, 0.0};
  }
  std::mt19937 rng(seed);
  std::vector<double> sharpe_values;
  sharpe_values.reserve(n_resamples);
  for (int i = 0; i < n_resamples; ++i) {
    auto sample = stationary_bootstrap_sample(returns, mean_block_length, rng);
    sharpe_values.push_back(annualized_sharpe(sample));
  }
  std::sort(sharpe_values.begin(), sharpe_values.end());
  return {quantile(sharpe_values, 0.025), quantile(sharpe_values, 0.975)};
}

std::vector<std::vector<double>> cross_sectional_score_shuffle_runs(
    const std::vector<std::vector<std::pair<double, double>>>& months,
    int top_n,
    int n_runs,
    int seed) {
  std::mt19937 rng(seed);
  std::vector<std::vector<double>> runs;
  runs.reserve(n_runs);
  for (int i = 0; i < n_runs; ++i) {
    std::vector<double> run_returns;
    for (const auto& month : months) {
      std::vector<std::pair<double, double>> shuffled = month;
      std::shuffle(shuffled.begin(), shuffled.end(), rng);
      std::sort(shuffled.begin(), shuffled.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      double sum = 0.0;
      int take = std::min<int>(top_n, static_cast<int>(shuffled.size()));
      for (int j = 0; j < take; ++j) {
        sum += shuffled[j].second;
      }
      run_returns.push_back(take > 0 ? sum / static_cast<double>(take) : 0.0);
    }
    runs.push_back(std::move(run_returns));
  }
  return runs;
}

std::vector<std::vector<double>> block_shuffled_return_path_runs(
    const std::vector<double>& returns,
    int block_length,
    int n_runs,
    int seed) {
  if (block_length <= 0) {
    throw std::runtime_error("block_length must be positive");
  }
  std::vector<std::vector<double>> runs;
  if (returns.empty()) {
    return runs;
  }
  std::mt19937 rng(seed);
  std::vector<std::vector<double>> blocks;
  for (size_t i = 0; i < returns.size(); i += block_length) {
    blocks.emplace_back(returns.begin() + i,
                        returns.begin() + std::min(returns.size(), i + block_length));
  }
  for (int i = 0; i < n_runs; ++i) {
    std::vector<std::vector<double>> shuffled = blocks;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    std::vector<double> run;
    for (const auto& block : shuffled) {
      run.insert(run.end(), block.begin(), block.end());
    }
    run.resize(returns.size());
    runs.push_back(std::move(run));
  }
  return runs;
}

double negative_control_pass_rate(
    const std::vector<std::vector<double>>& return_runs,
    int n_trials,
    double sharpe_threshold,
    int bootstrap_resamples,
    int seed) {
  if (return_runs.empty()) {
    return 1.0;
  }
  int pass_count = 0;
  int run_index = 0;
  for (const auto& run : return_runs) {
    double sharpe = annualized_sharpe(run);
    auto ci = stationary_bootstrap_sharpe_ci(run, 6, bootstrap_resamples, seed + run_index);
    auto dsr = deflated_sharpe_metrics(run, n_trials);
    if (sharpe > sharpe_threshold && ci.first > 0.0 && dsr.score > 0.0) {
      pass_count += 1;
    }
    run_index += 1;
  }
  return static_cast<double>(pass_count) / static_cast<double>(return_runs.size());
}

}  // namespace am
