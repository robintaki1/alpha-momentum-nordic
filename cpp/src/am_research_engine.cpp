#include "am/core.h"
#include "am/config.h"
#include "am/dataset.h"
#include "am/validation.h"

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <unordered_map>

namespace am {
using json = nlohmann::json;
namespace {

struct CandidateParams {
  int lookback = 12;
  int skip = 1;
  int top_n = 5;
};

struct Evaluation {
  std::string fold_id;
  std::string validate_start;
  std::string validate_end;
  std::vector<double> monthly_returns;
  std::vector<double> primary_benchmark_returns;
  std::vector<double> secondary_benchmark_returns;
  std::string universe_variant;
  std::string execution_model;
  std::string fx_scenario;
  std::string cost_model_name;
  int periods_per_year = 12;
};

struct Candidate {
  CandidateParams params;
  std::vector<Evaluation> evaluations;
};

struct CandidateAggregate {
  std::string candidate_id;
  CandidateParams params;
  std::vector<Evaluation> evaluations;
  std::unordered_map<std::string, double> per_fold_sharpes;
  std::unordered_map<std::string, double> per_fold_drawdowns;
  int fold_pass_count = 0;
  double median_validation_sharpe = 0.0;
  double overall_sharpe = 0.0;
  double bootstrap_ci_low = 0.0;
  double bootstrap_ci_high = 0.0;
  double deflated_sharpe_score = 0.0;
  double deflated_sharpe_probability = 0.0;
  double max_drawdown = 0.0;
  double total_return = 0.0;
  std::vector<double> concatenated_returns;
  std::vector<double> primary_benchmark_returns;
  std::vector<double> secondary_benchmark_returns;
  std::optional<double> primary_benchmark_total_return;
  std::optional<double> secondary_benchmark_total_return;
  double universe_sensitivity_std = std::numeric_limits<double>::infinity();
  bool gate_fold_count = false;
  bool gate_deflated_sharpe = false;
  bool gate_bootstrap = false;
  bool gate_negative_controls = false;
  bool selected = false;
  int rank = 0;
  double plateau_neighbor_median_sharpe = 0.0;
  double plateau_neighbor_ratio = 0.0;
  std::vector<std::string> neighbor_ids;
};

constexpr const char* kPrimaryUniverse = "Full Nordics";
constexpr const char* kPrimaryExecution = "next_open";
constexpr const char* kPrimaryFx = "base";

const std::vector<std::string> kRequiredUniverseVariants = {
    "Full Nordics",
    "SE-only",
    "largest-third-by-market-cap",
};
const std::vector<std::string> kRequiredExecutionModels = {"next_open", "next_close"};
const std::vector<std::string> kRequiredFxScenarios = {"low", "base", "high"};

std::string iso_timestamp_now() {
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &tt);
#else
  gmtime_r(&tt, &tm);
#endif
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900,
                tm.tm_mon + 1,
                tm.tm_mday,
                tm.tm_hour,
                tm.tm_min,
                tm.tm_sec);
  return std::string(buffer);
}

std::string candidate_id(const CandidateParams& params) {
  return fmt::format("l{}_s{}_n{}", params.lookback, params.skip, params.top_n);
}

double mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double sum = 0.0;
  for (double v : values) sum += v;
  return sum / static_cast<double>(values.size());
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

double pstdev(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double mu = mean(values);
  double accum = 0.0;
  for (double v : values) {
    double diff = v - mu;
    accum += diff * diff;
  }
  return std::sqrt(accum / static_cast<double>(values.size()));
}

std::vector<CandidateParams> build_param_grid(const ProfileSettings& profile) {
  std::vector<CandidateParams> grid;
  for (int lookback : profile.lookbacks) {
    for (int skip : profile.skips) {
      for (int top_n : profile.top_ns) {
        grid.push_back({lookback, skip, top_n});
      }
    }
  }
  return grid;
}

std::vector<const Evaluation*> filter_evaluations(
    const std::vector<Evaluation>& evaluations,
    const std::string& universe_variant,
    const std::string& execution_model,
    const std::string& fx_scenario,
    const std::string& cost_model_name) {
  std::vector<const Evaluation*> rows;
  for (const auto& eval : evaluations) {
    if (!universe_variant.empty() && eval.universe_variant != universe_variant) continue;
    if (!execution_model.empty() && eval.execution_model != execution_model) continue;
    if (!fx_scenario.empty() && eval.fx_scenario != fx_scenario) continue;
    if (!cost_model_name.empty() && eval.cost_model_name != cost_model_name) continue;
    rows.push_back(&eval);
  }
  return rows;
}

std::vector<double> concatenate_returns(
    const std::vector<const Evaluation*>& evaluations,
    const std::vector<std::string>& ordered_folds) {
  if (evaluations.empty()) return {};
  std::unordered_map<std::string, const Evaluation*> lookup;
  for (const auto* eval : evaluations) {
    lookup[eval->fold_id] = eval;
  }
  std::vector<double> series;
  for (const auto& fold_id : ordered_folds) {
    auto it = lookup.find(fold_id);
    if (it == lookup.end()) continue;
    const auto& returns = it->second->monthly_returns;
    series.insert(series.end(), returns.begin(), returns.end());
  }
  return series;
}

std::vector<double> concatenate_series(
    const std::vector<const Evaluation*>& evaluations,
    const std::vector<std::string>& ordered_folds,
    bool primary) {
  if (evaluations.empty()) return {};
  std::unordered_map<std::string, const Evaluation*> lookup;
  for (const auto* eval : evaluations) {
    lookup[eval->fold_id] = eval;
  }
  std::vector<double> series;
  for (const auto& fold_id : ordered_folds) {
    auto it = lookup.find(fold_id);
    if (it == lookup.end()) continue;
    const auto& values = primary ? it->second->primary_benchmark_returns
                                 : it->second->secondary_benchmark_returns;
    if (values.empty()) {
      return {};
    }
    series.insert(series.end(), values.begin(), values.end());
  }
  return series;
}

std::vector<std::string> active_selection_variants(
    const ResearchDataset& dataset,
    const std::vector<std::string>& excluded_countries) {
  std::vector<std::string> active;
  for (const auto& variant : kRequiredUniverseVariants) {
    if (dataset.variant_has_any(variant, excluded_countries)) {
      active.push_back(variant);
    }
  }
  return active;
}

Candidate select_control_candidate(
    const std::vector<Candidate>& candidates,
    int periods_per_year,
    const Config& config) {
  if (candidates.empty()) {
    return Candidate{};
  }
  const auto& ordered_folds = config.rolling_folds;
  std::vector<std::string> fold_ids;
  fold_ids.reserve(ordered_folds.size());
  for (const auto& fold : ordered_folds) {
    fold_ids.push_back(fold.fold_id);
  }

  const Candidate* best = &candidates.front();
  double best_score = -1e9;
  for (const auto& candidate : candidates) {
    auto main_track = filter_evaluations(
        candidate.evaluations,
        kPrimaryUniverse,
        kPrimaryExecution,
        kPrimaryFx,
        config.primary_selection_cost_model);
    std::unordered_map<std::string, const Evaluation*> by_fold;
    for (const auto* eval : main_track) {
      by_fold[eval->fold_id] = eval;
    }
    std::vector<double> fold_scores;
    for (const auto& fold_id : fold_ids) {
      auto it = by_fold.find(fold_id);
      if (it == by_fold.end()) continue;
      fold_scores.push_back(annualized_sharpe(it->second->monthly_returns, periods_per_year));
    }
    if (fold_scores.empty()) continue;
    double score = median(fold_scores);
    if (score > best_score) {
      best_score = score;
      best = &candidate;
    }
  }
  return *best;
}
json compute_negative_controls(
    const ResearchDataset& dataset,
    const std::vector<Candidate>& candidates,
    int periods_per_year,
    const ProfileSettings& profile,
    const std::vector<std::string>& excluded_countries,
    const Config& config) {
  if (candidates.empty()) {
    return json{
        {"cross_sectional_shuffle", {{"pass_count", 0}, {"run_count", 0}}},
        {"block_shuffled_null", {{"pass_count", 0}, {"run_count", 0}}},
    };
  }

  auto control_candidate = select_control_candidate(candidates, periods_per_year, config);
  SimulationParams params{control_candidate.params.lookback, control_candidate.params.skip, control_candidate.params.top_n};

  std::vector<std::vector<std::pair<double, double>>> months;
  auto negative_months = dataset.negative_control_months(params, config.rolling_folds.front().validate_start,
                                                        config.rolling_folds.back().validate_end, excluded_countries);
  for (const auto& month : negative_months) {
    std::vector<std::pair<double, double>> positions;
    positions.reserve(month.size());
    for (const auto& pos : month) {
      positions.push_back({pos.score, pos.next_return});
    }
    months.push_back(std::move(positions));
  }

  auto shuffle_runs = cross_sectional_score_shuffle_runs(months, params.top_n, profile.cross_sectional_shuffle_runs, 11);
  int cross_pass_count = 0;
  for (const auto& run : shuffle_runs) {
    double sharpe = annualized_sharpe(run, periods_per_year);
    auto ci = stationary_bootstrap_sharpe_ci(run, config.bootstrap_block_length_months,
                                             config.negative_control_bootstrap_resamples, 7);
    auto dsr = deflated_sharpe_metrics(run, static_cast<int>(candidates.size()));
    if (sharpe > 0.4 && ci.first > 0.0 && dsr.score > 0.0) {
      cross_pass_count += 1;
    }
  }

  std::vector<double> main_returns;
  {
    auto main_track = filter_evaluations(
        control_candidate.evaluations,
        kPrimaryUniverse,
        kPrimaryExecution,
        kPrimaryFx,
        config.primary_selection_cost_model);
    std::vector<std::string> fold_ids;
    for (const auto& fold : config.rolling_folds) {
      fold_ids.push_back(fold.fold_id);
    }
    main_returns = concatenate_returns(main_track, fold_ids);
  }

  int block_length_periods = std::max(
      1,
      static_cast<int>(std::round(static_cast<double>(config.bootstrap_block_length_months) * periods_per_year / 12.0)));
  auto block_runs = block_shuffled_return_path_runs(
      main_returns, block_length_periods, profile.block_shuffled_null_runs, 17);
  int block_pass_count = 0;
  for (const auto& run : block_runs) {
    double sharpe = annualized_sharpe(run, periods_per_year);
    auto ci = stationary_bootstrap_sharpe_ci(run, config.bootstrap_block_length_months,
                                             config.negative_control_bootstrap_resamples, 23);
    auto dsr = deflated_sharpe_metrics(run, static_cast<int>(candidates.size()));
    if (sharpe > 0.4 && ci.first > 0.0 && dsr.score > 0.0) {
      block_pass_count += 1;
    }
  }

  return json{
      {"cross_sectional_shuffle", {{"pass_count", cross_pass_count}, {"run_count", profile.cross_sectional_shuffle_runs}}},
      {"block_shuffled_null", {{"pass_count", block_pass_count}, {"run_count", profile.block_shuffled_null_runs}}},
  };
}

bool negative_controls_pass(const json& negative_controls, double threshold_max) {
  if (!negative_controls.contains("cross_sectional_shuffle") || !negative_controls.contains("block_shuffled_null")) {
    return false;
  }
  for (const auto& key : {"cross_sectional_shuffle", "block_shuffled_null"}) {
    const auto& payload = negative_controls.at(key);
    double run_count = payload.value("run_count", 0);
    double pass_count = payload.value("pass_count", 0);
    if (run_count <= 0) return false;
    if ((pass_count / run_count) > threshold_max) {
      return false;
    }
  }
  return true;
}

std::vector<Candidate> build_candidate_evaluations(
    const ResearchDataset& dataset,
    const ThesisSettings& thesis,
    const std::vector<CandidateParams>& params_grid,
    int periods_per_year,
    const Config& config) {
  std::vector<Candidate> candidates;
  for (const auto& params : params_grid) {
    Candidate candidate;
    candidate.params = params;
    for (const auto& fold : config.rolling_folds) {
      for (const auto& variant : kRequiredUniverseVariants) {
        for (const auto& exec : kRequiredExecutionModels) {
          for (const auto& fx : kRequiredFxScenarios) {
            WindowSimulation simulation = dataset.simulate_window(
                {params.lookback, params.skip, params.top_n},
                variant,
                exec,
                fx,
                fold.validate_start,
                fold.validate_end,
                thesis.excluded_countries);
            Evaluation eval;
            eval.fold_id = fold.fold_id;
            eval.validate_start = fold.validate_start;
            eval.validate_end = fold.validate_end;
            eval.monthly_returns = std::move(simulation.monthly_returns);
            eval.primary_benchmark_returns = std::move(simulation.primary_benchmark_returns);
            eval.secondary_benchmark_returns = std::move(simulation.secondary_benchmark_returns);
            eval.universe_variant = variant;
            eval.execution_model = exec;
            eval.fx_scenario = fx;
            eval.cost_model_name = config.primary_selection_cost_model;
            eval.periods_per_year = periods_per_year;
            candidate.evaluations.push_back(std::move(eval));
          }
        }
      }
    }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

std::vector<CandidateAggregate> aggregate_candidates(
    const std::vector<Candidate>& candidates,
    const json& negative_controls,
    int periods_per_year,
    const Config& config) {
  std::vector<CandidateAggregate> aggregates;
  if (candidates.empty()) return aggregates;

  std::vector<std::string> ordered_folds;
  ordered_folds.reserve(config.rolling_folds.size());
  for (const auto& fold : config.rolling_folds) {
    ordered_folds.push_back(fold.fold_id);
  }

  bool neg_controls_gate = negative_controls_pass(negative_controls, config.negative_control_pass_rate_max);

  for (const auto& candidate : candidates) {
    CandidateAggregate agg;
    agg.params = candidate.params;
    agg.candidate_id = candidate_id(candidate.params);
    agg.evaluations = candidate.evaluations;

    auto main_track = filter_evaluations(
        candidate.evaluations,
        kPrimaryUniverse,
        kPrimaryExecution,
        kPrimaryFx,
        config.primary_selection_cost_model);

    std::unordered_map<std::string, const Evaluation*> fold_map;
    for (const auto* eval : main_track) {
      fold_map[eval->fold_id] = eval;
    }
    for (const auto& fold_id : ordered_folds) {
      if (!fold_map.count(fold_id)) {
        throw std::runtime_error("Candidate missing required folds: " + agg.candidate_id);
      }
    }

    std::vector<double> fold_sharpes;
    for (const auto& fold_id : ordered_folds) {
      const auto* eval = fold_map[fold_id];
      double sharpe = annualized_sharpe(eval->monthly_returns, periods_per_year);
      double drawdown = max_drawdown(eval->monthly_returns);
      agg.per_fold_sharpes[fold_id] = sharpe;
      agg.per_fold_drawdowns[fold_id] = drawdown;
      fold_sharpes.push_back(sharpe);
    }

    agg.concatenated_returns = concatenate_returns(main_track, ordered_folds);
    agg.primary_benchmark_returns = concatenate_series(main_track, ordered_folds, true);
    agg.secondary_benchmark_returns = concatenate_series(main_track, ordered_folds, false);
    agg.median_validation_sharpe = median(fold_sharpes);
    agg.overall_sharpe = annualized_sharpe(agg.concatenated_returns, periods_per_year);
    for (double value : fold_sharpes) {
      if (value > 0.4) agg.fold_pass_count += 1;
    }
    auto ci = stationary_bootstrap_sharpe_ci(
        agg.concatenated_returns,
        config.bootstrap_block_length_months,
        config.bootstrap_resamples,
        7);
    agg.bootstrap_ci_low = ci.first;
    agg.bootstrap_ci_high = ci.second;
    auto dsr = deflated_sharpe_metrics(agg.concatenated_returns, static_cast<int>(candidates.size()));
    agg.deflated_sharpe_score = dsr.score;
    agg.deflated_sharpe_probability = dsr.probability;
    agg.max_drawdown = max_drawdown(agg.concatenated_returns);
    agg.total_return = total_return(agg.concatenated_returns);
    if (!agg.primary_benchmark_returns.empty()) {
      agg.primary_benchmark_total_return = total_return(agg.primary_benchmark_returns);
    }
    if (!agg.secondary_benchmark_returns.empty()) {
      agg.secondary_benchmark_total_return = total_return(agg.secondary_benchmark_returns);
    }

    std::vector<double> universe_variant_sharpes;
    for (const auto& variant : kRequiredUniverseVariants) {
      auto rows = filter_evaluations(
          candidate.evaluations,
          variant,
          kPrimaryExecution,
          kPrimaryFx,
          config.primary_selection_cost_model);
      if (rows.size() != ordered_folds.size()) {
        universe_variant_sharpes.clear();
        break;
      }
      auto variant_returns = concatenate_returns(rows, ordered_folds);
      universe_variant_sharpes.push_back(annualized_sharpe(variant_returns, periods_per_year));
    }
    if (!universe_variant_sharpes.empty()) {
      agg.universe_sensitivity_std = pstdev(universe_variant_sharpes);
    }

    agg.gate_fold_count = agg.fold_pass_count >= config.mega_wf_passes_required;
    agg.gate_deflated_sharpe = agg.deflated_sharpe_score > 0.0;
    agg.gate_bootstrap = agg.bootstrap_ci_low > 0.0;
    agg.gate_negative_controls = neg_controls_gate;

    aggregates.push_back(std::move(agg));
  }
  return aggregates;
}

void attach_plateau_diagnostics(std::vector<CandidateAggregate>& aggregates) {
  if (aggregates.empty()) return;
  std::map<std::tuple<int, int, int>, size_t> lookup;
  std::set<int> l_values;
  std::set<int> skip_values;
  std::set<int> top_values;
  for (size_t idx = 0; idx < aggregates.size(); ++idx) {
    const auto& params = aggregates[idx].params;
    lookup[{params.lookback, params.skip, params.top_n}] = idx;
    l_values.insert(params.lookback);
    skip_values.insert(params.skip);
    top_values.insert(params.top_n);
  }
  std::vector<int> l_list(l_values.begin(), l_values.end());
  std::vector<int> skip_list(skip_values.begin(), skip_values.end());
  std::vector<int> top_list(top_values.begin(), top_values.end());

  auto neighbor_indices = [&](const CandidateParams& params) {
    std::vector<size_t> neighbors;
    auto neighbor_for = [&](const std::vector<int>& grid, int value, auto setter) {
      auto it = std::find(grid.begin(), grid.end(), value);
      if (it == grid.end()) return;
      int index = static_cast<int>(std::distance(grid.begin(), it));
      for (int offset : {-1, 1}) {
        int neighbor_index = index + offset;
        if (neighbor_index < 0 || neighbor_index >= static_cast<int>(grid.size())) continue;
        CandidateParams neighbor = params;
        setter(neighbor, grid[neighbor_index]);
        auto key = std::make_tuple(neighbor.lookback, neighbor.skip, neighbor.top_n);
        auto found = lookup.find(key);
        if (found != lookup.end()) {
          neighbors.push_back(found->second);
        }
      }
    };
    neighbor_for(l_list, params.lookback, [](CandidateParams& p, int value) { p.lookback = value; });
    neighbor_for(skip_list, params.skip, [](CandidateParams& p, int value) { p.skip = value; });
    neighbor_for(top_list, params.top_n, [](CandidateParams& p, int value) { p.top_n = value; });
    return neighbors;
  };

  for (auto& aggregate : aggregates) {
    auto neighbors = neighbor_indices(aggregate.params);
    std::vector<double> neighbor_sharpes;
    aggregate.neighbor_ids.clear();
    for (auto idx : neighbors) {
      neighbor_sharpes.push_back(aggregates[idx].median_validation_sharpe);
      aggregate.neighbor_ids.push_back(aggregates[idx].candidate_id);
    }
    if (!neighbor_sharpes.empty()) {
      aggregate.plateau_neighbor_median_sharpe = median(neighbor_sharpes);
      if (aggregate.median_validation_sharpe != 0.0) {
        aggregate.plateau_neighbor_ratio =
            aggregate.plateau_neighbor_median_sharpe / aggregate.median_validation_sharpe;
      } else {
        aggregate.plateau_neighbor_ratio = 0.0;
      }
    } else {
      aggregate.plateau_neighbor_median_sharpe = -INFINITY;
      aggregate.plateau_neighbor_ratio = 0.0;
    }
  }
}

bool hard_gate_passed(const CandidateAggregate& agg) {
  return agg.gate_fold_count && agg.gate_bootstrap && agg.gate_deflated_sharpe && agg.gate_negative_controls;
}

void attach_ranks(std::vector<CandidateAggregate>& aggregates) {
  std::vector<size_t> order(aggregates.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const auto& lhs = aggregates[a];
    const auto& rhs = aggregates[b];
    auto lhs_gate = hard_gate_passed(lhs) ? 1 : 0;
    auto rhs_gate = hard_gate_passed(rhs) ? 1 : 0;
    if (lhs_gate != rhs_gate) return lhs_gate > rhs_gate;
    if (lhs.median_validation_sharpe != rhs.median_validation_sharpe) {
      return lhs.median_validation_sharpe > rhs.median_validation_sharpe;
    }
    if (lhs.universe_sensitivity_std != rhs.universe_sensitivity_std) {
      return lhs.universe_sensitivity_std < rhs.universe_sensitivity_std;
    }
    if (lhs.plateau_neighbor_median_sharpe != rhs.plateau_neighbor_median_sharpe) {
      return lhs.plateau_neighbor_median_sharpe > rhs.plateau_neighbor_median_sharpe;
    }
    if (lhs.plateau_neighbor_ratio != rhs.plateau_neighbor_ratio) {
      return lhs.plateau_neighbor_ratio > rhs.plateau_neighbor_ratio;
    }
    if (lhs.max_drawdown != rhs.max_drawdown) {
      return lhs.max_drawdown < rhs.max_drawdown;
    }
    if (lhs.params.lookback != rhs.params.lookback) {
      return lhs.params.lookback > rhs.params.lookback;
    }
    if (lhs.params.skip != rhs.params.skip) {
      return lhs.params.skip > rhs.params.skip;
    }
    return lhs.params.top_n > rhs.params.top_n;
  });
  for (size_t rank = 0; rank < order.size(); ++rank) {
    aggregates[order[rank]].rank = static_cast<int>(rank + 1);
  }
  if (!order.empty()) {
    auto& selected = aggregates[order[0]];
    selected.selected = hard_gate_passed(selected);
  }
}
json compute_cscv_pbo(
    const std::vector<Candidate>& candidates,
    int periods_per_year,
    const std::string& period_label,
    const Config& config) {
  json base = {
      {"status", "unavailable"},
      {"method", "cscv_pbo_main_track_sharpe"},
      {"score_function", "annualized_sharpe"},
      {"pbo_threshold_max", config.pbo_threshold_max},
      {"passes_pbo_threshold", false},
      {"passes_threshold", false},
      {"period_label", period_label},
      {"periods_per_year", periods_per_year},
  };
  if (candidates.empty()) {
    return base;
  }

  std::vector<std::string> ordered_folds;
  for (const auto& fold : config.rolling_folds) {
    ordered_folds.push_back(fold.fold_id);
  }

  std::vector<std::vector<double>> candidate_returns;
  for (const auto& candidate : candidates) {
    auto main_track = filter_evaluations(
        candidate.evaluations,
        kPrimaryUniverse,
        kPrimaryExecution,
        kPrimaryFx,
        config.primary_selection_cost_model);
    if (main_track.size() != ordered_folds.size()) {
      return base;
    }
    candidate_returns.push_back(concatenate_returns(main_track, ordered_folds));
  }
  if (candidate_returns.empty()) return base;
  size_t total_periods = candidate_returns.front().size();
  if (total_periods == 0) return base;
  for (const auto& series : candidate_returns) {
    if (series.size() != total_periods) {
      return base;
    }
  }

  int target_slices = config.pbo_target_slice_count;
  int min_slice_months = config.pbo_min_slice_length_months;
  std::optional<int> slice_count;
  if (total_periods % target_slices == 0) {
    int candidate_slice_length = static_cast<int>(total_periods / target_slices);
    double slice_months = static_cast<double>(candidate_slice_length) / periods_per_year * 12.0;
    if (slice_months >= min_slice_months) {
      slice_count = target_slices;
    }
  }
  if (!slice_count) {
    for (int value = target_slices; value >= 2; value -= 2) {
      if (total_periods % value != 0) continue;
      int slice_length = static_cast<int>(total_periods / value);
      double slice_months = static_cast<double>(slice_length) / periods_per_year * 12.0;
      if (slice_months >= min_slice_months) {
        slice_count = value;
        break;
      }
    }
  }
  if (!slice_count) {
    return base;
  }

  int slice_length_periods = static_cast<int>(total_periods / *slice_count);
  double total_months = static_cast<double>(total_periods) / periods_per_year * 12.0;
  double slice_length_months = static_cast<double>(slice_length_periods) / periods_per_year * 12.0;

  std::vector<std::vector<std::vector<double>>> slices;
  slices.reserve(candidate_returns.size());
  for (const auto& series : candidate_returns) {
    std::vector<std::vector<double>> candidate_slices;
    for (int idx = 0; idx < *slice_count; ++idx) {
      auto start = series.begin() + idx * slice_length_periods;
      auto end = start + slice_length_periods;
      candidate_slices.emplace_back(start, end);
    }
    slices.push_back(std::move(candidate_slices));
  }

  int combination_count = 0;
  std::vector<double> logit_values;
  std::vector<double> percentile_values;
  int half = *slice_count / 2;
  double epsilon = 1e-6;

  std::vector<int> combo_mask(*slice_count, 0);
  for (int i = 0; i < half; ++i) combo_mask[i] = 1;
  std::sort(combo_mask.begin(), combo_mask.end(), std::greater<int>());
  do {
    std::set<int> in_indices;
    for (int i = 0; i < *slice_count; ++i) {
      if (combo_mask[i] == 1) in_indices.insert(i);
    }
    std::vector<int> out_indices;
    for (int i = 0; i < *slice_count; ++i) {
      if (!in_indices.count(i)) out_indices.push_back(i);
    }
    combination_count += 1;

    std::vector<double> in_scores;
    std::vector<double> out_scores;
    for (const auto& candidate_slices : slices) {
      std::vector<double> in_returns;
      std::vector<double> out_returns;
      for (int idx : in_indices) {
        const auto& slice = candidate_slices[idx];
        in_returns.insert(in_returns.end(), slice.begin(), slice.end());
      }
      for (int idx : out_indices) {
        const auto& slice = candidate_slices[idx];
        out_returns.insert(out_returns.end(), slice.begin(), slice.end());
      }
      in_scores.push_back(annualized_sharpe(in_returns, periods_per_year));
      out_scores.push_back(annualized_sharpe(out_returns, periods_per_year));
    }

    std::vector<std::pair<int, double>> ranked;
    for (size_t i = 0; i < in_scores.size(); ++i) {
      ranked.push_back({static_cast<int>(i), in_scores[i]});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });
    int best_index = ranked.front().first;

    std::vector<std::pair<int, double>> oos_ranked;
    for (size_t i = 0; i < out_scores.size(); ++i) {
      oos_ranked.push_back({static_cast<int>(i), out_scores[i]});
    }
    std::sort(oos_ranked.begin(), oos_ranked.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });
    int oos_rank = 1;
    for (size_t i = 0; i < oos_ranked.size(); ++i) {
      if (oos_ranked[i].first == best_index) {
        oos_rank = static_cast<int>(i + 1);
        break;
      }
    }
    double percentile = 0.5;
    if (out_scores.size() > 1) {
      percentile = 1.0 - static_cast<double>(oos_rank - 1) / static_cast<double>(out_scores.size() - 1);
    }
    percentile_values.push_back(percentile);
    double safe_percentile = std::min(1.0 - epsilon, std::max(epsilon, percentile));
    logit_values.push_back(std::log(safe_percentile / (1.0 - safe_percentile)));
  } while (std::prev_permutation(combo_mask.begin(), combo_mask.end()));

  if (combination_count == 0) return base;

  int pbo_count = 0;
  for (double value : logit_values) {
    if (value <= 0.0) pbo_count += 1;
  }
  double pbo = static_cast<double>(pbo_count) / static_cast<double>(combination_count);
  double mean_percentile = mean(percentile_values);
  double median_percentile = median(percentile_values);
  double mean_logit = mean(logit_values);
  double median_logit = median(logit_values);
  bool passes = pbo <= config.pbo_threshold_max;
  std::string interpretation = passes ? "healthy" : "overfitting_warning";

  json result = {
      {"status", "ok"},
      {"method", "cscv_pbo_main_track_sharpe"},
      {"score_function", "annualized_sharpe"},
      {"pbo", pbo},
      {"pbo_threshold_max", config.pbo_threshold_max},
      {"passes_pbo_threshold", passes},
      {"passes_threshold", passes},
      {"slice_count", *slice_count},
      {"slice_length_periods", slice_length_periods},
      {"slice_length_months", slice_length_months},
      {"combination_count", combination_count},
      {"candidate_count", static_cast<int>(candidate_returns.size())},
      {"total_periods", static_cast<int>(total_periods)},
      {"total_months", total_months},
      {"lambda_logit_mean", mean_logit},
      {"lambda_logit_median", median_logit},
      {"mean_oos_rank_percentile", mean_percentile},
      {"median_oos_rank_percentile", median_percentile},
      {"interpretation", interpretation},
      {"period_label", period_label},
      {"periods_per_year", periods_per_year},
      {"threshold_max", config.pbo_threshold_max},
  };
  return result;
}

json selection_summary(
    const std::vector<CandidateAggregate>& aggregates,
    const json& negative_controls,
    const std::string& mode,
    int periods_per_year,
    const std::vector<std::string>& active_variants) {
  json summary;
  summary["mode"] = mode;
  summary["selection_status"] = "no_candidate_passed_hard_gates";
  summary["locked_candidate"] = nullptr;
  summary["ranked_candidates"] = json::array();
  summary["negative_controls"] = negative_controls;
  summary["neighbor_diagnostics"] = json::array();
  summary["period_label"] = "months";
  summary["periods_per_year"] = periods_per_year;

  std::vector<CandidateAggregate> ranked = aggregates;
  std::sort(ranked.begin(), ranked.end(),
            [](const CandidateAggregate& a, const CandidateAggregate& b) { return a.rank < b.rank; });
  const CandidateAggregate* selected = nullptr;
  for (const auto& agg : ranked) {
    if (agg.selected) {
      selected = &agg;
      summary["selection_status"] = "selected";
      break;
    }
  }

  for (const auto& agg : ranked) {
    json candidate;
    candidate["candidate_id"] = agg.candidate_id;
    candidate["params"] = {{"l", agg.params.lookback}, {"skip", agg.params.skip}, {"top_n", agg.params.top_n}};
    candidate["rank"] = agg.rank;
    candidate["fold_pass_count"] = agg.fold_pass_count;
    candidate["median_validation_sharpe"] = agg.median_validation_sharpe;
    candidate["overall_sharpe"] = agg.overall_sharpe;
    candidate["deflated_sharpe_score"] = agg.deflated_sharpe_score;
    candidate["deflated_sharpe_probability"] = agg.deflated_sharpe_probability;
    candidate["bootstrap_ci_low"] = agg.bootstrap_ci_low;
    candidate["bootstrap_ci_high"] = agg.bootstrap_ci_high;
    candidate["universe_sensitivity_std"] = agg.universe_sensitivity_std;
    candidate["plateau_neighbor_median_sharpe"] = agg.plateau_neighbor_median_sharpe;
    candidate["plateau_neighbor_ratio"] = agg.plateau_neighbor_ratio;
    candidate["max_drawdown"] = agg.max_drawdown;
    candidate["total_return"] = agg.total_return;
    candidate["concatenated_returns"] = agg.concatenated_returns;
    candidate["primary_benchmark_returns"] = agg.primary_benchmark_returns;
    candidate["secondary_benchmark_returns"] = agg.secondary_benchmark_returns;
    candidate["primary_benchmark_total_return"] =
        agg.primary_benchmark_total_return ? json(*agg.primary_benchmark_total_return) : json(nullptr);
    candidate["secondary_benchmark_total_return"] =
        agg.secondary_benchmark_total_return ? json(*agg.secondary_benchmark_total_return) : json(nullptr);
    candidate["per_fold_sharpes"] = agg.per_fold_sharpes;
    candidate["per_fold_drawdowns"] = agg.per_fold_drawdowns;
    candidate["gate_fold_count"] = agg.gate_fold_count;
    candidate["gate_deflated_sharpe"] = agg.gate_deflated_sharpe;
    candidate["gate_bootstrap"] = agg.gate_bootstrap;
    candidate["gate_negative_controls"] = agg.gate_negative_controls;
    candidate["selected"] = agg.selected;
    candidate["period_label"] = "months";
    candidate["periods_per_year"] = periods_per_year;
    candidate["active_selection_universe_variants"] = active_variants;
    summary["ranked_candidates"].push_back(candidate);
  }

  if (selected && !summary["ranked_candidates"].empty()) {
    summary["locked_candidate"] = summary["ranked_candidates"][0];
    for (const auto& neighbor_id : selected->neighbor_ids) {
      auto it = std::find_if(ranked.begin(), ranked.end(),
                             [&](const CandidateAggregate& agg) { return agg.candidate_id == neighbor_id; });
      if (it == ranked.end()) continue;
      json neighbor;
      neighbor["candidate_id"] = it->candidate_id;
      neighbor["params"] = {{"l", it->params.lookback}, {"skip", it->params.skip}, {"top_n", it->params.top_n}};
      json reasons = json::array();
      if (!it->gate_fold_count) {
        reasons.push_back("fewer than 4 of 5 folds cleared Sharpe > 0.4");
      }
      if (!it->gate_deflated_sharpe) {
        reasons.push_back("deflated Sharpe score <= 0");
      }
      if (!it->gate_bootstrap) {
        reasons.push_back("bootstrap CI lower bound <= 0");
      }
      if (!it->gate_negative_controls) {
        reasons.push_back("negative-control pass rate exceeded 5%");
      }
      if (reasons.empty()) {
        reasons.push_back("lower rank than the locked candidate on median Sharpe, robustness, or drawdown");
      }
      neighbor["reasons"] = reasons;
      summary["neighbor_diagnostics"].push_back(neighbor);
    }
  }
  return summary;
}

json evaluate_holdout(
    const ResearchDataset& dataset,
    const ThesisSettings& thesis,
    const CandidateParams& params,
    int periods_per_year,
    const Config& config) {
  json results = json::object();
  for (const auto& variant : kRequiredUniverseVariants) {
    for (const auto& exec : kRequiredExecutionModels) {
      for (const auto& fx : kRequiredFxScenarios) {
        WindowSimulation simulation = dataset.simulate_window(
            {params.lookback, params.skip, params.top_n},
            variant,
            exec,
            fx,
            config.oos_start,
            config.oos_end,
            thesis.excluded_countries);
        const auto& returns = simulation.monthly_returns;
        json row;
        row["net_sharpe"] = annualized_sharpe(returns, periods_per_year);
        row["max_drawdown"] = max_drawdown(returns);
        row["total_return"] = total_return(returns);
        row["months"] = static_cast<int>(returns.size());
        row["period_count"] = static_cast<int>(returns.size());
        row["period_label"] = "months";
        row["periods_per_year"] = periods_per_year;
        if (!simulation.primary_benchmark_returns.empty()) {
          row["primary_benchmark_total_return"] = total_return(simulation.primary_benchmark_returns);
          row["beats_primary_benchmark"] =
              row["total_return"].get<double>() > row["primary_benchmark_total_return"].get<double>();
        }
        if (!simulation.secondary_benchmark_returns.empty()) {
          row["secondary_benchmark_total_return"] = total_return(simulation.secondary_benchmark_returns);
        }
        results[variant][exec][fx] = row;
      }
    }
  }

  auto base_main = results[kPrimaryUniverse][kPrimaryExecution][kPrimaryFx];
  bool meets_sharpe = base_main["net_sharpe"].get<double>() >= config.oos_sharpe_min;
  bool beats_benchmark = base_main.value("beats_primary_benchmark", false);
  bool phase4_eligible = meets_sharpe && beats_benchmark;
  json holdout;
  holdout["selected_params"] = {{"l", params.lookback}, {"skip", params.skip}, {"top_n", params.top_n}};
  holdout["holdout_window"] = {{"start", config.oos_start}, {"end", config.oos_end}};
  holdout["results"] = results;
  holdout["phase4_gate"] = {
      {"base_main_net_sharpe", base_main["net_sharpe"]},
      {"meets_sharpe_gate", meets_sharpe},
      {"beats_primary_benchmark", beats_benchmark},
      {"phase4_eligible", phase4_eligible},
  };
  holdout["period_label"] = "months";
  holdout["periods_per_year"] = periods_per_year;
  return holdout;
}

struct ParsedArgs {
  std::string data_dir = "data";
  std::string output_dir = "results/research_engine";
  std::vector<std::string> theses;
  std::vector<std::string> profiles;
  std::string profile_set = "default";
  bool skip_holdout = false;
};

std::vector<std::string> parse_list_flag(const CliArgs& args, const std::string& flag) {
  std::vector<std::string> values;
  for (size_t i = 0; i < args.args.size(); ++i) {
    if (args.args[i] != flag) continue;
    for (size_t j = i + 1; j < args.args.size(); ++j) {
      if (!args.args[j].empty() && args.args[j].rfind("--", 0) == 0) {
        break;
      }
      values.push_back(args.args[j]);
    }
  }
  return values;
}

ParsedArgs parse_args(const CliArgs& args, const Config& config) {
  ParsedArgs parsed;
  parsed.data_dir = args.value_or("--data-dir", parsed.data_dir);
  parsed.output_dir = args.value_or("--output-dir", parsed.output_dir);
  parsed.profile_set = args.value_or("--profile-set", parsed.profile_set);
  parsed.skip_holdout = args.has_flag("--skip-holdout");

  parsed.theses = parse_list_flag(args, "--theses");
  if (parsed.theses.empty()) {
    for (const auto& entry : config.theses) {
      parsed.theses.push_back(entry.first);
    }
    std::sort(parsed.theses.begin(), parsed.theses.end());
  }
  parsed.profiles = parse_list_flag(args, "--profiles");
  if (parsed.profiles.empty()) {
    parsed.profiles = {"quick", "mega", "certification_baseline"};
  }
  return parsed;
}

}  // namespace
}  // namespace am

int main(int argc, char** argv) {
  using namespace am;
  CliArgs cli(argc, argv);
  Config config = load_config("config_cpp.json");
  ParsedArgs args = parse_args(cli, config);

  if (!config.profile_sets.count(args.profile_set)) {
    throw std::runtime_error("Unknown profile-set: " + args.profile_set);
  }
  const auto& profile_set = config.profile_sets.at(args.profile_set);

  ResearchDataset dataset(args.data_dir, config);

  std::filesystem::path output_dir(args.output_dir);
  std::filesystem::create_directories(output_dir);
  std::filesystem::path summary_dir = output_dir / "summary";
  std::filesystem::create_directories(summary_dir);

  json summary_payload;
  summary_payload["generated_at"] = iso_timestamp_now();
  summary_payload["results_root"] = output_dir.string();
  summary_payload["research_cycle"] = "research_engine";
  summary_payload["profiles"] = args.profiles;
  summary_payload["profile_set"] = args.profile_set;
  summary_payload["theses"] = json::array();

  int periods_per_year = 12;

  for (const auto& thesis_name : args.theses) {
    auto thesis_it = config.theses.find(thesis_name);
    if (thesis_it == config.theses.end()) {
      spdlog::warn("Skipping unknown thesis '{}'", thesis_name);
      continue;
    }
    const auto& thesis = thesis_it->second;
    json thesis_meta = {
        {"name", thesis_name},
        {"label", thesis.label},
        {"excluded_countries", thesis.excluded_countries},
        {"scope_note", thesis.scope_note},
    };

    std::filesystem::path thesis_dir = output_dir / thesis_name;
    std::filesystem::create_directories(thesis_dir);
    json summaries = json::object();

    for (const auto& profile_name : args.profiles) {
      auto profile_it = profile_set.find(profile_name);
      if (profile_it == profile_set.end()) {
        spdlog::warn("Skipping profile '{}' (not in profile-set {})", profile_name, args.profile_set);
        continue;
      }
      const auto& profile = profile_it->second;
      auto grid = build_param_grid(profile);
      auto candidates = build_candidate_evaluations(dataset, thesis, grid, periods_per_year, config);
      auto negative_controls = compute_negative_controls(
          dataset, candidates, periods_per_year, profile, thesis.excluded_countries, config);
      auto aggregates = aggregate_candidates(candidates, negative_controls, periods_per_year, config);
      attach_plateau_diagnostics(aggregates);
      attach_ranks(aggregates);
      auto active_variants = active_selection_variants(dataset, thesis.excluded_countries);
      auto summary = selection_summary(aggregates, negative_controls, profile_name, periods_per_year, active_variants);
      summary["backtest_overfitting"] = compute_cscv_pbo(candidates, periods_per_year, "months", config);
      summary["thesis"] = thesis_meta;
      summary["profile_name"] = profile_name;
      summaries[profile_name] = summary;

      std::string filename;
      if (profile_name == "quick") {
        filename = "quick_summary.json";
      } else if (profile_name == "mega") {
        filename = "mega_summary.json";
      } else if (profile_name == "certification_baseline") {
        filename = "selection_summary.json";
      } else {
        filename = profile_name + std::string("_summary.json");
      }
      write_json(thesis_dir / filename, summary);
    }

    json certification = summaries.contains("certification_baseline")
                             ? summaries["certification_baseline"]
                             : json::object();
    json holdout;
    if (!args.skip_holdout && certification.contains("locked_candidate") &&
        !certification["locked_candidate"].is_null()) {
      auto locked = certification["locked_candidate"];
      CandidateParams params{locked["params"]["l"], locked["params"]["skip"], locked["params"]["top_n"]};
      holdout = evaluate_holdout(dataset, thesis, params, periods_per_year, config);
      holdout["status"] = "ok";
    } else {
      holdout["status"] = args.skip_holdout ? "skipped" : "blocked_by_missing_selection";
      holdout["phase4_gate"] = json::object();
    }
    write_json(thesis_dir / "holdout_results.json", holdout);

    summary_payload["theses"].push_back({
        {"thesis", thesis_meta},
        {"quick", summaries.value("quick", json::object())},
        {"mega", summaries.value("mega", json::object())},
        {"certification", certification},
        {"holdout", holdout},
        {"output_dir", thesis_dir.string()},
    });
  }

  write_json(summary_dir / "research_engine_summary.json", summary_payload);
  fmt::print("Research engine ready. Summary at {}\n", (summary_dir / "research_engine_summary.json").string());
  return 0;
}
