#include "am/core.h"
#include "am/config.h"
#include "am/dataset.h"
#include "am/validation.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

struct CadenceSpec {
  std::string cadence_id;
  std::string cadence_label;
  std::string schedule_type;
  int periods_per_year = 12;
  int canonical_offset_id = 0;
  std::vector<int> offset_ids;
  std::string offset_label_prefix;
};

struct PositionSample {
  double score = 0.0;
  double next_return = 0.0;
};

constexpr const char* kPrimaryUniverse = "Full Nordics";
constexpr const char* kPrimaryExecution = "next_open";
constexpr const char* kPrimaryFx = "base";
constexpr const char* kRebalanceLogic = "legacy_entries_exits";
constexpr const char* kRebalanceLogicLabel = "Legacy Entry/Exit Only";

const std::vector<std::string> kRequiredUniverseVariants = {
    "Full Nordics",
    "SE-only",
    "largest-third-by-market-cap",
};
const std::vector<std::string> kRequiredExecutionModels = {"next_open", "next_close"};
const std::vector<std::string> kRequiredFxScenarios = {"low", "base", "high"};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct IsoWeekKey {
  int year = 0;
  int week = 0;
  bool operator==(const IsoWeekKey& other) const { return year == other.year && week == other.week; }
};

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

int month_ordinal(const std::string& month) {
  if (month.size() < 7) return 0;
  int year = std::stoi(month.substr(0, 4));
  int mon = std::stoi(month.substr(5, 2));
  return year * 12 + (mon - 1);
}

std::string month_label_from_ts(int64_t ts) {
  auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
      std::chrono::nanoseconds(ts));
  auto days = std::chrono::floor<std::chrono::days>(tp);
  std::chrono::year_month_day ymd(days);
  int year = static_cast<int>(ymd.year());
  unsigned month = static_cast<unsigned>(ymd.month());
  char buffer[8];
  std::snprintf(buffer, sizeof(buffer), "%04d-%02u", year, month);
  return std::string(buffer);
}

IsoWeekKey iso_week_key(int64_t ts) {
  auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
      std::chrono::nanoseconds(ts));
  auto days = std::chrono::floor<std::chrono::days>(tp);
  std::chrono::weekday wd{days};
  int iso_wd = wd.iso_encoding();
  auto thursday = days + std::chrono::days{4 - iso_wd};
  std::chrono::year_month_day thursday_ymd(thursday);
  int iso_year = static_cast<int>(thursday_ymd.year());
  auto jan4 = std::chrono::sys_days{std::chrono::year{iso_year} / std::chrono::January / 4};
  int jan4_wd = std::chrono::weekday{jan4}.iso_encoding();
  auto week1_monday = jan4 - std::chrono::days{jan4_wd - 1};
  int week = 1 + static_cast<int>((thursday - week1_monday).count() / 7);
  return {iso_year, week};
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

std::string candidate_id(const CandidateParams& params) {
  return fmt::format("l{}_s{}_n{}", params.lookback, params.skip, params.top_n);
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

CadenceSpec load_cadence_spec(const Config& config, const std::string& cadence_id) {
  auto it = config.cadence_settings.find(cadence_id);
  if (it == config.cadence_settings.end()) {
    throw std::runtime_error("Unknown cadence_id: " + cadence_id);
  }
  const auto& settings = it->second;
  CadenceSpec spec;
  spec.cadence_id = cadence_id;
  spec.cadence_label = settings.label;
  spec.schedule_type = settings.schedule_type;
  spec.periods_per_year = settings.periods_per_year;
  spec.canonical_offset_id = settings.canonical_offset_id;
  spec.offset_ids = settings.offset_ids;
  spec.offset_label_prefix = settings.offset_label_prefix;
  return spec;
}

std::string cadence_period_label(int periods_per_year) {
  return periods_per_year == 12 ? "months" : "periods";
}

struct ArrowColumnCache {
  std::shared_ptr<arrow::StringArray> string_column;
  std::shared_ptr<arrow::TimestampArray> timestamp_column;
  std::shared_ptr<arrow::DoubleArray> double_column;
};

ArrowColumnCache combine_string(const std::shared_ptr<arrow::ChunkedArray>& chunked) {
  auto casted = arrow::compute::Cast(chunked, arrow::utf8()).ValueOrDie();
  auto combined = arrow::Concatenate(casted.chunked_array()->chunks(), arrow::default_memory_pool()).ValueOrDie();
  ArrowColumnCache cache;
  cache.string_column = std::static_pointer_cast<arrow::StringArray>(combined);
  return cache;
}

ArrowColumnCache combine_timestamp(const std::shared_ptr<arrow::ChunkedArray>& chunked) {
  auto combined = arrow::Concatenate(chunked->chunks(), arrow::default_memory_pool()).ValueOrDie();
  ArrowColumnCache cache;
  cache.timestamp_column = std::static_pointer_cast<arrow::TimestampArray>(combined);
  return cache;
}

ArrowColumnCache combine_double(const std::shared_ptr<arrow::ChunkedArray>& chunked) {
  auto combined = arrow::Concatenate(chunked->chunks(), arrow::default_memory_pool()).ValueOrDie();
  ArrowColumnCache cache;
  cache.double_column = std::static_pointer_cast<arrow::DoubleArray>(combined);
  return cache;
}

struct FxKey {
  int currency_id;
  int64_t date;
  bool operator==(const FxKey& other) const { return currency_id == other.currency_id && date == other.date; }
};
struct FxHash {
  size_t operator()(const FxKey& key) const {
    return std::hash<long long>()((static_cast<long long>(key.currency_id) << 32) ^ key.date);
  }
};

struct SecDateKey {
  int sec_idx;
  int64_t date;
  bool operator==(const SecDateKey& other) const { return sec_idx == other.sec_idx && date == other.date; }
};
struct SecDateHash {
  size_t operator()(const SecDateKey& key) const {
    return std::hash<long long>()((static_cast<long long>(key.sec_idx) << 32) ^ key.date);
  }
};

class CadenceDataset {
 public:
  CadenceDataset(const std::filesystem::path& data_dir, const Config& config, const CadenceSpec& cadence, int offset_id)
      : data_dir_(data_dir.string()), config_(config), base_(data_dir.string(), config) {
    if (std::find(cadence.offset_ids.begin(), cadence.offset_ids.end(), offset_id) == cadence.offset_ids.end()) {
      throw std::runtime_error("offset_id not valid for cadence");
    }
    cadence_ = cadence;
    offset_id_ = offset_id;
    for (size_t i = 0; i < base_.security_ids().size(); ++i) {
      security_index_[base_.security_ids()[i]] = static_cast<int>(i);
    }
    build_trade_calendar();
    build_rebalance_dates();
    prepare_daily_prices();
    build_entry_exit_indices();
    build_holding_months();
    precompute_scores();
    precompute_rank_orders();
    precompute_holding_returns();
    build_benchmark_period_returns();
  }

  const CadenceSpec& cadence() const { return cadence_; }

  WindowSimulation simulate_window(
      const CandidateParams& params,
      const std::string& universe_variant,
      const std::string& execution_model,
      const std::string& fx_scenario,
      const std::string& start_month,
      const std::string& end_month,
      const std::vector<std::string>& excluded_countries,
      std::optional<int> shuffled_selection_seed = std::nullopt) const;

  std::vector<std::vector<PositionSample>> negative_control_months(
      const CandidateParams& params,
      const std::string& start_month,
      const std::string& end_month,
      const std::vector<std::string>& excluded_countries) const;

  bool variant_has_any(const std::string& universe_variant, const std::vector<std::string>& excluded_countries) const;

 private:
  long long score_key(int lookback, int skip) const {
    return (static_cast<long long>(lookback) << 32) | static_cast<unsigned int>(skip);
  }

  void build_trade_calendar();
  void build_rebalance_dates();
  void prepare_daily_prices();
  int price_index_for_date(int64_t target_date) const;
  void build_entry_exit_indices();
  void build_holding_months();
  void precompute_scores();
  void precompute_rank_orders();
  void precompute_holding_returns();
  void build_benchmark_period_returns();
  std::vector<int> window_signal_indices(const std::string& start_month, const std::string& end_month) const;

  std::string data_dir_;
  const Config& config_;
  ResearchDataset base_;
  CadenceSpec cadence_;
  int offset_id_ = 0;

  std::vector<int64_t> trade_calendar_;
  std::vector<int64_t> rebalance_dates_;
  std::vector<int> entry_indices_;
  std::vector<int> exit_indices_;
  std::vector<std::string> holding_months_;
  std::vector<int> period_month_indices_;
  std::unordered_map<std::string, int> period_index_by_month_;

  std::unordered_map<std::string, int> security_index_;

  std::vector<double> open_bfill_;
  std::vector<double> close_bfill_;
  std::vector<double> close_ffill_;

  std::unordered_map<long long, std::vector<double>> scores_;
  std::unordered_map<long long, std::vector<std::vector<int>>> rank_orders_;

  std::vector<double> returns_open_;
  std::vector<double> returns_close_;
  std::vector<uint8_t> entry_open_ok_;
  std::vector<uint8_t> entry_close_ok_;

  std::unordered_map<std::string, std::vector<double>> benchmark_period_returns_;
};

WindowSimulation CadenceDataset::simulate_window(
    const CandidateParams& params,
    const std::string& universe_variant,
    const std::string& execution_model,
    const std::string& fx_scenario,
    const std::string& start_month,
    const std::string& end_month,
    const std::vector<std::string>& excluded_countries,
    std::optional<int> shuffled_selection_seed) const {
  WindowSimulation simulation;
  auto indices = window_signal_indices(start_month, end_month);
  if (indices.empty()) return simulation;

  int exec_id = execution_model == "next_open" ? 0 : 1;
  const auto& score_matrix = scores_.at(score_key(params.lookback, params.skip));
  const auto& rank_orders = rank_orders_.at(score_key(params.lookback, params.skip));
  const auto& capacity_mask = base_.capacity_mask(params.top_n);
  const auto& cost_fraction = base_.cost_fractions(params.top_n, execution_model, fx_scenario);
  const auto& base_mask = (universe_variant == "SE-only")
                              ? base_.eligible_se_only_mask()
                              : base_.eligible_full_nordics_mask();
  const auto& asof_mask = base_.asof_matches_anchor_mask();

  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());
  std::vector<int> previous_selection;
  std::mt19937 rng;
  if (shuffled_selection_seed) {
    rng.seed(*shuffled_selection_seed);
  }

  int64_t n_sec = base_.security_count();
  for (int period_idx : indices) {
    int month_idx = period_month_indices_[period_idx];
    if (month_idx < 0) continue;

    std::vector<int> eligible;
    eligible.reserve(params.top_n * 2);
    for (int sec_idx : rank_orders[period_idx]) {
      int64_t idx = static_cast<int64_t>(month_idx) * n_sec + sec_idx;
      if (!base_mask[idx] || !asof_mask[idx] || !capacity_mask[idx]) {
        continue;
      }
      if (excluded.count(base_.security_country()[sec_idx])) {
        continue;
      }
      double score = score_matrix[static_cast<int64_t>(period_idx) * n_sec + sec_idx];
      if (!std::isfinite(score)) {
        continue;
      }
      int64_t period_offset = static_cast<int64_t>(period_idx) * n_sec + sec_idx;
      double ret = exec_id == 0 ? returns_open_[period_offset] : returns_close_[period_offset];
      if (!std::isfinite(ret)) {
        continue;
      }
      bool entry_ok = exec_id == 0 ? entry_open_ok_[period_offset] : entry_close_ok_[period_offset];
      if (!entry_ok) {
        continue;
      }
      eligible.push_back(sec_idx);
      if (!shuffled_selection_seed && static_cast<int>(eligible.size()) >= params.top_n) {
        break;
      }
    }

    std::vector<int> selected;
    if (shuffled_selection_seed) {
      std::shuffle(eligible.begin(), eligible.end(), rng);
      selected.assign(eligible.begin(), eligible.begin() + std::min<int>(params.top_n, eligible.size()));
    } else {
      selected = eligible;
    }

    std::unordered_set<int> prev_set(previous_selection.begin(), previous_selection.end());
    std::unordered_set<int> current_set(selected.begin(), selected.end());
    double gross_return = 0.0;
    for (int sec_idx : selected) {
      int64_t period_offset = static_cast<int64_t>(period_idx) * n_sec + sec_idx;
      double ret = exec_id == 0 ? returns_open_[period_offset] : returns_close_[period_offset];
      if (std::isfinite(ret)) {
        gross_return += ret;
      }
    }
    gross_return = selected.empty() ? 0.0 : gross_return / static_cast<double>(params.top_n);
    double trade_cost = 0.0;
    for (int sec_idx : selected) {
      if (!prev_set.count(sec_idx)) {
        int64_t idx = static_cast<int64_t>(month_idx) * n_sec + sec_idx;
        trade_cost += cost_fraction[idx];
      }
    }
    for (int sec_idx : previous_selection) {
      if (!current_set.count(sec_idx)) {
        int64_t idx = static_cast<int64_t>(month_idx) * n_sec + sec_idx;
        trade_cost += cost_fraction[idx];
      }
    }
    double net_return = gross_return - trade_cost;
    simulation.months.push_back(holding_months_[period_idx]);
    simulation.monthly_returns.push_back(net_return);
    previous_selection = selected;
  }

  auto primary_it = benchmark_period_returns_.find(config_.primary_passive_benchmark_id);
  auto secondary_it = benchmark_period_returns_.find(config_.secondary_opportunity_cost_benchmark_id);
  bool primary_ok = primary_it != benchmark_period_returns_.end();
  bool secondary_ok = secondary_it != benchmark_period_returns_.end();
  for (const auto& month : simulation.months) {
    auto period_it = period_index_by_month_.find(month);
    if (period_it == period_index_by_month_.end()) continue;
    int period_idx = period_it->second;
    if (primary_ok) {
      simulation.primary_benchmark_returns.push_back(primary_it->second[period_idx]);
    }
    if (secondary_ok) {
      simulation.secondary_benchmark_returns.push_back(secondary_it->second[period_idx]);
    }
  }

  return simulation;
}

std::vector<std::vector<PositionSample>> CadenceDataset::negative_control_months(
    const CandidateParams& params,
    const std::string& start_month,
    const std::string& end_month,
    const std::vector<std::string>& excluded_countries) const {
  std::vector<std::vector<PositionSample>> months;
  auto indices = window_signal_indices(start_month, end_month);
  if (indices.empty()) return months;

  const auto& score_matrix = scores_.at(score_key(params.lookback, params.skip));
  const auto& capacity_mask = base_.capacity_mask(params.top_n);
  const auto& base_mask = base_.eligible_full_nordics_mask();
  const auto& asof_mask = base_.asof_matches_anchor_mask();

  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());
  int64_t n_sec = base_.security_count();
  for (int period_idx : indices) {
    int month_idx = period_month_indices_[period_idx];
    if (month_idx < 0) continue;
    std::vector<PositionSample> positions;
    positions.reserve(n_sec);
    for (int sec_idx = 0; sec_idx < n_sec; ++sec_idx) {
      int64_t idx = static_cast<int64_t>(month_idx) * n_sec + sec_idx;
      if (!base_mask[idx] || !asof_mask[idx] || !capacity_mask[idx]) continue;
      if (excluded.count(base_.security_country()[sec_idx])) continue;
      double score = score_matrix[static_cast<int64_t>(period_idx) * n_sec + sec_idx];
      double ret = returns_open_[static_cast<int64_t>(period_idx) * n_sec + sec_idx];
      if (!std::isfinite(score) || !std::isfinite(ret)) continue;
      if (!entry_open_ok_[static_cast<int64_t>(period_idx) * n_sec + sec_idx]) continue;
      positions.push_back({score, ret});
    }
    months.push_back(std::move(positions));
  }
  return months;
}

bool CadenceDataset::variant_has_any(const std::string& universe_variant, const std::vector<std::string>& excluded_countries) const {
  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());
  const auto& base_mask = (universe_variant == "SE-only")
                              ? base_.eligible_se_only_mask()
                              : base_.eligible_full_nordics_mask();
  const auto& asof_mask = base_.asof_matches_anchor_mask();
  int64_t n_sec = base_.security_count();
  for (int64_t month_idx = 0; month_idx < base_.month_count(); ++month_idx) {
    for (int64_t sec_idx = 0; sec_idx < n_sec; ++sec_idx) {
      int64_t idx = month_idx * n_sec + sec_idx;
      if (!base_mask[idx] || !asof_mask[idx]) continue;
      if (excluded.count(base_.security_country()[sec_idx])) continue;
      return true;
    }
  }
  return false;
}

void CadenceDataset::build_trade_calendar() {
  auto raw_result = read_parquet(std::filesystem::path(data_dir_) / "prices_raw_daily.parquet");
  if (!raw_result.ok()) {
    throw std::runtime_error(raw_result.status().ToString());
  }
  auto raw = *raw_result;
  auto date_col = combine_timestamp(raw->GetColumnByName("date")).timestamp_column;
  trade_calendar_.reserve(raw->num_rows());
  for (int64_t i = 0; i < raw->num_rows(); ++i) {
    trade_calendar_.push_back(date_col->Value(i));
  }
  std::sort(trade_calendar_.begin(), trade_calendar_.end());
  trade_calendar_.erase(std::unique(trade_calendar_.begin(), trade_calendar_.end()), trade_calendar_.end());
}

void CadenceDataset::build_rebalance_dates() {
  if (cadence_.schedule_type == "month_end" || cadence_.schedule_type == "quarter_end" ||
      cadence_.schedule_type == "half_year_end") {
    std::vector<int64_t> month_end;
    int current_year = -1;
    int current_month = -1;
    int64_t last_date = 0;
    for (auto ts : trade_calendar_) {
      auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
          std::chrono::nanoseconds(ts));
      auto days = std::chrono::floor<std::chrono::days>(tp);
      std::chrono::year_month_day ymd(days);
      int year = static_cast<int>(ymd.year());
      int month = static_cast<unsigned>(ymd.month());
      if (year != current_year || month != current_month) {
        if (last_date != 0) month_end.push_back(last_date);
        current_year = year;
        current_month = month;
      }
      last_date = ts;
    }
    if (last_date != 0) month_end.push_back(last_date);

    if (cadence_.schedule_type == "month_end") {
      rebalance_dates_ = month_end;
      return;
    }
    int period = cadence_.schedule_type == "quarter_end" ? 3 : 6;
    int remainder = (period - offset_id_) % period;
    for (auto ts : month_end) {
      auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
          std::chrono::nanoseconds(ts));
      auto days = std::chrono::floor<std::chrono::days>(tp);
      std::chrono::year_month_day ymd(days);
      int month = static_cast<unsigned>(ymd.month());
      if ((month % period) == remainder) {
        rebalance_dates_.push_back(ts);
      }
    }
    return;
  }

  if (cadence_.schedule_type == "iso_weekly" || cadence_.schedule_type == "iso_biweekly") {
    IsoWeekKey current{0, 0};
    int64_t last_date = 0;
    for (auto ts : trade_calendar_) {
      auto key = iso_week_key(ts);
      if (key.year != current.year || key.week != current.week) {
        if (last_date != 0) {
          rebalance_dates_.push_back(last_date);
        }
        current = key;
      }
      last_date = ts;
    }
    if (last_date != 0) rebalance_dates_.push_back(last_date);

    if (cadence_.schedule_type == "iso_biweekly") {
      std::vector<int64_t> filtered;
      for (auto ts : rebalance_dates_) {
        auto key = iso_week_key(ts);
        int week_index = key.year * 53 + key.week;
        int parity = week_index % 2;
        if (parity == offset_id_) {
          filtered.push_back(ts);
        }
      }
      rebalance_dates_.swap(filtered);
    }
    return;
  }

  throw std::runtime_error("Unsupported schedule_type " + cadence_.schedule_type);
}

void CadenceDataset::prepare_daily_prices() {
  auto raw_result = read_parquet(std::filesystem::path(data_dir_) / "prices_raw_daily.parquet");
  auto adj_result = read_parquet(std::filesystem::path(data_dir_) / "prices_adjusted_daily.parquet");
  auto fx_result = read_parquet(std::filesystem::path(data_dir_) / "riksbank_fx_daily.parquet");
  if (!raw_result.ok() || !adj_result.ok() || !fx_result.ok()) {
    throw std::runtime_error("Failed to load daily price artifacts.");
  }
  auto raw = *raw_result;
  auto adj = *adj_result;
  auto fx = *fx_result;

  auto fx_currency = combine_string(fx->GetColumnByName("currency")).string_column;
  auto fx_date = combine_timestamp(fx->GetColumnByName("date")).timestamp_column;
  auto fx_value = combine_double(fx->GetColumnByName("sek_per_ccy")).double_column;
  std::unordered_map<std::string, int> currency_ids;
  int next_currency_id = 0;
  std::unordered_map<FxKey, double, FxHash> fx_map;
  fx_map.reserve(fx->num_rows());
  for (int64_t i = 0; i < fx->num_rows(); ++i) {
    std::string currency = std::string(fx_currency->GetView(i));
    auto it = currency_ids.find(currency);
    if (it == currency_ids.end()) {
      it = currency_ids.emplace(currency, next_currency_id++).first;
    }
    fx_map[{it->second, fx_date->Value(i)}] = fx_value->Value(i);
  }

  auto adj_security = combine_string(adj->GetColumnByName("security_id")).string_column;
  auto adj_date = combine_timestamp(adj->GetColumnByName("date")).timestamp_column;
  auto adj_factor = combine_double(adj->GetColumnByName("adj_factor")).double_column;
  std::unordered_map<SecDateKey, double, SecDateHash> adj_factor_map;
  adj_factor_map.reserve(adj->num_rows());
  for (int64_t i = 0; i < adj->num_rows(); ++i) {
    std::string security = std::string(adj_security->GetView(i));
    auto it = security_index_.find(security);
    if (it == security_index_.end()) continue;
    adj_factor_map[{it->second, adj_date->Value(i)}] = adj_factor->Value(i);
  }

  int64_t n_dates = static_cast<int64_t>(trade_calendar_.size());
  int64_t n_sec = base_.security_count();
  open_bfill_.assign(n_dates * n_sec, kNaN);
  close_bfill_.assign(n_dates * n_sec, kNaN);
  close_ffill_.assign(n_dates * n_sec, kNaN);

  std::unordered_map<int64_t, int64_t> date_index;
  date_index.reserve(trade_calendar_.size());
  for (int64_t i = 0; i < n_dates; ++i) {
    date_index[trade_calendar_[i]] = i;
  }

  auto raw_security = combine_string(raw->GetColumnByName("security_id")).string_column;
  auto raw_date = combine_timestamp(raw->GetColumnByName("date")).timestamp_column;
  auto raw_open = combine_double(raw->GetColumnByName("open_raw")).double_column;
  auto raw_close = combine_double(raw->GetColumnByName("close_raw")).double_column;
  auto raw_currency = combine_string(raw->GetColumnByName("currency")).string_column;
  for (int64_t i = 0; i < raw->num_rows(); ++i) {
    std::string security = std::string(raw_security->GetView(i));
    auto sec_it = security_index_.find(security);
    if (sec_it == security_index_.end()) continue;
    int sec_idx = sec_it->second;
    int64_t date = raw_date->Value(i);
    auto date_it = date_index.find(date);
    if (date_it == date_index.end()) continue;
    int64_t day_idx = date_it->second;
    auto factor_it = adj_factor_map.find({sec_idx, date});
    if (factor_it == adj_factor_map.end()) continue;
    std::string currency = std::string(raw_currency->GetView(i));
    auto c_it = currency_ids.find(currency);
    if (c_it == currency_ids.end()) continue;
    auto fx_it = fx_map.find({c_it->second, date});
    if (fx_it == fx_map.end()) continue;
    double factor = factor_it->second;
    double fx_rate = fx_it->second;
    double adj_open = raw_open->Value(i) * factor * fx_rate;
    double adj_close = raw_close->Value(i) * factor * fx_rate;
    open_bfill_[day_idx * n_sec + sec_idx] = adj_open;
    close_bfill_[day_idx * n_sec + sec_idx] = adj_close;
  }

  for (int sec_idx = 0; sec_idx < n_sec; ++sec_idx) {
    double next_open = kNaN;
    double next_close = kNaN;
    for (int64_t day = n_dates - 1; day >= 0; --day) {
      int64_t idx = day * n_sec + sec_idx;
      if (std::isfinite(open_bfill_[idx])) {
        next_open = open_bfill_[idx];
      } else if (std::isfinite(next_open)) {
        open_bfill_[idx] = next_open;
      }
      if (std::isfinite(close_bfill_[idx])) {
        next_close = close_bfill_[idx];
      } else if (std::isfinite(next_close)) {
        close_bfill_[idx] = next_close;
      }
    }
    double last_close = kNaN;
    for (int64_t day = 0; day < n_dates; ++day) {
      int64_t idx = day * n_sec + sec_idx;
      if (std::isfinite(close_bfill_[idx])) {
        last_close = close_bfill_[idx];
        close_ffill_[idx] = close_bfill_[idx];
      } else if (std::isfinite(last_close)) {
        close_ffill_[idx] = last_close;
      }
    }
  }
}

int CadenceDataset::price_index_for_date(int64_t target_date) const {
  auto it = std::upper_bound(trade_calendar_.begin(), trade_calendar_.end(), target_date);
  if (it == trade_calendar_.begin()) return -1;
  return static_cast<int>(std::distance(trade_calendar_.begin(), it) - 1);
}

void CadenceDataset::build_entry_exit_indices() {
  entry_indices_.clear();
  for (auto ts : rebalance_dates_) {
    auto it = std::upper_bound(trade_calendar_.begin(), trade_calendar_.end(), ts);
    if (it == trade_calendar_.end()) {
      entry_indices_.push_back(-1);
    } else {
      entry_indices_.push_back(static_cast<int>(std::distance(trade_calendar_.begin(), it)));
    }
  }
  exit_indices_.assign(entry_indices_.begin() + 1, entry_indices_.end());
  exit_indices_.push_back(-1);
}

void CadenceDataset::build_holding_months() {
  holding_months_.assign(entry_indices_.size(), "");
  period_month_indices_.assign(entry_indices_.size(), -1);
  for (size_t idx = 0; idx < entry_indices_.size(); ++idx) {
    int entry_index = entry_indices_[idx];
    if (entry_index < 0 || entry_index >= static_cast<int>(trade_calendar_.size())) {
      continue;
    }
    std::string month_label = month_label_from_ts(trade_calendar_[entry_index]);
    holding_months_[idx] = month_label;
    period_month_indices_[idx] = base_.month_index_for(month_label);
    period_index_by_month_[month_label] = static_cast<int>(idx);
  }
}

void CadenceDataset::precompute_scores() {
  std::unordered_set<int> lookbacks;
  std::unordered_set<int> skips;
  for (const auto& [set_name, profiles] : config_.profile_sets) {
    for (const auto& [profile_name, profile] : profiles) {
      lookbacks.insert(profile.lookbacks.begin(), profile.lookbacks.end());
      skips.insert(profile.skips.begin(), profile.skips.end());
    }
  }
  int64_t n_sec = base_.security_count();
  for (int lookback : lookbacks) {
    for (int skip : skips) {
      std::vector<double> matrix(rebalance_dates_.size() * n_sec, kNaN);
      for (size_t period_idx = 0; period_idx < rebalance_dates_.size(); ++period_idx) {
        auto rebalance_ts = rebalance_dates_[period_idx];
        auto rebalance_tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
            std::chrono::nanoseconds(rebalance_ts));
        auto rebalance_days = std::chrono::floor<std::chrono::days>(rebalance_tp);
        std::chrono::year_month_day rebalance_ymd(rebalance_days);
        auto current_ymd = rebalance_ymd - std::chrono::months(skip);
        auto previous_ymd = rebalance_ymd - std::chrono::months(skip + lookback);
        auto current_days = std::chrono::sys_days{current_ymd};
        auto previous_days = std::chrono::sys_days{previous_ymd};
        int current_index = price_index_for_date(std::chrono::time_point_cast<std::chrono::nanoseconds>(current_days).time_since_epoch().count());
        int previous_index = price_index_for_date(std::chrono::time_point_cast<std::chrono::nanoseconds>(previous_days).time_since_epoch().count());
        if (current_index < 0 || previous_index < 0) continue;
        for (int64_t sec_idx = 0; sec_idx < n_sec; ++sec_idx) {
          double current = close_ffill_[static_cast<int64_t>(current_index) * n_sec + sec_idx];
          double previous = close_ffill_[static_cast<int64_t>(previous_index) * n_sec + sec_idx];
          if (!std::isfinite(current) || !std::isfinite(previous) || current <= 0.0 || previous <= 0.0) {
            continue;
          }
          matrix[static_cast<int64_t>(period_idx) * n_sec + sec_idx] = current / previous - 1.0;
        }
      }
      scores_.emplace(score_key(lookback, skip), std::move(matrix));
    }
  }
}

void CadenceDataset::precompute_rank_orders() {
  int64_t n_sec = base_.security_count();
  for (const auto& [key, matrix] : scores_) {
    std::vector<std::vector<int>> orders;
    orders.reserve(rebalance_dates_.size());
    for (size_t period_idx = 0; period_idx < rebalance_dates_.size(); ++period_idx) {
      std::vector<int> indices(n_sec);
      std::iota(indices.begin(), indices.end(), 0);
      std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        double score_a = matrix[static_cast<int64_t>(period_idx) * n_sec + a];
        double score_b = matrix[static_cast<int64_t>(period_idx) * n_sec + b];
        bool a_valid = std::isfinite(score_a);
        bool b_valid = std::isfinite(score_b);
        if (a_valid != b_valid) return a_valid > b_valid;
        if (!a_valid && !b_valid) return a < b;
        if (score_a == score_b) return a < b;
        return score_a > score_b;
      });
      orders.push_back(std::move(indices));
    }
    rank_orders_.emplace(key, std::move(orders));
  }
}

void CadenceDataset::precompute_holding_returns() {
  int64_t n_periods = static_cast<int64_t>(rebalance_dates_.size());
  int64_t n_sec = base_.security_count();
  returns_open_.assign(n_periods * n_sec, kNaN);
  returns_close_.assign(n_periods * n_sec, kNaN);
  entry_open_ok_.assign(n_periods * n_sec, 0);
  entry_close_ok_.assign(n_periods * n_sec, 0);

  double fallback_multiplier = 1.0 + config_.delist_fallback_haircut;
  for (int64_t period_idx = 0; period_idx < n_periods; ++period_idx) {
    int entry_index = entry_indices_[period_idx];
    int exit_index = exit_indices_[period_idx];
    if (entry_index < 0 || exit_index < 0 || exit_index <= entry_index) {
      continue;
    }
    for (int64_t sec_idx = 0; sec_idx < n_sec; ++sec_idx) {
      int64_t entry_offset = static_cast<int64_t>(entry_index) * n_sec + sec_idx;
      int64_t exit_offset = static_cast<int64_t>(exit_index) * n_sec + sec_idx;
      double entry_open = open_bfill_[entry_offset];
      double entry_close = close_bfill_[entry_offset];
      double exit_open_forward = open_bfill_[exit_offset];
      double exit_close_forward = close_bfill_[exit_offset];
      double exit_close_backward = close_ffill_[exit_offset];
      double exit_open = std::isfinite(exit_open_forward) ? exit_open_forward
                                                         : exit_close_backward * fallback_multiplier;
      double exit_close = std::isfinite(exit_close_forward) ? exit_close_forward
                                                           : exit_close_backward * fallback_multiplier;
      bool entry_open_ok = std::isfinite(entry_open) && entry_open > 0.0;
      bool entry_close_ok = std::isfinite(entry_close) && entry_close > 0.0;
      entry_open_ok_[period_idx * n_sec + sec_idx] = entry_open_ok ? 1 : 0;
      entry_close_ok_[period_idx * n_sec + sec_idx] = entry_close_ok ? 1 : 0;
      if (entry_open_ok && std::isfinite(exit_open) && exit_open > 0.0) {
        returns_open_[period_idx * n_sec + sec_idx] = exit_open / entry_open - 1.0;
      }
      if (entry_close_ok && std::isfinite(exit_close) && exit_close > 0.0) {
        returns_close_[period_idx * n_sec + sec_idx] = exit_close / entry_close - 1.0;
      }
    }
  }
}

void CadenceDataset::build_benchmark_period_returns() {
  auto bench_result = read_parquet(std::filesystem::path(data_dir_) / "benchmark_prices.parquet");
  auto fx_result = read_parquet(std::filesystem::path(data_dir_) / "riksbank_fx_daily.parquet");
  if (!bench_result.ok() || !fx_result.ok()) {
    return;
  }
  auto bench = *bench_result;
  auto fx = *fx_result;

  auto fx_currency = combine_string(fx->GetColumnByName("currency")).string_column;
  auto fx_date = combine_timestamp(fx->GetColumnByName("date")).timestamp_column;
  auto fx_value = combine_double(fx->GetColumnByName("sek_per_ccy")).double_column;
  struct FxKey2 {
    std::string currency;
    int64_t date;
    bool operator==(const FxKey2& other) const { return currency == other.currency && date == other.date; }
  };
  struct FxHash2 {
    size_t operator()(const FxKey2& key) const {
      return std::hash<std::string>()(key.currency) ^ std::hash<long long>()(key.date);
    }
  };
  std::unordered_map<FxKey2, double, FxHash2> fx_map;
  fx_map.reserve(fx->num_rows());
  for (int64_t i = 0; i < fx->num_rows(); ++i) {
    fx_map[{std::string(fx_currency->GetView(i)), fx_date->Value(i)}] = fx_value->Value(i);
  }

  auto bench_id = combine_string(bench->GetColumnByName("benchmark_id")).string_column;
  auto bench_date = combine_timestamp(bench->GetColumnByName("date")).timestamp_column;
  auto bench_currency = combine_string(bench->GetColumnByName("currency")).string_column;
  auto bench_adj = combine_double(bench->GetColumnByName("adj_close")).double_column;

  std::unordered_map<std::string, std::vector<double>> series_by_id;
  int64_t n_dates = static_cast<int64_t>(trade_calendar_.size());
  for (int64_t i = 0; i < bench->num_rows(); ++i) {
    std::string id = std::string(bench_id->GetView(i));
    auto fx_it = fx_map.find({std::string(bench_currency->GetView(i)), bench_date->Value(i)});
    if (fx_it == fx_map.end()) continue;
    double price_sek = bench_adj->Value(i) * fx_it->second;
    auto date_it = std::lower_bound(trade_calendar_.begin(), trade_calendar_.end(), bench_date->Value(i));
    if (date_it == trade_calendar_.end() || *date_it != bench_date->Value(i)) continue;
    int64_t idx = std::distance(trade_calendar_.begin(), date_it);
    if (!series_by_id.count(id)) {
      series_by_id[id] = std::vector<double>(n_dates, kNaN);
    }
    series_by_id[id][idx] = price_sek;
  }

  for (auto& [id, series] : series_by_id) {
    std::vector<double> bfill = series;
    double next_val = kNaN;
    for (int64_t i = n_dates - 1; i >= 0; --i) {
      if (std::isfinite(bfill[i])) {
        next_val = bfill[i];
      } else if (std::isfinite(next_val)) {
        bfill[i] = next_val;
      }
    }
    std::vector<double> period_returns;
    period_returns.reserve(entry_indices_.size());
    for (size_t p = 0; p < entry_indices_.size(); ++p) {
      int entry_index = entry_indices_[p];
      int exit_index = exit_indices_[p];
      if (entry_index < 0 || exit_index < 0 || exit_index <= entry_index) {
        period_returns.push_back(kNaN);
        continue;
      }
      double entry_price = bfill[entry_index];
      double exit_price = bfill[exit_index];
      if (!std::isfinite(entry_price) || !std::isfinite(exit_price) || entry_price <= 0.0) {
        period_returns.push_back(kNaN);
      } else {
        period_returns.push_back(exit_price / entry_price - 1.0);
      }
    }
    benchmark_period_returns_[id] = std::move(period_returns);
  }
}

std::vector<int> CadenceDataset::window_signal_indices(const std::string& start_month, const std::string& end_month) const {
  int start = month_ordinal(start_month);
  int end = month_ordinal(end_month);
  std::vector<int> indices;
  for (size_t idx = 0; idx < holding_months_.size(); ++idx) {
    if (holding_months_[idx].empty()) continue;
    int ord = month_ordinal(holding_months_[idx]);
    if (ord >= start && ord <= end) {
      indices.push_back(static_cast<int>(idx));
    }
  }
  return indices;
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
    const CadenceDataset& dataset,
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
    const CadenceDataset& dataset,
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
  CandidateParams params{control_candidate.params.lookback, control_candidate.params.skip, control_candidate.params.top_n};

  std::vector<std::vector<std::pair<double, double>>> months;
  auto negative_months = dataset.negative_control_months(
      params,
      config.rolling_folds.front().validate_start,
      config.rolling_folds.back().validate_end,
      excluded_countries);
  for (const auto& month : negative_months) {
    std::vector<std::pair<double, double>> positions;
    positions.reserve(month.size());
    for (const auto& pos : month) {
      positions.push_back({pos.score, pos.next_return});
    }
    months.push_back(std::move(positions));
  }

  auto shuffle_runs = cross_sectional_score_shuffle_runs(
      months, params.top_n, profile.cross_sectional_shuffle_runs, 11);
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
    const CadenceDataset& dataset,
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
                params,
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

json selection_summary(
    const std::vector<CandidateAggregate>& aggregates,
    const json& negative_controls,
    const std::string& mode,
    int periods_per_year,
    const std::string& period_label,
    const std::vector<std::string>& active_variants) {
  json summary;
  summary["mode"] = mode;
  summary["selection_status"] = "no_candidate_passed_hard_gates";
  summary["locked_candidate"] = nullptr;
  summary["ranked_candidates"] = json::array();
  summary["negative_controls"] = negative_controls;
  summary["neighbor_diagnostics"] = json::array();
  summary["period_label"] = period_label;
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
    candidate["period_label"] = period_label;
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

json evaluate_holdout(
    const CadenceDataset& dataset,
    const ThesisSettings& thesis,
    const CandidateParams& params,
    int periods_per_year,
    const std::string& period_label,
    const Config& config) {
  json results = json::object();
  for (const auto& variant : kRequiredUniverseVariants) {
    for (const auto& exec : kRequiredExecutionModels) {
      for (const auto& fx : kRequiredFxScenarios) {
        WindowSimulation simulation = dataset.simulate_window(
            params,
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
        row["period_label"] = period_label;
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
  holdout["period_label"] = period_label;
  holdout["periods_per_year"] = periods_per_year;
  return holdout;
}

struct ParsedArgs {
  std::string data_dir = "data";
  std::string results_root = "results/cadence_compare_rebuild";
  std::vector<std::string> theses;
  std::vector<std::string> cadences;
  std::string profile_set = "default";
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
  parsed.results_root = args.value_or("--results-root", parsed.results_root);
  parsed.profile_set = args.value_or("--profile-set", parsed.profile_set);

  parsed.theses = parse_list_flag(args, "--theses");
  if (parsed.theses.empty()) {
    for (const auto& entry : config.theses) {
      parsed.theses.push_back(entry.first);
    }
    std::sort(parsed.theses.begin(), parsed.theses.end());
  }

  parsed.cadences = parse_list_flag(args, "--cadences");
  if (parsed.cadences.empty()) {
    parsed.cadences = config.default_cadence_compare_cadences;
  }
  return parsed;
}

json attach_metadata(const json& summary, const json& thesis_meta, const json& cadence_meta) {
  json output = summary;
  output["thesis"] = thesis_meta;
  output["cadence"] = cadence_meta;
  output["cadence_id"] = cadence_meta.value("cadence_id", "");
  output["cadence_label"] = cadence_meta.value("cadence_label", "");
  return output;
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

  std::filesystem::path results_root(args.results_root);
  std::filesystem::create_directories(results_root);
  std::filesystem::path summary_dir = results_root / "summary";
  std::filesystem::create_directories(summary_dir);

  std::vector<json> pairs;

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

    for (const auto& cadence_id : args.cadences) {
      CadenceSpec spec = load_cadence_spec(config, cadence_id);
      CadenceDataset dataset(args.data_dir, config, spec, spec.canonical_offset_id);
      std::string period_label = cadence_period_label(spec.periods_per_year);

      json cadence_meta = {
          {"cadence_id", spec.cadence_id},
          {"cadence_label", spec.cadence_label},
          {"canonical_offset_id", spec.canonical_offset_id},
          {"schedule_type", spec.schedule_type},
          {"rebalance_logic", kRebalanceLogic},
          {"rebalance_logic_label", kRebalanceLogicLabel},
      };

      std::unordered_map<std::string, json> summaries;
      for (const auto& profile_name : {"quick", "mega", "certification_baseline"}) {
        auto profile_it = profile_set.find(profile_name);
        if (profile_it == profile_set.end()) {
          spdlog::warn("Skipping profile '{}' (not in profile-set {})", profile_name, args.profile_set);
          continue;
        }
        const auto& profile = profile_it->second;
        auto grid = build_param_grid(profile);
        auto candidates = build_candidate_evaluations(dataset, thesis, grid, spec.periods_per_year, config);
        auto negative_controls =
            compute_negative_controls(dataset, candidates, spec.periods_per_year, profile, thesis.excluded_countries, config);
        auto aggregates = aggregate_candidates(candidates, negative_controls, spec.periods_per_year, config);
        attach_plateau_diagnostics(aggregates);
        attach_ranks(aggregates);
        auto active_variants = active_selection_variants(dataset, thesis.excluded_countries);
        auto summary =
            selection_summary(aggregates, negative_controls, profile_name, spec.periods_per_year, period_label, active_variants);
        summary["backtest_overfitting"] = compute_cscv_pbo(candidates, spec.periods_per_year, period_label, config);
        summary["thesis"] = thesis_meta;
        summary["profile_name"] = profile_name;
        summaries[profile_name] = summary;
      }

      json quick = summaries.count("quick") ? summaries["quick"] : json::object();
      json mega = summaries.count("mega") ? summaries["mega"] : json::object();
      json certification = summaries.count("certification_baseline") ? summaries["certification_baseline"] : json::object();

      quick = attach_metadata(quick, thesis_meta, cadence_meta);
      mega = attach_metadata(mega, thesis_meta, cadence_meta);
      certification = attach_metadata(certification, thesis_meta, cadence_meta);

      json holdout;
      if (certification.contains("locked_candidate") && !certification["locked_candidate"].is_null()) {
        auto locked = certification["locked_candidate"];
        CandidateParams params{locked["params"]["l"], locked["params"]["skip"], locked["params"]["top_n"]};
        holdout = evaluate_holdout(dataset, thesis, params, spec.periods_per_year, period_label, config);
        holdout["status"] = "ok";
        holdout["selection_mode"] = "certification_baseline";
        holdout["thesis"] = thesis_meta;
        for (const auto& item : cadence_meta.items()) {
          holdout[item.key()] = item.value();
        }
      } else {
        holdout = cadence_meta;
        holdout["status"] = "blocked_by_missing_selection";
        holdout["thesis"] = thesis_meta;
      }

      std::filesystem::path output_dir = results_root / thesis_name / cadence_id;
      std::filesystem::create_directories(output_dir);
      write_json(output_dir / "selection_summary.json", certification);
      write_json(output_dir / "quick_summary.json", quick);
      write_json(output_dir / "mega_summary.json", mega);
      write_json(output_dir / "walk_forward_results.json", holdout);

      json pair = {
          {"thesis", thesis_meta},
          {"cadence", cadence_meta},
          {"quick", quick},
          {"mega", mega},
          {"certification", certification},
          {"holdout", holdout},
          {"output_dir", output_dir.string()},
          {"profile_set", args.profile_set},
      };
      pairs.push_back(pair);
    }
  }

  json winner = nullptr;
  double best_sharpe = -1e9;
  for (const auto& pair : pairs) {
    const auto& cert = pair.value("certification", json::object());
    const auto& holdout = pair.value("holdout", json::object());
    if (cert.value("selection_status", "") != "selected") {
      continue;
    }
    if (!holdout.contains("phase4_gate")) continue;
    const auto& phase4 = holdout["phase4_gate"];
    if (!phase4.value("phase4_eligible", false)) {
      continue;
    }
    if (!phase4.contains("base_main_net_sharpe")) continue;
    double sharpe = phase4["base_main_net_sharpe"].get<double>();
    if (sharpe > best_sharpe) {
      best_sharpe = sharpe;
      winner = pair;
    }
  }

  json summary = {
      {"authoritative_status", winner.is_null() ? "no_validated_winner" : "validated_winner_found"},
      {"authoritative_validation_model", "legacy_entry_exit_costs"},
      {"pairs", pairs},
      {"winner", winner},
      {"rebalance_logic", kRebalanceLogic},
      {"rebalance_logic_label", kRebalanceLogicLabel},
      {"research_cycle", "rebalance_cadence_compare"},
      {"profile_set", args.profile_set},
      {"results_root", results_root.string()},
  };

  std::filesystem::path base_results_root = results_root.parent_path();
  std::filesystem::path frozen_manifest = base_results_root / "forward_monitor" / "frozen_strategy_manifest.json";
  if (std::filesystem::exists(frozen_manifest)) {
    summary["preserved_frozen_manifest"] = frozen_manifest.string();
  }

  write_json(summary_dir / "cadence_comparison.json", summary);
  return 0;
}
