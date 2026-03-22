#pragma once

#include "am/config.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace am {

struct WindowSimulation {
  std::vector<std::string> months;
  std::vector<double> monthly_returns;
  std::vector<double> primary_benchmark_returns;
  std::vector<double> secondary_benchmark_returns;
};

struct PositionSample {
  double score = 0.0;
  double next_return = 0.0;
};

struct WindowSnapshot {
  std::string signal_month;
  std::string holding_month;
  std::string anchor_trade_date;
  std::string next_execution_date;
  std::vector<std::string> selected_security_ids;
  std::vector<double> selected_scores;
  int eligible_count = 0;
  int selected_count = 0;
  std::unordered_map<std::string, double> eligible_by_country;
  std::unordered_map<std::string, double> selected_by_country;
  std::unordered_map<std::string, double> weight_by_country;
};

struct SimulationParams {
  int lookback = 12;
  int skip = 1;
  int top_n = 5;
};

class ResearchDataset {
 public:
  explicit ResearchDataset(const std::string& data_dir, const Config& config);

  WindowSimulation simulate_window(
      const SimulationParams& params,
      const std::string& universe_variant,
      const std::string& execution_model,
      const std::string& fx_scenario,
      const std::string& start_month,
      const std::string& end_month,
      const std::vector<std::string>& excluded_countries = {},
      std::optional<int> shuffled_selection_seed = std::nullopt) const;

  WindowSnapshot selection_snapshot(
      const SimulationParams& params,
      const std::string& signal_month,
      const std::string& universe_variant,
      const std::string& execution_model,
      const std::string& fx_scenario,
      const std::vector<std::string>& excluded_countries = {}) const;

  std::vector<std::vector<PositionSample>> negative_control_months(
      const SimulationParams& params,
      const std::string& start_month,
      const std::string& end_month,
      const std::vector<std::string>& excluded_countries = {}) const;

  bool variant_has_any(
      const std::string& universe_variant,
      const std::vector<std::string>& excluded_countries = {}) const;

  int month_index_for(const std::string& month) const { return month_index(month); }
  int64_t security_count() const { return security_count_; }
  int64_t month_count() const { return month_count_; }
  const std::vector<std::string>& security_ids() const { return security_ids_; }
  const std::vector<std::string>& security_country() const { return security_country_; }
  const std::vector<uint8_t>& eligible_full_nordics_mask() const { return eligible_full_nordics_; }
  const std::vector<uint8_t>& eligible_se_only_mask() const { return eligible_se_only_; }
  const std::vector<uint8_t>& asof_matches_anchor_mask() const { return asof_matches_anchor_; }
  const std::vector<double>& median_daily_value_60d_sek() const { return median_daily_value_60d_sek_; }
  const std::vector<double>& close_raw_sek() const { return close_raw_sek_; }
  const std::vector<uint8_t>& capacity_mask(int top_n) const;
  const std::vector<double>& cost_fractions(
      int top_n,
      const std::string& execution_model,
      const std::string& fx_scenario) const;

  const std::vector<std::string>& signal_months() const { return signal_months_; }

 private:
  struct DailySeries {
    std::vector<int64_t> dates;
    std::vector<double> adj_open_sek;
    std::vector<double> adj_close_sek;
  };

  int64_t to_timestamp_ns(const std::string& date_str) const;
  std::string month_from_timestamp(int64_t timestamp_ns) const;

  void load_universe();
  void load_daily_prices();
  void load_benchmarks();
  void build_monthly_panel();
  void precompute_scores();
  void precompute_rank_orders();
  void precompute_capacity_masks();
  void precompute_cost_fractions();
  void precompute_benchmark_returns();

  int month_index(const std::string& month) const;
  std::vector<int> window_signal_indices(const std::string& start_month, const std::string& end_month) const;

  std::optional<size_t> lower_bound_index(const DailySeries& series, int64_t target) const;
  std::optional<size_t> upper_bound_index(const DailySeries& series, int64_t target) const;

  bool has_cashout(int security_index, int64_t start_date, int64_t end_date) const;
  std::unordered_map<std::string, double> counts_by_country(
      const std::vector<int>& security_indices, std::optional<double> denom) const;

  const Config& config_;
  std::string data_dir_;

  std::vector<std::string> signal_months_;
  std::vector<std::string> holding_months_;
  std::unordered_map<std::string, int> month_lookup_;

  std::vector<std::string> security_ids_;
  std::unordered_map<std::string, int> security_lookup_;
  std::vector<std::string> security_country_;
  std::vector<std::string> security_currency_;
  std::vector<bool> security_is_non_sek_;

  int64_t month_count_ = 0;
  int64_t security_count_ = 0;

  std::vector<double> anchor_adj_close_local_;
  std::vector<uint8_t> asof_matches_anchor_;
  std::vector<uint8_t> eligible_full_nordics_;
  std::vector<uint8_t> eligible_se_only_;
  std::vector<double> close_raw_sek_;
  std::vector<double> median_daily_value_60d_sek_;
  std::vector<int64_t> anchor_trade_date_ns_;
  std::vector<int64_t> next_execution_date_ns_;
  std::vector<int64_t> scheduled_exit_date_ns_;
  std::vector<uint8_t> entry_available_next_open_;
  std::vector<uint8_t> entry_available_next_close_;
  std::vector<double> gross_return_next_open_;
  std::vector<double> gross_return_next_close_;

  std::unordered_map<int, std::vector<uint8_t>> capacity_masks_;
  std::unordered_map<long long, std::vector<double>> cost_fractions_;

  std::unordered_map<long long, std::vector<double>> scores_;
  std::unordered_map<long long, std::vector<std::vector<int>>> rank_orders_;

  std::vector<DailySeries> daily_by_security_index_;
  std::vector<std::vector<int64_t>> cashout_dates_by_security_;
  std::unordered_map<std::string, std::unordered_map<std::string, double>> benchmark_monthly_returns_;
};

}  // namespace am
