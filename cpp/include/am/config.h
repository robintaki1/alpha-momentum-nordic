#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace am {

struct ProfileSettings {
  std::vector<int> lookbacks;
  std::vector<int> skips;
  std::vector<int> top_ns;
  int bootstrap_resamples = 0;
  int cross_sectional_shuffle_runs = 0;
  int block_shuffled_null_runs = 0;
};

struct CadenceSettings {
  std::string label;
  std::string schedule_type;
  int periods_per_year = 0;
  int canonical_offset_id = 0;
  std::vector<int> offset_ids;
  std::string offset_label_prefix;
};

struct ThesisSettings {
  std::string label;
  std::vector<std::string> excluded_countries;
  std::string scope_note;
};

struct RollingFold {
  std::string fold_id;
  std::string train_start;
  std::string train_end;
  std::string validate_start;
  std::string validate_end;
};

struct Config {
  std::unordered_map<std::string, std::unordered_map<std::string, ProfileSettings>> profile_sets;
  std::unordered_map<std::string, ThesisSettings> theses;
  std::vector<std::string> default_cadence_compare_cadences;
  std::unordered_map<std::string, CadenceSettings> cadence_settings;
  std::vector<RollingFold> rolling_folds;

  std::string insample_end;
  std::string oos_start;
  std::string oos_end;
  double oos_sharpe_min = 0.3;

  double sim_capital_sek = 250000.0;
  double max_order_fraction_of_60d_median_daily_value = 0.01;
  double brokerage_min_sek = 1.0;
  double brokerage_bps = 9.0;
  std::vector<std::pair<double, double>> spread_bps_buckets;
  std::vector<std::pair<double, double>> participation_bps_buckets;
  double low_price_threshold_sek = 50.0;
  double low_price_addon_bps = 10.0;
  double next_open_impact_multiplier = 1.25;
  double next_close_impact_multiplier = 1.0;
  std::unordered_map<std::string, double> fx_friction_scenarios_bps;
  double delist_fallback_haircut = -0.30;

  int mega_wf_passes_required = 4;
  int bootstrap_block_length_months = 6;
  int bootstrap_resamples = 2000;
  double pbo_threshold_max = 0.30;
  int pbo_target_slice_count = 10;
  int pbo_min_slice_length_months = 6;
  int cross_sectional_shuffle_runs = 500;
  int block_shuffled_null_runs = 200;
  double negative_control_pass_rate_max = 0.05;
  int negative_control_bootstrap_resamples = 250;
  std::string primary_selection_cost_model = "tiered_v1";
  std::string primary_passive_benchmark_id = "XACT_NORDEN";
  std::string secondary_opportunity_cost_benchmark_id = "VWRL_ALLWORLD";
};

Config load_config(const std::filesystem::path& path);

}  // namespace am
