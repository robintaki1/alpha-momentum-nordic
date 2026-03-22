#include "am/config.h"

#include <fstream>
#include <stdexcept>

namespace am {

namespace {

template <typename T>
T get_required(const nlohmann::json& obj, const char* key) {
  if (!obj.contains(key)) {
    throw std::runtime_error(std::string("Missing config key: ") + key);
  }
  return obj.at(key).get<T>();
}

ProfileSettings parse_profile(const nlohmann::json& obj) {
  ProfileSettings profile;
  profile.lookbacks = obj.at("lookbacks").get<std::vector<int>>();
  profile.skips = obj.at("skips").get<std::vector<int>>();
  profile.top_ns = obj.at("top_ns").get<std::vector<int>>();
  profile.bootstrap_resamples = obj.at("bootstrap_resamples").get<int>();
  profile.cross_sectional_shuffle_runs = obj.at("cross_sectional_shuffle_runs").get<int>();
  profile.block_shuffled_null_runs = obj.at("block_shuffled_null_runs").get<int>();
  return profile;
}

}  // namespace

Config load_config(const std::filesystem::path& path) {
  auto payload = nlohmann::json::parse(std::ifstream(path));
  Config config;

  const auto& profile_sets = payload.at("research_profile_sets");
  for (auto it = profile_sets.begin(); it != profile_sets.end(); ++it) {
    std::unordered_map<std::string, ProfileSettings> profiles;
    for (auto pit = it.value().begin(); pit != it.value().end(); ++pit) {
      profiles.emplace(pit.key(), parse_profile(pit.value()));
    }
    config.profile_sets.emplace(it.key(), profiles);
  }

  const auto& theses = payload.at("research_thesis_settings");
  for (auto it = theses.begin(); it != theses.end(); ++it) {
    ThesisSettings thesis;
    thesis.label = it.value().at("label").get<std::string>();
    thesis.excluded_countries = it.value().at("excluded_countries").get<std::vector<std::string>>();
    thesis.scope_note = it.value().at("scope_note").get<std::string>();
    config.theses.emplace(it.key(), thesis);
  }

  config.default_cadence_compare_cadences =
      payload.at("default_cadence_compare_cadences").get<std::vector<std::string>>();

  const auto& cadence = payload.at("cadence_compare_settings");
  for (auto it = cadence.begin(); it != cadence.end(); ++it) {
    CadenceSettings setting;
    setting.label = it.value().at("label").get<std::string>();
    setting.schedule_type = it.value().at("schedule_type").get<std::string>();
    setting.periods_per_year = it.value().at("periods_per_year").get<int>();
    setting.canonical_offset_id = it.value().at("canonical_offset_id").get<int>();
    setting.offset_ids = it.value().at("offset_ids").get<std::vector<int>>();
    setting.offset_label_prefix = it.value().at("offset_label_prefix").get<std::string>();
    config.cadence_settings.emplace(it.key(), setting);
  }

  for (const auto& fold : payload.at("rolling_origin_folds")) {
    config.rolling_folds.push_back(
        {
            fold.at("fold_id").get<std::string>(),
            fold.at("train_start").get<std::string>(),
            fold.at("train_end").get<std::string>(),
            fold.at("validate_start").get<std::string>(),
            fold.at("validate_end").get<std::string>(),
        }
    );
  }

  config.insample_end = get_required<std::string>(payload, "insample_end");
  config.oos_start = get_required<std::string>(payload, "oos_start");
  config.oos_end = get_required<std::string>(payload, "oos_end");
  config.oos_sharpe_min = get_required<double>(payload, "oos_sharpe_min");
  config.sim_capital_sek = get_required<double>(payload, "sim_capital_sek");
  config.max_order_fraction_of_60d_median_daily_value =
      get_required<double>(payload, "max_order_fraction_of_60d_median_daily_value");
  config.brokerage_min_sek = get_required<double>(payload, "brokerage_min_sek");
  config.brokerage_bps = get_required<double>(payload, "brokerage_bps");
  config.spread_bps_buckets = payload.at("spread_bps_buckets").get<std::vector<std::pair<double, double>>>();
  config.participation_bps_buckets =
      payload.at("participation_bps_buckets").get<std::vector<std::pair<double, double>>>();
  config.low_price_threshold_sek = get_required<double>(payload, "low_price_threshold_sek");
  config.low_price_addon_bps = get_required<double>(payload, "low_price_addon_bps");
  config.next_open_impact_multiplier = get_required<double>(payload, "next_open_impact_multiplier");
  config.next_close_impact_multiplier = get_required<double>(payload, "next_close_impact_multiplier");
  config.fx_friction_scenarios_bps =
      payload.at("fx_friction_scenarios_bps").get<std::unordered_map<std::string, double>>();
  config.delist_fallback_haircut = get_required<double>(payload, "delist_fallback_haircut");
  config.mega_wf_passes_required = get_required<int>(payload, "mega_wf_passes_required");
  config.bootstrap_block_length_months = get_required<int>(payload, "bootstrap_block_length_months");
  config.bootstrap_resamples = get_required<int>(payload, "bootstrap_resamples");
  config.pbo_threshold_max = get_required<double>(payload, "pbo_threshold_max");
  config.pbo_target_slice_count = get_required<int>(payload, "pbo_target_slice_count");
  config.pbo_min_slice_length_months = get_required<int>(payload, "pbo_min_slice_length_months");
  config.cross_sectional_shuffle_runs = get_required<int>(payload, "cross_sectional_shuffle_runs");
  config.block_shuffled_null_runs = get_required<int>(payload, "block_shuffled_null_runs");
  config.negative_control_pass_rate_max = get_required<double>(payload, "negative_control_pass_rate_max");
  config.negative_control_bootstrap_resamples = get_required<int>(payload, "negative_control_bootstrap_resamples");
  config.primary_selection_cost_model = get_required<std::string>(payload, "primary_selection_cost_model");
  config.primary_passive_benchmark_id = get_required<std::string>(payload, "primary_passive_benchmark_id");
  config.secondary_opportunity_cost_benchmark_id =
      get_required<std::string>(payload, "secondary_opportunity_cost_benchmark_id");

  return config;
}

}  // namespace am
