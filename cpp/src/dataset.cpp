#include "am/dataset.h"

#include "am/core.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>

namespace am {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <typename ArrayType>
std::shared_ptr<ArrayType> combine_chunks(const std::shared_ptr<arrow::ChunkedArray>& chunked) {
  auto combined = arrow::Concatenate(chunked->chunks(), arrow::default_memory_pool()).ValueOrDie();
  return std::static_pointer_cast<ArrayType>(combined);
}

int64_t to_timestamp_ns(const std::shared_ptr<arrow::Array>& array, int64_t index) {
  if (!array || array->IsNull(index)) {
    return 0;
  }
  if (array->type_id() == arrow::Type::TIMESTAMP) {
    auto ts = std::static_pointer_cast<arrow::TimestampArray>(array);
    return ts->Value(index);
  }
  if (array->type_id() == arrow::Type::DATE32) {
    auto dt = std::static_pointer_cast<arrow::Date32Array>(array);
    return static_cast<int64_t>(dt->Value(index)) * 24LL * 60LL * 60LL * 1000000000LL;
  }
  return 0;
}

std::string to_string_value(const std::shared_ptr<arrow::Array>& array, int64_t index) {
  if (!array || array->IsNull(index)) {
    return "";
  }
  auto str_array = std::static_pointer_cast<arrow::StringArray>(array);
  return std::string(str_array->GetView(index));
}

double to_double_value(const std::shared_ptr<arrow::Array>& array, int64_t index) {
  if (!array || array->IsNull(index)) {
    return kNaN;
  }
  switch (array->type_id()) {
    case arrow::Type::DOUBLE:
      return std::static_pointer_cast<arrow::DoubleArray>(array)->Value(index);
    case arrow::Type::FLOAT:
      return static_cast<double>(std::static_pointer_cast<arrow::FloatArray>(array)->Value(index));
    case arrow::Type::INT64:
      return static_cast<double>(std::static_pointer_cast<arrow::Int64Array>(array)->Value(index));
    case arrow::Type::INT32:
      return static_cast<double>(std::static_pointer_cast<arrow::Int32Array>(array)->Value(index));
    default:
      return kNaN;
  }
}

bool to_bool_value(const std::shared_ptr<arrow::Array>& array, int64_t index) {
  if (!array || array->IsNull(index)) {
    return false;
  }
  return std::static_pointer_cast<arrow::BooleanArray>(array)->Value(index);
}

int month_ordinal(const std::string& month) {
  if (month.size() < 7) {
    return 0;
  }
  int year = std::stoi(month.substr(0, 4));
  int mon = std::stoi(month.substr(5, 2));
  return year * 12 + (mon - 1);
}

long long score_key(int lookback, int skip) {
  return (static_cast<long long>(lookback) << 32) | static_cast<unsigned int>(skip);
}

long long cost_key(int top_n, int exec_id, int fx_id) {
  return (static_cast<long long>(top_n) << 32) | (exec_id << 16) | fx_id;
}

}  // namespace

ResearchDataset::ResearchDataset(const std::string& data_dir, const Config& config)
    : config_(config), data_dir_(data_dir) {
  load_universe();
  load_daily_prices();
  load_benchmarks();
  build_monthly_panel();
  precompute_scores();
  precompute_rank_orders();
  precompute_capacity_masks();
  precompute_cost_fractions();
  precompute_benchmark_returns();
}

int ResearchDataset::month_index(const std::string& month) const {
  auto it = month_lookup_.find(month);
  return it == month_lookup_.end() ? -1 : it->second;
}

int64_t ResearchDataset::to_timestamp_ns(const std::string& date_str) const {
  if (date_str.size() < 10) {
    return 0;
  }
  int year = std::stoi(date_str.substr(0, 4));
  int month = std::stoi(date_str.substr(5, 2));
  int day = std::stoi(date_str.substr(8, 2));
  std::chrono::year_month_day ymd{std::chrono::year{year},
                                  std::chrono::month{static_cast<unsigned>(month)},
                                  std::chrono::day{static_cast<unsigned>(day)}};
  auto sys_days = std::chrono::sys_days{ymd};
  auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(sys_days).time_since_epoch().count();
  return static_cast<int64_t>(ns);
}

std::string ResearchDataset::month_from_timestamp(int64_t timestamp_ns) const {
  if (timestamp_ns == 0) {
    return "";
  }
  auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
      std::chrono::nanoseconds(timestamp_ns));
  auto days = std::chrono::floor<std::chrono::days>(tp);
  std::chrono::year_month_day ymd(days);
  int year = static_cast<int>(ymd.year());
  unsigned month = static_cast<unsigned>(ymd.month());
  char buffer[8];
  std::snprintf(buffer, sizeof(buffer), "%04d-%02u", year, month);
  return std::string(buffer);
}

void ResearchDataset::load_universe() {
  auto table_result = read_parquet(std::filesystem::path(data_dir_) / "universe_pti.parquet");
  if (!table_result.ok()) {
    throw std::runtime_error(table_result.status().ToString());
  }
  auto table = *table_result;

  auto rebalance_col = combine_chunks<arrow::StringArray>(table->GetColumnByName("rebalance_month"));
  auto anchor_col = combine_chunks<arrow::TimestampArray>(table->GetColumnByName("anchor_trade_date"));
  auto next_exec_col = combine_chunks<arrow::TimestampArray>(table->GetColumnByName("next_execution_date"));
  auto asof_col = combine_chunks<arrow::TimestampArray>(table->GetColumnByName("asof_trade_date"));
  auto security_col = combine_chunks<arrow::StringArray>(table->GetColumnByName("security_id"));
  auto exchange_col = combine_chunks<arrow::StringArray>(table->GetColumnByName("exchange_group"));
  auto country_col = combine_chunks<arrow::StringArray>(table->GetColumnByName("country_code"));
  auto currency_col = combine_chunks<arrow::StringArray>(table->GetColumnByName("currency"));
  auto eligible_full = combine_chunks<arrow::BooleanArray>(table->GetColumnByName("is_eligible_full_nordics"));
  auto eligible_se = combine_chunks<arrow::BooleanArray>(table->GetColumnByName("is_eligible_se_only"));
  auto close_raw_sek = combine_chunks<arrow::DoubleArray>(table->GetColumnByName("close_raw_sek"));
  auto median_value = combine_chunks<arrow::DoubleArray>(table->GetColumnByName("median_daily_value_60d_sek"));

  struct Row {
    std::string month;
    std::string security;
    std::string exchange_group;
    std::string country;
    std::string currency;
    int64_t anchor_date;
    int64_t asof_date;
    int64_t next_exec;
    bool eligible_full;
    bool eligible_se;
    double close_raw_sek;
    double median_value;
  };

  std::vector<Row> rows;
  rows.reserve(table->num_rows());
  std::unordered_set<std::string> month_set;
  for (int64_t i = 0; i < table->num_rows(); ++i) {
    Row row{
        rebalance_col->GetString(i),
        security_col->GetString(i),
        exchange_col->IsNull(i) ? "" : exchange_col->GetString(i),
        country_col->IsNull(i) ? "" : country_col->GetString(i),
        currency_col->IsNull(i) ? "" : currency_col->GetString(i),
        anchor_col->Value(i),
        asof_col->Value(i),
        next_exec_col->Value(i),
        eligible_full->Value(i),
        eligible_se->Value(i),
        close_raw_sek->Value(i),
        median_value->Value(i),
    };
    rows.push_back(row);
    month_set.insert(row.month);
    if (security_lookup_.find(row.security) == security_lookup_.end()) {
      int index = static_cast<int>(security_ids_.size());
      security_lookup_[row.security] = index;
      security_ids_.push_back(row.security);
      security_country_.push_back(row.country);
      security_currency_.push_back(row.currency);
    }
  }

  signal_months_.assign(month_set.begin(), month_set.end());
  std::sort(signal_months_.begin(), signal_months_.end(),
            [](const std::string& a, const std::string& b) { return month_ordinal(a) < month_ordinal(b); });
  for (size_t idx = 0; idx < signal_months_.size(); ++idx) {
    month_lookup_[signal_months_[idx]] = static_cast<int>(idx);
  }

  month_count_ = static_cast<int64_t>(signal_months_.size());
  security_count_ = static_cast<int64_t>(security_ids_.size());
  holding_months_.assign(signal_months_.size(), "");

  anchor_adj_close_local_.assign(month_count_ * security_count_, kNaN);
  asof_matches_anchor_.assign(month_count_ * security_count_, 0);
  eligible_full_nordics_.assign(month_count_ * security_count_, 0);
  eligible_se_only_.assign(month_count_ * security_count_, 0);
  close_raw_sek_.assign(month_count_ * security_count_, kNaN);
  median_daily_value_60d_sek_.assign(month_count_ * security_count_, kNaN);
  anchor_trade_date_ns_.assign(month_count_ * security_count_, 0);
  next_execution_date_ns_.assign(month_count_ * security_count_, 0);
  scheduled_exit_date_ns_.assign(month_count_ * security_count_, 0);

  std::unordered_map<std::string, int64_t> month_next_exec;
  std::unordered_map<std::string, std::vector<std::pair<int, int64_t>>> exchange_schedule;
  for (const auto& row : rows) {
    int month_idx = month_lookup_[row.month];
    int sec_idx = security_lookup_[row.security];
    int64_t index = month_idx * security_count_ + sec_idx;
    asof_matches_anchor_[index] = static_cast<uint8_t>(row.asof_date == row.anchor_date);
    eligible_full_nordics_[index] = static_cast<uint8_t>(row.eligible_full);
    eligible_se_only_[index] = static_cast<uint8_t>(row.eligible_se);
    close_raw_sek_[index] = row.close_raw_sek;
    median_daily_value_60d_sek_[index] = row.median_value;
    anchor_trade_date_ns_[index] = row.anchor_date;
    next_execution_date_ns_[index] = row.next_exec;

    if (month_next_exec.find(row.month) == month_next_exec.end() && row.next_exec != 0) {
      month_next_exec[row.month] = row.next_exec;
    }
    if (!row.exchange_group.empty()) {
      exchange_schedule[row.exchange_group].push_back({month_idx, row.next_exec});
    }
  }

  for (const auto& entry : month_next_exec) {
    const auto& month = entry.first;
    auto month_idx = month_lookup_[month];
    holding_months_[month_idx] = month_from_timestamp(entry.second);
  }

  std::unordered_map<std::string, std::vector<int64_t>> scheduled_exit;
  for (auto& [exchange, items] : exchange_schedule) {
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<int64_t> schedule(month_count_, 0);
    for (size_t i = 0; i + 1 < items.size(); ++i) {
      schedule[items[i].first] = items[i + 1].second;
    }
    scheduled_exit.emplace(exchange, std::move(schedule));
  }

  for (const auto& row : rows) {
    auto schedule_it = scheduled_exit.find(row.exchange_group);
    if (schedule_it == scheduled_exit.end()) {
      continue;
    }
    int month_idx = month_lookup_[row.month];
    int sec_idx = security_lookup_[row.security];
    int64_t index = month_idx * security_count_ + sec_idx;
    scheduled_exit_date_ns_[index] = schedule_it->second[month_idx];
  }
}

void ResearchDataset::load_daily_prices() {
  auto raw_result = read_parquet(std::filesystem::path(data_dir_) / "prices_raw_daily.parquet");
  auto adj_result = read_parquet(std::filesystem::path(data_dir_) / "prices_adjusted_daily.parquet");
  auto fx_result = read_parquet(std::filesystem::path(data_dir_) / "riksbank_fx_daily.parquet");
  auto ca_result = read_parquet(std::filesystem::path(data_dir_) / "corporate_actions.parquet");
  if (!raw_result.ok() || !adj_result.ok() || !fx_result.ok() || !ca_result.ok()) {
    throw std::runtime_error("Failed to load daily price artifacts.");
  }

  auto raw = *raw_result;
  auto adj = *adj_result;
  auto fx = *fx_result;
  auto ca = *ca_result;

  // FX map
  auto fx_currency = combine_chunks<arrow::StringArray>(fx->GetColumnByName("currency"));
  auto fx_date = combine_chunks<arrow::TimestampArray>(fx->GetColumnByName("date"));
  auto fx_value = combine_chunks<arrow::DoubleArray>(fx->GetColumnByName("sek_per_ccy"));
  std::unordered_map<std::string, int> currency_ids;
  int next_currency_id = 0;
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
  std::unordered_map<FxKey, double, FxHash> fx_map;
  fx_map.reserve(fx->num_rows());
  for (int64_t i = 0; i < fx->num_rows(); ++i) {
    std::string currency = fx_currency->GetString(i);
    auto it = currency_ids.find(currency);
    if (it == currency_ids.end()) {
      it = currency_ids.emplace(currency, next_currency_id++).first;
    }
    fx_map[{it->second, fx_date->Value(i)}] = fx_value->Value(i);
  }

  // Adjusted prices map
  auto adj_security = combine_chunks<arrow::StringArray>(adj->GetColumnByName("security_id"));
  auto adj_date = combine_chunks<arrow::TimestampArray>(adj->GetColumnByName("date"));
  auto adj_factor = combine_chunks<arrow::DoubleArray>(adj->GetColumnByName("adj_factor"));
  auto adj_close = combine_chunks<arrow::DoubleArray>(adj->GetColumnByName("adj_close"));
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
  std::unordered_map<SecDateKey, double, SecDateHash> adj_factor_map;
  std::unordered_map<SecDateKey, double, SecDateHash> adj_close_map;
  adj_factor_map.reserve(adj->num_rows());
  adj_close_map.reserve(adj->num_rows());
  for (int64_t i = 0; i < adj->num_rows(); ++i) {
    auto security = adj_security->GetString(i);
    auto sec_it = security_lookup_.find(security);
    if (sec_it == security_lookup_.end()) {
      continue;
    }
    SecDateKey key{sec_it->second, adj_date->Value(i)};
    adj_factor_map[key] = adj_factor->Value(i);
    adj_close_map[key] = adj_close->Value(i);
  }

  for (int64_t month_idx = 0; month_idx < month_count_; ++month_idx) {
    for (int64_t sec_idx = 0; sec_idx < security_count_; ++sec_idx) {
      int64_t index = month_idx * security_count_ + sec_idx;
      int64_t anchor_date = anchor_trade_date_ns_[index];
      if (anchor_date == 0) {
        continue;
      }
      SecDateKey key{static_cast<int>(sec_idx), anchor_date};
      auto it = adj_close_map.find(key);
      if (it != adj_close_map.end()) {
        anchor_adj_close_local_[index] = it->second;
      }
    }
  }

  daily_by_security_index_.assign(security_count_, {});
  cashout_dates_by_security_.assign(security_count_, {});

  auto raw_security = combine_chunks<arrow::StringArray>(raw->GetColumnByName("security_id"));
  auto raw_date = combine_chunks<arrow::TimestampArray>(raw->GetColumnByName("date"));
  auto raw_open = combine_chunks<arrow::DoubleArray>(raw->GetColumnByName("open_raw"));
  auto raw_close = combine_chunks<arrow::DoubleArray>(raw->GetColumnByName("close_raw"));
  auto raw_currency = combine_chunks<arrow::StringArray>(raw->GetColumnByName("currency"));
  for (int64_t i = 0; i < raw->num_rows(); ++i) {
    auto security = raw_security->GetString(i);
    auto sec_it = security_lookup_.find(security);
    if (sec_it == security_lookup_.end()) {
      continue;
    }
    int sec_idx = sec_it->second;
    SecDateKey key{sec_idx, raw_date->Value(i)};
    auto factor_it = adj_factor_map.find(key);
    if (factor_it == adj_factor_map.end()) {
      continue;
    }
    std::string currency = raw_currency->GetString(i);
    auto currency_it = currency_ids.find(currency);
    if (currency_it == currency_ids.end()) {
      continue;
    }
    auto fx_it = fx_map.find({currency_it->second, raw_date->Value(i)});
    if (fx_it == fx_map.end()) {
      continue;
    }
    double factor = factor_it->second;
    double fx_rate = fx_it->second;
    double adj_open = raw_open->Value(i) * factor * fx_rate;
    double adj_close_value = raw_close->Value(i) * factor * fx_rate;
    auto& series = daily_by_security_index_[sec_idx];
    series.dates.push_back(raw_date->Value(i));
    series.adj_open_sek.push_back(adj_open);
    series.adj_close_sek.push_back(adj_close_value);
  }

  for (auto& series : daily_by_security_index_) {
    std::vector<size_t> order(series.dates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return series.dates[a] < series.dates[b]; });
    std::vector<int64_t> dates;
    std::vector<double> open;
    std::vector<double> close;
    dates.reserve(order.size());
    open.reserve(order.size());
    close.reserve(order.size());
    for (auto idx : order) {
      dates.push_back(series.dates[idx]);
      open.push_back(series.adj_open_sek[idx]);
      close.push_back(series.adj_close_sek[idx]);
    }
    series.dates.swap(dates);
    series.adj_open_sek.swap(open);
    series.adj_close_sek.swap(close);
  }

  auto ca_security = combine_chunks<arrow::StringArray>(ca->GetColumnByName("security_id"));
  auto ca_date = combine_chunks<arrow::TimestampArray>(ca->GetColumnByName("event_date"));
  auto ca_action = combine_chunks<arrow::StringArray>(ca->GetColumnByName("action_type"));
  for (int64_t i = 0; i < ca->num_rows(); ++i) {
    std::string action = ca_action->GetString(i);
    if (action != "cash_merger" && action != "delisting_cashout") {
      continue;
    }
    auto sec_it = security_lookup_.find(ca_security->GetString(i));
    if (sec_it == security_lookup_.end()) {
      continue;
    }
    cashout_dates_by_security_[sec_it->second].push_back(ca_date->Value(i));
  }
  for (auto& dates : cashout_dates_by_security_) {
    std::sort(dates.begin(), dates.end());
  }

  // Cache security currency flags.
  security_is_non_sek_.assign(security_count_, false);
  for (int i = 0; i < security_count_; ++i) {
    security_is_non_sek_[i] = (security_currency_[i] != "SEK");
  }
}

void ResearchDataset::load_benchmarks() {}

void ResearchDataset::build_monthly_panel() {
  entry_available_next_open_.assign(month_count_ * security_count_, 0);
  entry_available_next_close_.assign(month_count_ * security_count_, 0);
  gross_return_next_open_.assign(month_count_ * security_count_, kNaN);
  gross_return_next_close_.assign(month_count_ * security_count_, kNaN);

  for (int64_t month_idx = 0; month_idx < month_count_; ++month_idx) {
    for (int64_t sec_idx = 0; sec_idx < security_count_; ++sec_idx) {
      int64_t index = month_idx * security_count_ + sec_idx;
      int64_t entry_target = next_execution_date_ns_[index];
      if (entry_target == 0) {
        continue;
      }
      const auto& series = daily_by_security_index_[sec_idx];
      auto entry_idx_opt = lower_bound_index(series, entry_target);
      if (!entry_idx_opt) {
        continue;
      }
      size_t entry_idx = *entry_idx_opt;
      int64_t entry_date = series.dates[entry_idx];
      double entry_open = series.adj_open_sek[entry_idx];
      double entry_close = series.adj_close_sek[entry_idx];
      int64_t exit_target = scheduled_exit_date_ns_[index];

      bool entry_open_ok = std::isfinite(entry_open) && entry_open > 0.0;
      bool entry_close_ok = std::isfinite(entry_close) && entry_close > 0.0;
      if (exit_target == 0 || entry_date <= exit_target) {
        if (entry_open_ok) {
          entry_available_next_open_[index] = 1;
        }
        if (entry_close_ok) {
          entry_available_next_close_[index] = 1;
        }
      }

      bool has_cashout_flag = has_cashout(static_cast<int>(sec_idx), entry_target, exit_target);
      double fallback_multiplier = has_cashout_flag ? 1.0 : (1.0 + config_.delist_fallback_haircut);

      auto exit_forward_idx = exit_target == 0 ? std::nullopt : lower_bound_index(series, exit_target);
      auto exit_backward_idx = exit_target == 0 ? std::nullopt : upper_bound_index(series, exit_target);

      double exit_open_forward = kNaN;
      double exit_close_forward = kNaN;
      if (exit_forward_idx) {
        exit_open_forward = series.adj_open_sek[*exit_forward_idx];
        exit_close_forward = series.adj_close_sek[*exit_forward_idx];
      }
      double exit_close_backward = kNaN;
      int64_t exit_back_date = 0;
      if (exit_backward_idx) {
        exit_close_backward = series.adj_close_sek[*exit_backward_idx];
        exit_back_date = series.dates[*exit_backward_idx];
      }
      bool usable_backward = exit_back_date != 0 && exit_back_date >= entry_date;

      auto resolve_return = [&](double entry_price, double exit_forward_price) -> double {
        if (!std::isfinite(entry_price) || entry_price <= 0.0) {
          return kNaN;
        }
        double exit_price = kNaN;
        if (std::isfinite(exit_forward_price) && exit_forward_price > 0.0) {
          exit_price = exit_forward_price;
        } else if (usable_backward && std::isfinite(exit_close_backward) && exit_close_backward > 0.0) {
          exit_price = exit_close_backward * fallback_multiplier;
        }
        if (std::isfinite(exit_price) && exit_price > 0.0) {
          return exit_price / entry_price - 1.0;
        }
        if (entry_open_ok || entry_close_ok) {
          return config_.delist_fallback_haircut;
        }
        return kNaN;
      };

      if (entry_open_ok) {
        gross_return_next_open_[index] = resolve_return(entry_open, exit_open_forward);
      }
      if (entry_close_ok) {
        gross_return_next_close_[index] = resolve_return(entry_close, exit_close_forward);
      }
    }
  }
}

std::optional<size_t> ResearchDataset::lower_bound_index(const DailySeries& series, int64_t target) const {
  if (series.dates.empty()) {
    return std::nullopt;
  }
  auto it = std::lower_bound(series.dates.begin(), series.dates.end(), target);
  if (it == series.dates.end()) {
    return std::nullopt;
  }
  return static_cast<size_t>(std::distance(series.dates.begin(), it));
}

std::optional<size_t> ResearchDataset::upper_bound_index(const DailySeries& series, int64_t target) const {
  if (series.dates.empty()) {
    return std::nullopt;
  }
  auto it = std::upper_bound(series.dates.begin(), series.dates.end(), target);
  if (it == series.dates.begin()) {
    return std::nullopt;
  }
  return static_cast<size_t>(std::distance(series.dates.begin(), it) - 1);
}

bool ResearchDataset::has_cashout(int security_index, int64_t start_date, int64_t end_date) const {
  if (start_date == 0 || end_date == 0) {
    return false;
  }
  const auto& dates = cashout_dates_by_security_[security_index];
  if (dates.empty()) {
    return false;
  }
  auto left = std::lower_bound(dates.begin(), dates.end(), start_date);
  auto right = std::upper_bound(dates.begin(), dates.end(), end_date);
  return right != left;
}

void ResearchDataset::precompute_scores() {
  // Build score matrices for each lookback/skip.
  std::unordered_set<int> lookbacks;
  std::unordered_set<int> skips;
  for (const auto& [set_name, profiles] : config_.profile_sets) {
    for (const auto& [profile_name, profile] : profiles) {
      lookbacks.insert(profile.lookbacks.begin(), profile.lookbacks.end());
      skips.insert(profile.skips.begin(), profile.skips.end());
    }
  }
  for (int lookback : lookbacks) {
    for (int skip : skips) {
      long long key = score_key(lookback, skip);
      std::vector<double> matrix(month_count_ * security_count_, kNaN);
      for (int64_t month_idx = 0; month_idx < month_count_; ++month_idx) {
        int64_t current_idx = month_idx - skip;
        int64_t previous_idx = month_idx - skip - lookback;
        if (current_idx < 0 || previous_idx < 0) {
          continue;
        }
        for (int64_t sec_idx = 0; sec_idx < security_count_; ++sec_idx) {
          double current = anchor_adj_close_local_[current_idx * security_count_ + sec_idx];
          double previous = anchor_adj_close_local_[previous_idx * security_count_ + sec_idx];
          if (!std::isfinite(current) || !std::isfinite(previous) || current <= 0.0 || previous <= 0.0) {
            continue;
          }
          matrix[month_idx * security_count_ + sec_idx] = current / previous - 1.0;
        }
      }
      scores_.emplace(key, std::move(matrix));
    }
  }
}

void ResearchDataset::precompute_rank_orders() {
  for (const auto& [key, matrix] : scores_) {
    std::vector<std::vector<int>> orders;
    orders.reserve(month_count_);
    for (int64_t month_idx = 0; month_idx < month_count_; ++month_idx) {
      std::vector<int> indices(security_count_);
      std::iota(indices.begin(), indices.end(), 0);
      std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        double score_a = matrix[month_idx * security_count_ + a];
        double score_b = matrix[month_idx * security_count_ + b];
        bool a_valid = std::isfinite(score_a);
        bool b_valid = std::isfinite(score_b);
        if (a_valid != b_valid) {
          return a_valid > b_valid;
        }
        if (!a_valid && !b_valid) {
          return a < b;
        }
        if (score_a == score_b) {
          return a < b;
        }
        return score_a > score_b;
      });
      orders.push_back(std::move(indices));
    }
    rank_orders_.emplace(key, std::move(orders));
  }
}

void ResearchDataset::precompute_capacity_masks() {
  std::unordered_set<int> top_ns;
  for (const auto& [set_name, profiles] : config_.profile_sets) {
    for (const auto& [profile_name, profile] : profiles) {
      top_ns.insert(profile.top_ns.begin(), profile.top_ns.end());
    }
  }
  for (int top_n : top_ns) {
    double order_notional = config_.sim_capital_sek / static_cast<double>(top_n);
    std::vector<uint8_t> mask(month_count_ * security_count_, 0);
    for (int64_t idx = 0; idx < month_count_ * security_count_; ++idx) {
      double median_value = median_daily_value_60d_sek_[idx];
      if (median_value <= 0.0 || !std::isfinite(median_value)) {
        continue;
      }
      double ratio = order_notional / median_value;
      if (ratio <= config_.max_order_fraction_of_60d_median_daily_value) {
        mask[idx] = 1;
      }
    }
    capacity_masks_.emplace(top_n, std::move(mask));
  }
}

void ResearchDataset::precompute_cost_fractions() {
  for (const auto& [top_n, mask] : capacity_masks_) {
    double order_notional = config_.sim_capital_sek / static_cast<double>(top_n);
    double brokerage_sek = std::max(config_.brokerage_min_sek, order_notional * (config_.brokerage_bps / 10000.0));
    double brokerage_bps = (brokerage_sek / order_notional) * 10000.0;
    for (const auto& exec : {std::string("next_open"), std::string("next_close")}) {
      double impact_multiplier = exec == "next_open" ? config_.next_open_impact_multiplier
                                                    : config_.next_close_impact_multiplier;
      int exec_id = exec == "next_open" ? 0 : 1;
      for (const auto& fx_pair : config_.fx_friction_scenarios_bps) {
        int fx_id = fx_pair.first == "low" ? 0 : (fx_pair.first == "base" ? 1 : 2);
        double fx_bps = fx_pair.second;
        std::vector<double> costs(month_count_ * security_count_, kNaN);
        for (int64_t idx = 0; idx < month_count_ * security_count_; ++idx) {
          if (!mask[idx]) {
            continue;
          }
          double median_value = median_daily_value_60d_sek_[idx];
          if (!std::isfinite(median_value) || median_value <= 0.0) {
            continue;
          }
          double spread_bps = config_.spread_bps_buckets.back().second;
          for (const auto& bucket : config_.spread_bps_buckets) {
            if (median_value >= bucket.first) {
              spread_bps = bucket.second;
              break;
            }
          }
          double participation_bps = 0.0;
          double ratio = order_notional / median_value;
          bool passes = ratio <= config_.max_order_fraction_of_60d_median_daily_value;
          if (!passes) {
            continue;
          }
          for (const auto& bucket : config_.participation_bps_buckets) {
            if (ratio <= bucket.first) {
              participation_bps = bucket.second;
              break;
            }
          }
          double impact = (spread_bps + participation_bps) * impact_multiplier;
          double low_price = close_raw_sek_[idx] < config_.low_price_threshold_sek ? config_.low_price_addon_bps : 0.0;
          double fx_cost = security_is_non_sek_[idx % security_count_] ? fx_bps : 0.0;
          double total_bps = brokerage_bps + impact + low_price + fx_cost;
          costs[idx] = (total_bps / 10000.0) / static_cast<double>(top_n);
        }
        cost_fractions_.emplace(cost_key(top_n, exec_id, fx_id), std::move(costs));
      }
    }
  }
}

void ResearchDataset::precompute_benchmark_returns() {
  auto bench_result = read_parquet(std::filesystem::path(data_dir_) / "benchmark_prices.parquet");
  auto fx_result = read_parquet(std::filesystem::path(data_dir_) / "riksbank_fx_daily.parquet");
  if (!bench_result.ok() || !fx_result.ok()) {
    spdlog::warn("Benchmark data unavailable.");
    return;
  }
  auto bench = *bench_result;
  auto fx = *fx_result;

  auto fx_currency = combine_chunks<arrow::StringArray>(fx->GetColumnByName("currency"));
  auto fx_date = combine_chunks<arrow::TimestampArray>(fx->GetColumnByName("date"));
  auto fx_value = combine_chunks<arrow::DoubleArray>(fx->GetColumnByName("sek_per_ccy"));
  struct FxKey {
    std::string currency;
    int64_t date;
    bool operator==(const FxKey& other) const { return currency == other.currency && date == other.date; }
  };
  struct FxHash {
    size_t operator()(const FxKey& key) const {
      return std::hash<std::string>()(key.currency) ^ std::hash<long long>()(key.date);
    }
  };
  std::unordered_map<FxKey, double, FxHash> fx_map;
  fx_map.reserve(fx->num_rows());
  for (int64_t i = 0; i < fx->num_rows(); ++i) {
    fx_map[{fx_currency->GetString(i), fx_date->Value(i)}] = fx_value->Value(i);
  }

  auto bench_id = combine_chunks<arrow::StringArray>(bench->GetColumnByName("benchmark_id"));
  auto bench_date = combine_chunks<arrow::TimestampArray>(bench->GetColumnByName("date"));
  auto bench_currency = combine_chunks<arrow::StringArray>(bench->GetColumnByName("currency"));
  auto bench_adj = combine_chunks<arrow::DoubleArray>(bench->GetColumnByName("adj_close"));

  struct BenchRow {
    int64_t date;
    double price_sek;
  };
  std::unordered_map<std::string, std::vector<BenchRow>> by_id;
  for (int64_t i = 0; i < bench->num_rows(); ++i) {
    std::string currency = bench_currency->GetString(i);
    auto fx_it = fx_map.find({currency, bench_date->Value(i)});
    if (fx_it == fx_map.end()) {
      continue;
    }
    double price_sek = bench_adj->Value(i) * fx_it->second;
    by_id[bench_id->GetString(i)].push_back({bench_date->Value(i), price_sek});
  }
  for (auto& [id, rows] : by_id) {
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.date < b.date; });
  }

  std::unordered_set<std::string> holding_months_set;
  for (const auto& month : holding_months_) {
    if (!month.empty()) {
      holding_months_set.insert(month);
    }
  }

  for (const auto& [id, rows] : by_id) {
    std::unordered_map<std::string, double> month_returns;
    for (const auto& month : holding_months_set) {
      std::string start_str = month + "-01";
      int64_t start_ts = to_timestamp_ns(start_str);
      // Next month
      int ord = month_ordinal(month) + 1;
      int next_year = ord / 12;
      int next_mon = ord % 12 + 1;
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "%04d-%02d", next_year, next_mon);
      std::string next_month = std::string(buffer);
      int64_t end_ts = to_timestamp_ns(next_month + "-01");

      auto it_start = std::lower_bound(rows.begin(), rows.end(), start_ts,
                                       [](const auto& row, int64_t value) { return row.date < value; });
      auto it_end = std::lower_bound(rows.begin(), rows.end(), end_ts,
                                     [](const auto& row, int64_t value) { return row.date < value; });
      if (it_start == rows.end() || it_end == rows.end()) {
        continue;
      }
      double entry_price = it_start->price_sek;
      double exit_price = it_end->price_sek;
      if (entry_price > 0.0 && exit_price > 0.0) {
        month_returns[month] = exit_price / entry_price - 1.0;
      }
    }
    benchmark_monthly_returns_[id] = std::move(month_returns);
  }
}

std::vector<int> ResearchDataset::window_signal_indices(const std::string& start_month, const std::string& end_month) const {
  int start = month_index(start_month);
  int end = month_index(end_month);
  std::vector<int> indices;
  if (start < 0 || end < 0) {
    return indices;
  }
  for (int idx = start; idx <= end; ++idx) {
    if (!holding_months_[idx].empty()) {
      indices.push_back(idx);
    }
  }
  return indices;
}

WindowSimulation ResearchDataset::simulate_window(
    const SimulationParams& params,
    const std::string& universe_variant,
    const std::string& execution_model,
    const std::string& fx_scenario,
    const std::string& start_month,
    const std::string& end_month,
    const std::vector<std::string>& excluded_countries,
    std::optional<int> shuffled_selection_seed) const {
  WindowSimulation simulation;
  auto indices = window_signal_indices(start_month, end_month);
  if (indices.empty()) {
    return simulation;
  }
  int exec_id = execution_model == "next_open" ? 0 : 1;
  int fx_id = fx_scenario == "low" ? 0 : (fx_scenario == "base" ? 1 : 2);
  auto score_it = scores_.find(score_key(params.lookback, params.skip));
  if (score_it == scores_.end()) {
    return simulation;
  }
  const auto& score_matrix = score_it->second;
  const auto& rank_orders = rank_orders_.at(score_key(params.lookback, params.skip));
  const auto& capacity_mask = capacity_masks_.at(params.top_n);
  const auto& cost_fraction = cost_fractions_.at(cost_key(params.top_n, exec_id, fx_id));

  std::vector<uint8_t> base_mask = (universe_variant == "SE-only") ? eligible_se_only_ : eligible_full_nordics_;
  if (universe_variant == "largest-third-by-market-cap") {
    base_mask.assign(month_count_ * security_count_, 0);
  }

  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());

  std::vector<int> previous_selection;
  std::mt19937 rng;
  if (shuffled_selection_seed) {
    rng.seed(*shuffled_selection_seed);
  }

  for (int month_idx : indices) {
    std::vector<int> eligible;
    eligible.reserve(params.top_n * 2);
    for (int sec_idx : rank_orders[month_idx]) {
      int64_t idx = month_idx * security_count_ + sec_idx;
      if (!base_mask[idx] || !asof_matches_anchor_[idx] || !capacity_mask[idx]) {
        continue;
      }
      double score = score_matrix[idx];
      if (!std::isfinite(score)) {
        continue;
      }
      if (excluded.find(security_country_[sec_idx]) != excluded.end()) {
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
      int64_t idx = month_idx * security_count_ + sec_idx;
      double ret = exec_id == 0 ? gross_return_next_open_[idx] : gross_return_next_close_[idx];
      if (std::isfinite(ret)) {
        gross_return += ret;
      }
    }
    gross_return = selected.empty() ? 0.0 : gross_return / static_cast<double>(params.top_n);
    double trade_cost = 0.0;
    for (int sec_idx : selected) {
      if (!prev_set.count(sec_idx)) {
        int64_t idx = month_idx * security_count_ + sec_idx;
        trade_cost += cost_fraction[idx];
      }
    }
    for (int sec_idx : previous_selection) {
      if (!current_set.count(sec_idx)) {
        int64_t idx = month_idx * security_count_ + sec_idx;
        trade_cost += cost_fraction[idx];
      }
    }
    double net_return = gross_return - trade_cost;
    simulation.months.push_back(holding_months_[month_idx]);
    simulation.monthly_returns.push_back(net_return);
    previous_selection = selected;
  }

  auto primary_it = benchmark_monthly_returns_.find(config_.primary_passive_benchmark_id);
  auto secondary_it = benchmark_monthly_returns_.find(config_.secondary_opportunity_cost_benchmark_id);
  bool primary_ok = primary_it != benchmark_monthly_returns_.end();
  bool secondary_ok = secondary_it != benchmark_monthly_returns_.end();
  for (const auto& month : simulation.months) {
    if (primary_ok) {
      auto it = primary_it->second.find(month);
      if (it != primary_it->second.end()) {
        simulation.primary_benchmark_returns.push_back(it->second);
      }
    }
    if (secondary_ok) {
      auto it = secondary_it->second.find(month);
      if (it != secondary_it->second.end()) {
        simulation.secondary_benchmark_returns.push_back(it->second);
      }
    }
  }

  return simulation;
}

WindowSnapshot ResearchDataset::selection_snapshot(
    const SimulationParams& params,
    const std::string& signal_month,
    const std::string& universe_variant,
    const std::string& execution_model,
    const std::string& fx_scenario,
    const std::vector<std::string>& excluded_countries) const {
  WindowSnapshot snapshot;
  int month_idx = month_index(signal_month);
  if (month_idx < 0) {
    return snapshot;
  }
  snapshot.signal_month = signal_month;
  snapshot.holding_month = holding_months_[month_idx];
  int exec_id = execution_model == "next_open" ? 0 : 1;
  int fx_id = fx_scenario == "low" ? 0 : (fx_scenario == "base" ? 1 : 2);
  auto score_it = scores_.find(score_key(params.lookback, params.skip));
  if (score_it == scores_.end()) {
    return snapshot;
  }
  const auto& score_matrix = score_it->second;
  const auto& rank_orders = rank_orders_.at(score_key(params.lookback, params.skip));
  const auto& capacity_mask = capacity_masks_.at(params.top_n);
  (void)fx_id;

  std::vector<uint8_t> base_mask = (universe_variant == "SE-only") ? eligible_se_only_ : eligible_full_nordics_;
  if (universe_variant == "largest-third-by-market-cap") {
    base_mask.assign(month_count_ * security_count_, 0);
  }

  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());

  std::vector<int> eligible;
  for (int sec_idx : rank_orders[month_idx]) {
    int64_t idx = month_idx * security_count_ + sec_idx;
    if (!base_mask[idx] || !asof_matches_anchor_[idx] || !capacity_mask[idx]) {
      continue;
    }
    double score = score_matrix[idx];
    if (!std::isfinite(score)) {
      continue;
    }
    if (excluded.find(security_country_[sec_idx]) != excluded.end()) {
      continue;
    }
    eligible.push_back(sec_idx);
    if (static_cast<int>(eligible.size()) >= params.top_n) {
      break;
    }
  }
  snapshot.eligible_count = static_cast<int>(eligible.size());
  snapshot.selected_count = static_cast<int>(eligible.size());
  for (int sec_idx : eligible) {
    snapshot.selected_security_ids.push_back(security_ids_[sec_idx]);
    snapshot.selected_scores.push_back(score_matrix[month_idx * security_count_ + sec_idx]);
  }
  snapshot.eligible_by_country = counts_by_country(eligible, std::nullopt);
  snapshot.selected_by_country = counts_by_country(eligible, std::nullopt);
  snapshot.weight_by_country = counts_by_country(eligible, static_cast<double>(std::max(1, snapshot.selected_count)));
  (void)execution_model;
  return snapshot;
}

std::unordered_map<std::string, double> ResearchDataset::counts_by_country(
    const std::vector<int>& security_indices, std::optional<double> denom) const {
  std::unordered_map<std::string, double> counts;
  for (int sec_idx : security_indices) {
    counts[security_country_[sec_idx]] += 1.0;
  }
  if (denom && *denom > 0.0) {
    for (auto& [key, value] : counts) {
      value /= *denom;
    }
  }
  return counts;
}

std::vector<std::vector<PositionSample>> ResearchDataset::negative_control_months(
    const SimulationParams& params,
    const std::string& start_month,
    const std::string& end_month,
    const std::vector<std::string>& excluded_countries) const {
  std::vector<std::vector<PositionSample>> months;
  auto indices = window_signal_indices(start_month, end_month);
  if (indices.empty()) {
    return months;
  }
  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());
  auto score_it = scores_.find(score_key(params.lookback, params.skip));
  if (score_it == scores_.end()) {
    return months;
  }
  const auto& scores = score_it->second;
  const auto& capacity_mask = capacity_masks_.at(params.top_n);
  for (int month_idx : indices) {
    std::vector<PositionSample> positions;
    positions.reserve(security_count_);
    for (int sec_idx = 0; sec_idx < security_count_; ++sec_idx) {
      int64_t idx = month_idx * security_count_ + sec_idx;
      if (!eligible_full_nordics_[idx] || !asof_matches_anchor_[idx] || !capacity_mask[idx]) {
        continue;
      }
      if (excluded.find(security_country_[sec_idx]) != excluded.end()) {
        continue;
      }
      double score = scores[idx];
      double next_return = gross_return_next_open_[idx];
      if (!std::isfinite(score) || !std::isfinite(next_return)) {
        continue;
      }
      positions.push_back({score, next_return});
    }
    months.push_back(std::move(positions));
  }
  return months;
}

bool ResearchDataset::variant_has_any(
    const std::string& universe_variant,
    const std::vector<std::string>& excluded_countries) const {
  std::unordered_set<std::string> excluded(excluded_countries.begin(), excluded_countries.end());
  const std::vector<uint8_t>* base_mask = nullptr;
  if (universe_variant == "Full Nordics") {
    base_mask = &eligible_full_nordics_;
  } else if (universe_variant == "SE-only") {
    base_mask = &eligible_se_only_;
  } else {
    return false;
  }
  for (int64_t idx = 0; idx < month_count_ * security_count_; ++idx) {
    if (!(*base_mask)[idx] || !asof_matches_anchor_[idx]) {
      continue;
    }
    int sec_idx = static_cast<int>(idx % security_count_);
    if (excluded.find(security_country_[sec_idx]) != excluded.end()) {
      continue;
    }
    return true;
  }
  return false;
}

const std::vector<uint8_t>& ResearchDataset::capacity_mask(int top_n) const {
  auto it = capacity_masks_.find(top_n);
  if (it == capacity_masks_.end()) {
    throw std::runtime_error("Missing capacity mask for top_n=" + std::to_string(top_n));
  }
  return it->second;
}

const std::vector<double>& ResearchDataset::cost_fractions(
    int top_n,
    const std::string& execution_model,
    const std::string& fx_scenario) const {
  int exec_id = execution_model == "next_open" ? 0 : 1;
  int fx_id = fx_scenario == "low" ? 0 : (fx_scenario == "base" ? 1 : 2);
  long long key = (static_cast<long long>(top_n) << 32) | (exec_id << 16) | fx_id;
  auto it = cost_fractions_.find(key);
  if (it == cost_fractions_.end()) {
    throw std::runtime_error("Missing cost fractions for top_n=" + std::to_string(top_n));
  }
  return it->second;
}

}  // namespace am
