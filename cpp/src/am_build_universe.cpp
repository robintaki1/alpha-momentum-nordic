#include "am/core.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>

namespace {

arrow::Result<std::shared_ptr<arrow::StringArray>> combine_to_string(
    const std::shared_ptr<arrow::ChunkedArray>& chunked) {
  ARROW_ASSIGN_OR_RAISE(auto casted, arrow::compute::Cast(chunked, arrow::utf8()));
  auto casted_chunked = casted.chunked_array();
  ARROW_ASSIGN_OR_RAISE(auto combined,
                        arrow::Concatenate(casted_chunked->chunks(), arrow::default_memory_pool()));
  return std::static_pointer_cast<arrow::StringArray>(combined);
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> build_keep_mask(
    const std::shared_ptr<arrow::Table>& table) {
  auto security_col = table->GetColumnByName("security_id");
  auto exchange_col = table->GetColumnByName("exchange_group");
  auto country_col = table->GetColumnByName("country_code");
  if (!security_col || !exchange_col || !country_col) {
    return arrow::Status::Invalid("Missing required columns: security_id, exchange_group, country_code");
  }
  ARROW_ASSIGN_OR_RAISE(auto security, combine_to_string(security_col));
  ARROW_ASSIGN_OR_RAISE(auto exchange, combine_to_string(exchange_col));
  ARROW_ASSIGN_OR_RAISE(auto country, combine_to_string(country_col));

  arrow::BooleanBuilder builder;
  const int64_t length = table->num_rows();
  ARROW_RETURN_NOT_OK(builder.Reserve(length));
  for (int64_t i = 0; i < length; ++i) {
    std::string security_id = security->IsNull(i) ? "" : std::string(security->GetView(i));
    std::string exchange_group = exchange->IsNull(i) ? "" : std::string(exchange->GetView(i));
    std::string country_code = country->IsNull(i) ? "" : std::string(country->GetView(i));

    bool is_fi = am::to_lower(country_code) == "fi";
    bool is_he = am::ends_with(am::to_lower(security_id), ".he");
    bool is_helsinki = am::contains_case_insensitive(exchange_group, "helsinki");
    bool keep = !(is_fi || is_he || is_helsinki);
    builder.UnsafeAppend(keep);
  }
  std::shared_ptr<arrow::Array> mask_array;
  ARROW_RETURN_NOT_OK(builder.Finish(&mask_array));
  return std::static_pointer_cast<arrow::BooleanArray>(mask_array);
}

}  // namespace

int main(int argc, char** argv) {
  am::CliArgs args(argc, argv);
  const std::filesystem::path input_path = args.value_or("--input", "data/universe_pti.parquet");
  const std::filesystem::path output_path = args.value_or("--out", input_path.string());
  const bool dry_run = args.has_flag("--dry-run");

  spdlog::info("am_build_universe: reading {}", input_path.string());
  auto table_result = am::read_parquet(input_path);
  if (!table_result.ok()) {
    spdlog::error("Failed to read parquet: {}", table_result.status().ToString());
    return 2;
  }
  auto table = *table_result;

  auto mask_result = build_keep_mask(table);
  if (!mask_result.ok()) {
    spdlog::error("Failed to build FI mask: {}", mask_result.status().ToString());
    return 3;
  }
  auto mask = *mask_result;
  auto filtered_result = arrow::compute::Filter(table, mask);
  if (!filtered_result.ok()) {
    spdlog::error("Failed to filter table: {}", filtered_result.status().ToString());
    return 3;
  }
  auto filtered_table = filtered_result->table();

  const int64_t dropped = table->num_rows() - filtered_table->num_rows();
  spdlog::info("Filtered {} rows containing FI/HE/Helsinki", dropped);

  if (dry_run) {
    fmt::print("Dry run complete. Would write {} rows to {}.\n",
               filtered_table->num_rows(),
               output_path.string());
    return 0;
  }

  auto status = am::write_parquet(output_path, filtered_table);
  if (!status.ok()) {
    spdlog::error("Failed to write parquet: {}", status.ToString());
    return 4;
  }
  fmt::print("Wrote {} rows to {}.\n", filtered_table->num_rows(), output_path.string());
  return 0;
}
