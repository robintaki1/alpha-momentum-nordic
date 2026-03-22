#pragma once

#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace am {

struct CliArgs {
  std::vector<std::string> args;

  explicit CliArgs(int argc, char** argv);
  [[nodiscard]] bool has_flag(const std::string& flag) const;
  [[nodiscard]] std::optional<std::string> value(const std::string& flag) const;
  [[nodiscard]] std::string value_or(const std::string& flag, const std::string& fallback) const;
};

arrow::Result<std::shared_ptr<arrow::Table>> read_parquet(const std::filesystem::path& path);
arrow::Status write_parquet(const std::filesystem::path& path, const std::shared_ptr<arrow::Table>& table);

std::string to_lower(std::string input);
bool ends_with(const std::string& value, const std::string& suffix);
bool contains_case_insensitive(const std::string& value, const std::string& needle);

nlohmann::json read_json(const std::filesystem::path& path);
void write_json(const std::filesystem::path& path, const nlohmann::json& payload);

}  // namespace am
