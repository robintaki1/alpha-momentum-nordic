#include "am/core.h"

#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <spdlog/spdlog.h>
#include <parquet/exception.h>

#include <fstream>
#include <sstream>

namespace am {

CliArgs::CliArgs(int argc, char** argv) {
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    args.emplace_back(argv[i] ? argv[i] : "");
  }
}

bool CliArgs::has_flag(const std::string& flag) const {
  for (const auto& arg : args) {
    if (arg == flag) {
      return true;
    }
  }
  return false;
}

std::optional<std::string> CliArgs::value(const std::string& flag) const {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == flag) {
      return args[i + 1];
    }
  }
  return std::nullopt;
}

std::string CliArgs::value_or(const std::string& flag, const std::string& fallback) const {
  auto found = value(flag);
  return found ? *found : fallback;
}

arrow::Result<std::shared_ptr<arrow::Table>> read_parquet(const std::filesystem::path& path) {
  ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(path.string()));
  PARQUET_ASSIGN_OR_THROW(auto reader,
                          parquet::arrow::OpenFile(input, arrow::default_memory_pool()));
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
  return table;
}

arrow::Status write_parquet(const std::filesystem::path& path, const std::shared_ptr<arrow::Table>& table) {
  ARROW_ASSIGN_OR_RAISE(auto output, arrow::io::FileOutputStream::Open(path.string()));
  return parquet::arrow::WriteTable(
      *table,
      arrow::default_memory_pool(),
      output,
      static_cast<int64_t>(table->num_rows()));
}

std::string to_lower(std::string input) {
  for (char& ch : input) {
    ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
  }
  return input;
}

bool ends_with(const std::string& value, const std::string& suffix) {
  if (suffix.size() > value.size()) {
    return false;
  }
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains_case_insensitive(const std::string& value, const std::string& needle) {
  if (needle.empty()) {
    return true;
  }
  auto haystack = to_lower(value);
  auto target = to_lower(needle);
  return haystack.find(target) != std::string::npos;
}

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open JSON file: " + path.string());
  }
  nlohmann::json payload;
  input >> payload;
  return payload;
}

void write_json(const std::filesystem::path& path, const nlohmann::json& payload) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write JSON file: " + path.string());
  }
  output << payload.dump(2) << "\n";
}

}  // namespace am
