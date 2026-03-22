#include <fmt/core.h>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    spdlog::info("{} invoked", argv[0]);
    fmt::print("{} is not implemented yet.\n", argv[0]);
    return 1;
}
