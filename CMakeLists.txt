cmake_minimum_required(VERSION 3.20)
project(alpha_momentum_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_library(am_core STATIC cpp/src/am_core.cpp cpp/src/config.cpp cpp/src/dataset.cpp cpp/src/validation.cpp)
target_include_directories(am_core PUBLIC cpp/include)
target_link_libraries(
  am_core
  PUBLIC
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Arrow::arrow_shared
    Parquet::parquet_shared
)
target_compile_options(am_core PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/EHsc>)

define_property(TARGET PROPERTY AM_COMPONENT BRIEF_DOCS "" FULL_DOCS "")

function(am_add_cli name)
  add_executable(${name} cpp/src/${name}.cpp)
  target_link_libraries(${name} PRIVATE am_core)
endfunction()

am_add_cli(am_build_universe)
am_add_cli(am_research_engine)
am_add_cli(am_cadence_compare)
am_add_cli(am_forward_monitor_payload)
