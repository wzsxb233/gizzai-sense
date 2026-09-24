# The Sense readout in C++, on top of llama.cpp. The same code serves a laptop, an iPhone (Metal) and an
# Android phone (Vulkan): only llama.cpp's own backend flags change.
#
#   cmake -B build -DLLAMA_CPP_DIR=/root/llama.cpp -DLLAMA_CPP_BUILD=/root/llama.cpp/build
#   cmake --build build -j
cmake_minimum_required(VERSION 3.16)
project(sense_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_DIR   "" CACHE PATH "llama.cpp source directory")
set(LLAMA_CPP_BUILD "" CACHE PATH "llama.cpp build directory (already built)")
if(NOT LLAMA_CPP_DIR OR NOT LLAMA_CPP_BUILD)
  message(FATAL_ERROR "set -DLLAMA_CPP_DIR=... and -DLLAMA_CPP_BUILD=...")
endif()

add_library(sense STATIC sense.cpp)
target_include_directories(sense PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLAMA_CPP_DIR}/include
  ${LLAMA_CPP_DIR}/common
  ${LLAMA_CPP_DIR}/ggml/include
  ${LLAMA_CPP_DIR}/vendor)

find_library(LLAMA_LIB    NAMES llama        PATHS ${LLAMA_CPP_BUILD}/bin ${LLAMA_CPP_BUILD} REQUIRED)
find_library(COMMON_LIB   NAMES llama-common PATHS ${LLAMA_CPP_BUILD}/bin ${LLAMA_CPP_BUILD} REQUIRED)
find_library(GGML_LIB     NAMES ggml         PATHS ${LLAMA_CPP_BUILD}/bin ${LLAMA_CPP_BUILD} REQUIRED)
target_link_libraries(sense PUBLIC ${COMMON_LIB} ${LLAMA_LIB} ${GGML_LIB})

add_executable(sense-cli sense-cli.cpp)
target_link_libraries(sense-cli PRIVATE sense)

# The series side: Chronos-2's encoder as a ggml graph, and the tool that checks it stage by stage
# against the real encoder's own tensors.
add_library(series STATIC series.cpp)
target_include_directories(series PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLAMA_CPP_DIR}/ggml/include)
find_library(GGML_BASE_LIB NAMES ggml-base PATHS ${LLAMA_CPP_BUILD}/bin ${LLAMA_CPP_BUILD})
find_library(GGML_CPU_LIB  NAMES ggml-cpu  PATHS ${LLAMA_CPP_BUILD}/bin ${LLAMA_CPP_BUILD})
target_link_libraries(series PUBLIC ${GGML_LIB})
foreach(lib ${GGML_BASE_LIB} ${GGML_CPU_LIB})
  if(lib)
    target_link_libraries(series PUBLIC ${lib})
  endif()
endforeach()

add_executable(series-cli series-cli.cpp)
target_link_libraries(series-cli PRIVATE series)

# The text half of Sense-T's input, and the check that it is byte-identical to the released Python.
add_library(prompt STATIC prompt.cpp)
target_include_directories(prompt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(prompt-cli prompt-cli.cpp)
target_include_directories(prompt-cli PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_link_libraries(prompt-cli PRIVATE prompt)

# Sense-T end to end: prompt text + series encoder + the readout, in one binary.
add_executable(sense-t-cli sense-t-cli.cpp)
target_include_directories(sense-t-cli PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_link_libraries(sense-t-cli PRIVATE sense series prompt)

# Whether llama.cpp's embedding path carries soft tokens the way transformers does.
add_executable(splice-cli splice-cli.cpp)
target_include_directories(splice-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LLAMA_CPP_DIR}/include
                                              ${LLAMA_CPP_DIR}/ggml/include)
target_link_libraries(splice-cli PRIVATE ${LLAMA_LIB} ${GGML_LIB})
