cmake_minimum_required(VERSION 3.0.0)
project(Reproducer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Debug)
set(OFFLOAD_TARGETS "icllp") # ocloc compile --help to get list of supported targets

find_library(OpenCL_LIBRARY OpenCL REQUIRED PATHS ENV LD_LIBRARY_PATH)
# find_library(Level0_LIBRARY ze_loader REQUIRED PATHS ENV LD_LIBRARY_PATH)

if(NOT DEFINED LevelZero_LIBRARY)
  message(STATUS "LevelZero_LIBRARY was not set. Searching for ze_loader.so in LD_LIBRARY_PATH")
  find_library(LevelZero_LIBRARY NAMES ze_loader PATHS ENV LD_LIBRARY_PATH NO_CACHE)
  if(LevelZero_LIBRARY)
    get_filename_component(LevelZeroLib_DIR ${LevelZero_LIBRARY} DIRECTORY CACHE)
    # Check if the library is in the system path
    if("${LevelZeroLib_DIR}" STREQUAL "/usr/lib/x86_64-linux-gnu")
      set(LevelZeroInclude_DIR "/usr/include/level_zero")
    else()
      # get the path to the LevelZero directory
      set(LevelZeroInclude_DIR "${LevelZeroLib_DIR}/../include/level_zero")
    endif()

    message(STATUS "Setting LevelZeroInclude_DIR to ${LevelZeroInclude_DIR}")
    message(STATUS "Setting LevelZeroLib_DIR to ${LevelZeroLib_DIR}")

    # make sure that ze_api.h exists in LevelZeroInclude_DIR
    if(NOT EXISTS "${LevelZeroInclude_DIR}/ze_api.h")
      message(FATAL_ERROR "ze_api.h was not found in ${LevelZeroInclude_DIR}")
    endif()

    # Re-enable this check once more deployments reach 1.6
    # # grep ze_api.h for @version and make sure it's at least 1.6
    # execute_process(COMMAND grep "@version" "${LevelZeroInclude_DIR}/ze_api.h" OUTPUT_VARIABLE ZE_API_VERSION)
    # string(REGEX MATCHALL "[0-9]+" ZE_API_VERSION "${ZE_API_VERSION}")
    # list(GET ZE_API_VERSION 0 ZE_API_MAJOR)
    # list(GET ZE_API_VERSION 1 ZE_API_MINOR)
    # if(ZE_API_MAJOR LESS 1 OR (ZE_API_MAJOR EQUAL 1 AND ZE_API_MINOR LESS 6))
    #   message(FATAL_ERROR "LevelZero version must be at least 1.6")
    # else()
    #   message(STATUS "LevelZero version: ${ZE_API_MAJOR}.${ZE_API_MINOR}")
    # endif()
    
  else()
    message(STATUS "Level Zero not found")
  endif()
endif()


add_executable(driver main.cpp)
target_link_libraries(driver ${OpenCL_LIBRARY} ${LevelZero_LIBRARY})
target_include_directories(driver PRIVATE ${LevelZeroInclude_DIR})

# add_custom_command( OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/KernelGPU.spv"
#                     DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/KernelGPU.cl"
#         COMMAND ocloc compile 
#         -file "${CMAKE_CURRENT_SOURCE_DIR}/KernelGPU.cl"
#         -device ${OFFLOAD_TARGETS}
#         -output_no_suffix
#         COMMENT "Building KernelGPU.spv"
#         VERBATIM)

# add_custom_target(Kernel DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/KernelGPU.spv")
# add_dependencies(driver Kernel)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
