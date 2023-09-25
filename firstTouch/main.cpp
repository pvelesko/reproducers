// Example for dispatching a SPIR-V Kernel using Level Zero on the Intel HD
// Graphics Sample based on the test-suite exanples from Level-Zero:
//      https://github.com/intel/compute-runtime/blob/master/level_zero/core/test/black_box_tests/zello_world_gpu.cpp

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "common.hpp"
#include "ze_api.h"

struct Data {
  int *A_d;
} typedef Data;

int main(int argc, char **argv) {
  std::cout << "Using immediate command list\n";
  // Initialization
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // Get the driver
  uint32_t driverCount = 0;
  ZE_CHECK(zeDriverGet(&driverCount, nullptr));

  ze_driver_handle_t driverHandle;
  ZE_CHECK(zeDriverGet(&driverCount, &driverHandle));

  // Create the context
  ze_context_desc_t contextDescription = {};
  contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context;
  ZE_CHECK(zeContextCreate(driverHandle, &contextDescription, &context));

  // Get the device
  uint32_t deviceCount = 0;
  ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, nullptr));

  ze_device_handle_t device;
  ZE_CHECK(zeDeviceGet(driverHandle, &deviceCount, &device));

  // Print basic properties of the device
  ze_device_properties_t deviceProperties = {};
  ZE_CHECK(zeDeviceGetProperties(device, &deviceProperties));
  std::cout << "Device   : " << deviceProperties.name << "\n"
            << "Type     : "
            << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA")
            << "\n"
            << "Vendor ID: " << std::hex << deviceProperties.vendorId
            << std::dec << "\n";

  // Create a command queue
  uint32_t numQueueGroups = 0;
  ZE_CHECK(
      zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr));
  if (numQueueGroups == 0) {
    std::cout << "No queue groups found\n";
    std::terminate();
  } else {
    std::cout << "#Queue Groups: " << numQueueGroups << std::endl;
  }
  std::vector<ze_command_queue_group_properties_t> queueProperties(
      numQueueGroups);
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups,
                                                      queueProperties.data()));

  ze_command_queue_handle_t cmdQueue;
  ze_command_queue_desc_t cmdQueueDesc = {};
  for (uint32_t i = 0; i < numQueueGroups; i++) {
    if (queueProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      cmdQueueDesc.ordinal = i;
    }
  }

  cmdQueueDesc.index = 0;
  cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ZE_CHECK(zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue));

  // Create a command list
  ze_command_list_handle_t cmdList;
  ze_command_list_desc_t cmdListDesc = {};
  cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;
  ZE_CHECK(
      zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &cmdList));

  // Create two buffers
  constexpr size_t allocSize =  1 * sizeof(Data);

  ze_device_mem_alloc_desc_t memAllocDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  memAllocDesc.ordinal = 0;

  void *sharedA = nullptr;
  ZE_CHECK(zeMemAllocDevice(context, &memAllocDesc, allocSize, 1,
                                device, &sharedA));

//   Uncomment to PASS
  int firstTouch = 0;
  ZE_CHECK(zeCommandListAppendMemoryCopy(cmdList, sharedA, &firstTouch,
                                             sizeof(int), nullptr, 0, nullptr)); 
  ZE_CHECK(zeCommandListAppendBarrier(cmdList, nullptr, 0, nullptr));

  // Module Initialization
  ze_module_handle_t module = nullptr;
  ze_kernel_handle_t kernel = nullptr;

  std::ifstream file("firstTouch.spv", std::ios::binary);
  if (!file.is_open()) {
    std::cout << "binary file not found\n";
    std::terminate();
  }

  file.seekg(0, file.end);
  auto length = file.tellg();
  file.seekg(0, file.beg);

  std::unique_ptr<char[]> spirvInput(new char[length]);
  file.read(spirvInput.get(), length);
  file.close();

  ze_module_desc_t moduleDesc = {};
  ze_module_build_log_handle_t buildLog;
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(spirvInput.get());
  moduleDesc.inputSize = length;
  moduleDesc.pBuildFlags = "";

  auto status =
      zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
  if (status != ZE_RESULT_SUCCESS) {
    // print log
    size_t szLog = 0;
    zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

    char *stringLog = (char *)malloc(szLog);
    zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
    std::cout << "zeModuleCreate failed: Build log: " << stringLog << std::endl;
    std::abort();
  }
  ZE_CHECK(zeModuleBuildLogDestroy(buildLog));

  ze_kernel_desc_t kernelDesc = {};
  kernelDesc.pKernelName = "setOne";
  ZE_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));

  uint32_t groupSizeX = 1u;
  uint32_t groupSizeY = 1u;
  uint32_t groupSizeZ = 1u;
  ZE_CHECK(
      zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, groupSizeZ));

  Data data;
  data.A_d = reinterpret_cast<int*>(sharedA);
  // Push arguments
  ZE_CHECK(
      zeKernelSetArgumentValue(kernel, 0, sizeof(data), &data));

  // Kernel thread-dispatch
  ze_group_count_t dispatch;
  dispatch.groupCountX = 1;
  dispatch.groupCountY = 1;
  dispatch.groupCountZ = 1;

  // Launch kernel on the GPU
  ZE_CHECK(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch,
                                               nullptr, 0, nullptr));

  int hostA[1] = {0};
  ZE_CHECK(zeCommandListAppendMemoryCopy(cmdList, hostA, sharedA,
                                                 sizeof(int), nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendBarrier(cmdList, nullptr, 0, nullptr));
  std::cout << "HOST: sharedA[0] = " << static_cast<int>(hostA[0]) << std::endl;
  // Cleanup
  ZE_CHECK(zeMemFree(context, sharedA));
  ZE_CHECK(zeCommandListDestroy(cmdList));
  ZE_CHECK(zeCommandQueueDestroy(cmdQueue));
  ZE_CHECK(zeContextDestroy(context));

  return 0;
}