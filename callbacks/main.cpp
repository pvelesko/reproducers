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

#include "KernelGPU.hpp"
#include "common.hpp"
#include "ze_api.h"

#define IMMEDIATE 1
int main(int argc, char **argv) {
#if IMMEDIATE
  std::cout << "Using immediate command list\n";
#else
  std::cout << "Using regular command list\n";
#endif
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
#if IMMEDIATE
  ZE_CHECK(
      zeCommandListCreateImmediate(context, device, &cmdQueueDesc, &cmdList));
#else
  ZE_CHECK(zeCommandListCreate(context, device, &cmdListDesc, &cmdList));
#endif

  // Create an event pool and a single event
  ze_event_handle_t Event;
  ze_event_pool_handle_t EventPool_;
  unsigned int PoolFlags =
      ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,  // stype
      nullptr,                            // pNext
      PoolFlags,                          // Flags
      10                                   // count
  };

  ZE_CHECK(
      zeEventPoolCreate(context, &EventPoolDesc, 0, nullptr, &EventPool_));

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC,  // stype
      nullptr,                       // pNext
      0,                             // index
      ZE_EVENT_SCOPE_FLAG_HOST,  // ensure memory/cache coherency required on
                                 // signal
      ZE_EVENT_SCOPE_FLAG_HOST   // ensure memory coherency across device and
                                 // Host after Event completes
  };

  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &Event));

  ze_event_handle_t HostSignalEvent;
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &HostSignalEvent));

  ze_event_handle_t GpuReady;
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &GpuReady));

  // Create two buffers
  const uint32_t items = 1024;
  constexpr size_t allocSize = items * items * sizeof(int);
  ze_device_mem_alloc_desc_t memAllocDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  memAllocDesc.ordinal = 0;

  ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};

  void *sharedA = nullptr;
  ZE_CHECK(zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1,
                                device, &sharedA));

  void *sharedB = nullptr;
  ZE_CHECK(zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1,
                                device, &sharedB));

  void *dstResult = nullptr;
  ZE_CHECK(zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1,
                                device, &dstResult));

  // memory initialization
  constexpr uint8_t val = 2;
  memset(sharedA, val, allocSize);
  memset(sharedB, 3, allocSize);
  memset(dstResult, 0, allocSize);

  // Module Initialization
  ze_module_handle_t module = nullptr;
  ze_kernel_handle_t kernel = nullptr;

  std::ifstream file("KernelGPU.spv", std::ios::binary);
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
  kernelDesc.pKernelName = "mxm";
  ZE_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));

  uint32_t groupSizeX = 32u;
  uint32_t groupSizeY = 32u;
  uint32_t groupSizeZ = 1u;
  ZE_CHECK(zeKernelSuggestGroupSize(kernel, items, items, 1U, &groupSizeX,
                                        &groupSizeY, &groupSizeZ));
  ZE_CHECK(
      zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, groupSizeZ));

  std::cout << "Group X: " << groupSizeX << std::endl;
  std::cout << "Group Y: " << groupSizeY << std::endl;

  // Push arguments
  ZE_CHECK(
      zeKernelSetArgumentValue(kernel, 0, sizeof(dstResult), &dstResult));
  ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(sharedA), &sharedA));
  ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(sharedB), &sharedB));
  ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(int), &items));

  // Kernel thread-dispatch
  ze_group_count_t dispatch;
  dispatch.groupCountX = items / groupSizeX;
  dispatch.groupCountY = items / groupSizeY;
  dispatch.groupCountZ = 1;

  std::cout << "Enqueue barrier prior to kernel\n";
  ZE_CHECK(zeCommandListAppendBarrier(cmdList, GpuReady, 0, nullptr));
  // Launch kernel on the GPU
  std::cout << "Launching kernel\n";
  ZE_CHECK(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch,
                                               Event, 1, &HostSignalEvent));
  // query GpuReady Event
  std::cout << "Querying GpuReady Event\n";
  zeEventHostSynchronize(GpuReady, 10000);

  std::cout << "Host Signal Blocking Event\n";
  ZE_CHECK(zeEventHostSignal(HostSignalEvent));

  auto begin = std::chrono::steady_clock::now();

#ifndef IMMEDIATE
  // Close list abd submit for execution
  ZE_CHECK(zeCommandListClose(cmdList));
  ZE_CHECK(
      zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr));
#endif
  ZE_CHECK(
      zeEventHostSynchronize(Event, std::numeric_limits<uint64_t>::max()));
  auto end = std::chrono::steady_clock::now();

  ze_kernel_timestamp_result_t res{};
  ZE_CHECK(zeEventQueryKernelTimestamp(Event, &res));
  std::cout << "Kernel Event Query: " << res.context.kernelEnd << std::endl;

  // Validate
  bool outputValidationSuccessful = true;

  uint32_t *resultSeq = (uint32_t *)malloc(allocSize);
  uint32_t *dstInt = static_cast<uint32_t *>(dstResult);
  uint32_t *srcA = static_cast<uint32_t *>(sharedA);
  uint32_t *srcB = static_cast<uint32_t *>(sharedB);

  std::chrono::steady_clock::time_point beginSeq =
      std::chrono::steady_clock::now();
  KernelCPU(srcA, srcB, resultSeq, items);
  std::chrono::steady_clock::time_point endSeq =
      std::chrono::steady_clock::now();

  auto elapsedParallel =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  auto elapsedSequential =
      std::chrono::duration_cast<std::chrono::nanoseconds>(endSeq - beginSeq)
          .count();
  std::cout << "GPU Kernel = " << elapsedParallel << " [ns]" << std::endl;
  std::cout << "SEQ Kernel = " << elapsedSequential << " [ns]" << std::endl;
  auto speedup = elapsedSequential / elapsedParallel;
  std::cout << "Speedup = " << speedup << "x" << std::endl;

  int n = items;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (resultSeq[i * n + j] != dstInt[i * n + j]) {
        outputValidationSuccessful = false;
        break;
      }
    }
  }

  std::cout << "\nMatrix Multiply validation "
            << (outputValidationSuccessful ? "PASSED" : "FAILED") << "\n";

  // Cleanup
  ZE_CHECK(zeMemFree(context, dstResult));
  ZE_CHECK(zeMemFree(context, sharedA));
  ZE_CHECK(zeMemFree(context, sharedB));
  ZE_CHECK(zeCommandListDestroy(cmdList));
  ZE_CHECK(zeCommandQueueDestroy(cmdQueue));
  ZE_CHECK(zeContextDestroy(context));

  return 0;
}