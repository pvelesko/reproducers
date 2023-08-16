// Example for dispatching a SPIR-V Kernel using Level Zero on the Intel HD
// Graphics Sample based on the test-suite exanples from Level-Zero:
//      https://github.com/intel/compute-runtime/blob/master/level_zero/core/test/black_box_tests/zello_world_gpu.cpp

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

// #define IMMEDIATE
#include "common.hpp"
#include "ze_api.h"

int main(int argc, char **argv) {
  setupLevelZero();
  compileKernel("SlowKernel.spv", "myKernel");

  ze_event_pool_handle_t EventPool_;
  unsigned int PoolFlags =
      ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

  ze_event_pool_desc_t EventPoolDesc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, // stype
      nullptr,                           // pNext
      PoolFlags,                         // Flags
      10                                 // count
  };

  ZE_CHECK(zeEventPoolCreate(context, &EventPoolDesc, 0, nullptr, &EventPool_));

  ze_event_desc_t EventDesc = {
      ZE_STRUCTURE_TYPE_EVENT_DESC, // stype
      nullptr,                      // pNext
      0,                            // index
      ZE_EVENT_SCOPE_FLAG_HOST,     // ensure memory/cache coherency required on
                                    // signal
      ZE_EVENT_SCOPE_FLAG_HOST      // ensure memory coherency across device and
                                    // Host after Event completes
  };

  ze_event_handle_t StartEvent, EndEvent, timestampRecordEventStart,
      timestampRecordEventStop, myEvent;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &StartEvent));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &EndEvent));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &timestampRecordEventStart));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &timestampRecordEventStop));
  EventDesc.index++;
  ZE_CHECK(zeEventCreate(EventPool_, &EventDesc, &myEvent));

  ze_device_mem_alloc_desc_t deviceMemDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  deviceMemDesc.ordinal = 0;

  ze_host_mem_alloc_desc_t hostMemDesc = {
      ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  hostMemDesc.pNext = nullptr;

  void *startTime = nullptr;
  void *endTime = nullptr;
  zeMemAllocDevice(context, &deviceMemDesc, sizeof(uint64_t), 1, device,
                   &startTime);
  zeMemAllocDevice(context, &deviceMemDesc, sizeof(uint64_t), 1, device,
                   &endTime);
  //   zeMemAllocShared(context, &deviceMemDesc, &hostMemDesc, sizeof(uint64_t),
  //   8, device,
  //                    (void**)(&startTime));
  //   zeMemAllocShared(context, &deviceMemDesc, &hostMemDesc, sizeof(uint64_t),
  //   8, device,
  //                    (void**)(&endTime));


  uint64_t hostTimestampStart, deviceTimestampStart;
  uint64_t hostTimestampStop, deviceTimestampStop;
  zeDeviceGetGlobalTimestamps(device, &hostTimestampStart,
                              &deviceTimestampStart);

  // get the start time for the host
  auto start = std::chrono::steady_clock::now();

  zeCommandListAppendWriteGlobalTimestamp(
      cmdList, (uint64_t *)startTime, timestampRecordEventStart, 0, nullptr);
  std::cout << "Launching Kernel" << std::endl;
  // Launch kernel on the GPU
  ze_group_count_t dispatch;
  dispatch.groupCountX = 1;
  dispatch.groupCountY = 1;
  dispatch.groupCountZ = 1;
  ZE_CHECK(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch, EndEvent,
                                           1, &timestampRecordEventStart));
  std::cout << "Kernel Launched" << std::endl;
  zeCommandListAppendWriteGlobalTimestamp(
      cmdList, (uint64_t *)endTime, timestampRecordEventStop, 1, &EndEvent);

  // copy back the timestamp from device to host
  uint64_t startTimeHost = 0, endTimeHost = 0;
  // copy from device
  zeCommandListAppendBarrier(cmdList, nullptr, 0, nullptr);
  ZE_CHECK(zeCommandListAppendMemoryCopy(cmdList, &startTimeHost, startTime,
                                         sizeof(startTime), nullptr, 0,
                                         nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(cmdList, &endTimeHost, endTime,
                                         sizeof(endTime), nullptr, 0, nullptr));
  zeCommandListAppendBarrier(cmdList, myEvent, 0, nullptr);

  // query StartEvent, then Event, then StartEvent, then Event
  //   ZE_CHECK(
  //       zeEventHostSynchronize(StartEvent,
  //       std::numeric_limits<uint64_t>::max()));
  //   ze_result_t Status;
  //   Status = zeEventQueryStatus(StartEvent);
  //   std::cout << "StartEvent Query: " << resultToString(Status) << std::endl;
  //   Status = zeEventQueryStatus(timestampRecordEventStop);
  //   std::cout << "EndEvent Query: " << resultToString(Status) << std::endl;
  //   Status = zeEventQueryStatus(StartEvent);
  //   std::cout << "StartEvent Query: " << resultToString(Status) << std::endl;
  //   Status = zeEventQueryStatus(timestampRecordEventStop);
  //   std::cout << "EndEvent Query: " << resultToString(Status) << std::endl;

  execCmdList(cmdList);
  std::cout << "Host Synchronize ...";
  ZE_CHECK(
      zeEventHostSynchronize(myEvent, std::numeric_limits<uint64_t>::max()));
  ZE_CHECK(zeEventHostSynchronize(timestampRecordEventStop,
                                  std::numeric_limits<uint64_t>::max()));
  auto end = std::chrono::steady_clock::now();
  auto hostTimeInMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << " complete" << std::endl;

  zeDeviceGetGlobalTimestamps(device, &hostTimestampStop, &deviceTimestampStop);
  ze_kernel_timestamp_result_t res{};
  ZE_CHECK(zeEventQueryKernelTimestamp(EndEvent, &res));

  // print the time in seconds
  std::cout << "std::chrono Host: " << hostTimeInMs << " ms" << std::endl;
  std::cout << "zeDeviceGetGlobalTimestamps Host: "
            << timestampToMs(hostTimestampStart, hostTimestampStop) << " ms"
            << std::endl;
  std::cout << "zeDeviceGetGlobalTimestamps Device: "
            << timestampToMs(deviceTimestampStart, deviceTimestampStop) << " ms"
            << std::endl;
  std::cout << "zeEventQueryKernelTimestamp Context: "
            << timestampToMsKernel(res.context.kernelStart,
                                   res.context.kernelEnd)
            << " ms" << std::endl;
  std::cout << "zeEventQueryKernelTimestamp Global: "
            << timestampToMsKernel(res.global.kernelStart, res.global.kernelEnd)
            << " ms" << std::endl;

  ze_device_properties_t devProperties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  zeDeviceGetProperties(device, &devProperties);

  uint64_t timerResolution = devProperties.timerResolution;
  float copyOutDuration =
      ((endTimeHost - startTimeHost) * timerResolution) / 1000000.0;
  std::cout << "zeCommandListAppendWriteGlobalTimestamp Device: "
            << copyOutDuration << " ms" << std::endl;

  //   std::cout << "zeCommandListAppendWriteGlobalTimestamp Device: "
  //             << timestampToMs(startTimeHost, endTimeHost) << " ms" <<
  //             std::endl;

  cleanupLevelZero();
  return 0;
}