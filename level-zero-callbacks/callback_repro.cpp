// Level Zero Callback Reproducer - Aurora failure scenario
// Event used on IMMEDIATE cmd list cannot be used on REGULAR cmd list after reset
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <stack>
#include <mutex>
#include <memory>

#define ZE_CHECK(call) do { \
    ze_result_t res = (call); \
    if (res != ZE_RESULT_SUCCESS) { \
      fprintf(stderr, "L0 error 0x%x at %s:%d in %s\n", res, __FILE__, __LINE__, #call); \
      std::abort(); \
    } \
  } while (0)

struct L0Context {
  ze_driver_handle_t driver;
  ze_device_handle_t device;
  ze_context_handle_t context;
  ze_command_queue_handle_t cmdQueue;
  ze_command_list_handle_t cmdListImm;
  ze_command_list_desc_t cmdListDesc;
  ze_event_pool_handle_t eventPool;
  int32_t computeOrdinal;
  unsigned eventIndex = 0;
};

ze_event_handle_t createEvent(L0Context &ctx) {
  ze_event_desc_t desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, ctx.eventIndex++,
                          ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
  ze_event_handle_t event;
  ZE_CHECK(zeEventCreate(ctx.eventPool, &desc, &event));
  return event;
}

void initL0(L0Context &ctx) {
  printf("Initializing Level Zero...\n");
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  uint32_t driverCount = 0;
  ZE_CHECK(zeDriverGet(&driverCount, nullptr));
  std::vector<ze_driver_handle_t> drivers(driverCount);
  ZE_CHECK(zeDriverGet(&driverCount, drivers.data()));
  ctx.driver = drivers[0];

  uint32_t deviceCount = 0;
  ZE_CHECK(zeDeviceGet(ctx.driver, &deviceCount, nullptr));
  std::vector<ze_device_handle_t> devices(deviceCount);
  ZE_CHECK(zeDeviceGet(ctx.driver, &deviceCount, devices.data()));
  ctx.device = devices[0];

  ze_device_properties_t devProps = {};
  devProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ZE_CHECK(zeDeviceGetProperties(ctx.device, &devProps));
  printf("Device: %s (0x%x:0x%x)\n", devProps.name, devProps.vendorId, devProps.deviceId);

  ze_context_desc_t ctxDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ZE_CHECK(zeContextCreate(ctx.driver, &ctxDesc, &ctx.context));

  uint32_t queueGroupCount = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(ctx.device, &queueGroupCount, nullptr));
  std::vector<ze_command_queue_group_properties_t> props(queueGroupCount);
  for (auto &p : props) p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(ctx.device, &queueGroupCount, props.data()));

  ctx.computeOrdinal = -1;
  for (uint32_t i = 0; i < queueGroupCount; i++)
    if (props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ctx.computeOrdinal = i; break; }
  assert(ctx.computeOrdinal >= 0);

  ze_command_queue_desc_t queueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,
      (uint32_t)ctx.computeOrdinal, 0, ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ZE_CHECK(zeCommandQueueCreate(ctx.context, ctx.device, &queueDesc, &ctx.cmdQueue));
  ZE_CHECK(zeCommandListCreateImmediate(ctx.context, ctx.device, &queueDesc, &ctx.cmdListImm));

  ctx.cmdListDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, (uint32_t)ctx.computeOrdinal, 0};

  ze_event_pool_desc_t poolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                   ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 100};
  ZE_CHECK(zeEventPoolCreate(ctx.context, &poolDesc, 0, nullptr, &ctx.eventPool));
  printf("Initialized.\n\n");
}

int main() {
  printf("=== Level Zero Callback Reproducer - Aurora Failure ===\n\n");
  printf("Replicates: Event used on IMMEDIATE cmd list, reset,\n");
  printf("then used as WAIT on REGULAR cmd list -> ZE_RESULT_ERROR_INVALID_ARGUMENT\n\n");

  L0Context ctx;
  initL0(ctx);

  // Get event (CpuCallbackComplete in chipStar)
  ze_event_handle_t cpuCallbackComplete = createEvent(ctx);
  printf("1. Created event: %p\n", (void*)cpuCallbackComplete);

  // Use on IMMEDIATE as SIGNAL
  printf("2. Barrier on IMMEDIATE (signal)...\n");
  ZE_CHECK(zeCommandListAppendBarrier(ctx.cmdListImm, cpuCallbackComplete, 0, nullptr));
  printf("   SUCCESS\n");

  // Use on IMMEDIATE as WAIT
  printf("3. Barrier on IMMEDIATE (wait)...\n");
  ZE_CHECK(zeCommandListAppendBarrier(ctx.cmdListImm, nullptr, 1, &cpuCallbackComplete));
  printf("   SUCCESS\n");

  // Sync
  printf("4. Synchronize...\n");
  ZE_CHECK(zeEventHostSynchronize(cpuCallbackComplete, UINT64_MAX));
  ZE_CHECK(zeCommandListHostSynchronize(ctx.cmdListImm, UINT64_MAX));
  printf("   SUCCESS\n");

  // Reset event
  printf("5. zeEventHostReset...\n");
  ZE_CHECK(zeEventHostReset(cpuCallbackComplete));
  printf("   SUCCESS\n");

  // Create REGULAR cmd list
  printf("6. Create REGULAR command list...\n");
  ze_command_list_handle_t regularCmdList;
  ZE_CHECK(zeCommandListCreate(ctx.context, ctx.device, &ctx.cmdListDesc, &regularCmdList));
  printf("   Created: %p\n", (void*)regularCmdList);

  // GpuReady barrier (trace line 275-278)
  ze_event_handle_t gpuReady = createEvent(ctx);
  printf("7. Barrier on REGULAR (GpuReady)...\n");
  ZE_CHECK(zeCommandListAppendBarrier(regularCmdList, gpuReady, 0, nullptr));
  printf("   SUCCESS\n");

  // THE FAILING CALL: using event created on IMMEDIATE cmd list as WAIT on REGULAR cmd list
  ze_event_handle_t gpuAck = createEvent(ctx);
  printf("\n8. THE CRITICAL CALL:\n");
  printf("   zeCommandListAppendBarrier(\n");
  printf("     hCommandList = %p (REGULAR),\n", (void*)regularCmdList);
  printf("     hSignalEvent = %p (GpuAck),\n", (void*)gpuAck);
  printf("     numWaitEvents = 1,\n");
  printf("     phWaitEvents = [%p] (was used on IMMEDIATE)\n", (void*)cpuCallbackComplete);
  printf("   )\n");

  ze_result_t result = zeCommandListAppendBarrier(regularCmdList, gpuAck, 1, &cpuCallbackComplete);

  if (result == ZE_RESULT_SUCCESS) {
    printf("\n   -> SUCCESS\n");
    printf("\n*** Issue does NOT reproduce on this system ***\n");
  } else {
    printf("\n   -> ERROR 0x%x\n", result);
    printf("\n*** REPRODUCED Aurora failure! ***\n");
    printf("Root cause: Event used on IMMEDIATE cmd list cannot be\n");
    printf("used as WAIT on REGULAR cmd list after reset.\n");
  }

  // Cleanup
  zeCommandListDestroy(regularCmdList);
  zeEventDestroy(cpuCallbackComplete);
  zeEventDestroy(gpuReady);
  zeEventDestroy(gpuAck);
  zeEventPoolDestroy(ctx.eventPool);
  zeCommandListDestroy(ctx.cmdListImm);
  zeCommandQueueDestroy(ctx.cmdQueue);
  zeContextDestroy(ctx.context);

  printf("\n=== Done ===\n");
  return result == ZE_RESULT_SUCCESS ? 0 : 1;
}
