// Minimal reproducer for Intel Data Center GPU Max OpenCL bug:
// clSetUserEventStatus does not wake up barriers on IN-ORDER queues.
//
// Bug: On in-order queues, clFinish hangs even after user event is signaled.
// Out-of-order queues work correctly.
//
// Fails on: Intel Data Center GPU Max
// Works on: Intel Arc A770

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <thread>
#include <chrono>
#include <atomic>

std::atomic<bool> finished{false};

bool testQueue(cl_context context, cl_device_id device, bool outOfOrder) {
  cl_int err;
  cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES, 
    outOfOrder ? (cl_queue_properties)CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0,
    0
  };
  
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 
                                                               outOfOrder ? props : nullptr, &err);
  printf("\n=== Testing %s queue ===\n", outOfOrder ? "OUT-OF-ORDER" : "IN-ORDER");
  
  cl_event userEvent = clCreateUserEvent(context, &err);
  
  cl_event barrier;
  clEnqueueBarrierWithWaitList(queue, 1, &userEvent, &barrier);
  
  finished.store(false);
  
  // Signal user event from another thread
  std::thread([userEvent]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    printf("Signaling user event...\n");
    cl_int e = clSetUserEventStatus(userEvent, CL_COMPLETE);
    printf("clSetUserEventStatus returned: %d\n", e);
  }).detach();
  
  // Try clFinish with timeout
  std::thread([&queue]() {
    clFinish(queue);
    finished.store(true);
  }).detach();
  
  // Wait up to 3 seconds
  for (int i = 0; i < 30; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (finished.load()) {
      printf("PASSED - clFinish returned\n");
      clReleaseEvent(barrier);
      clReleaseEvent(userEvent);
      clReleaseCommandQueue(queue);
      return true;
    }
  }
  
  printf("FAILED - clFinish hung (timeout after 3s)\n");
  // Don't cleanup - queue is stuck
  return false;
}

int main() {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_int err;
  
  cl_uint numPlatforms;
  clGetPlatformIDs(0, nullptr, &numPlatforms);
  cl_platform_id* platforms = new cl_platform_id[numPlatforms];
  clGetPlatformIDs(numPlatforms, platforms, nullptr);
  
  for (cl_uint i = 0; i < numPlatforms; i++) {
    cl_uint numDevices;
    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
      platform = platforms[i];
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
      break;
    }
  }
  delete[] platforms;
  
  if (!device) {
    fprintf(stderr, "No GPU found\n");
    return 1;
  }
  
  char name[256];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  printf("Device: %s\n", name);
  
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  
  bool inOrderOk = testQueue(context, device, false);   // in-order
  bool outOfOrderOk = testQueue(context, device, true); // out-of-order
  
  printf("\n=== Summary ===\n");
  printf("In-order queue:     %s\n", inOrderOk ? "PASS" : "FAIL (BUG!)");
  printf("Out-of-order queue: %s\n", outOfOrderOk ? "PASS" : "FAIL");
  
  clReleaseContext(context);
  return (inOrderOk && outOfOrderOk) ? 0 : 1;
}
