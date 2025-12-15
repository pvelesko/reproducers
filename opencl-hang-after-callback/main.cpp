#include <CL/cl.h>
#include <iostream>
#include <atomic>
#include <cassert>
#include <vector>

std::atomic<int> callbackCount{0};
std::atomic<int> executionOrder{0};

struct CallbackData {
  cl_event CallbackFinishEvent;
  int* Order;
  int ExpectedOrder;
};

void CL_CALLBACK pfn_notify(cl_event Event, cl_int CommandExecStatus, void *UserData) {
  (void)Event;
  (void)CommandExecStatus;
  CallbackData *Cb = static_cast<CallbackData *>(UserData);
  if (Cb == nullptr) {
    std::cerr << "ERROR: Callback data is null" << std::endl;
    return;
  }
  
  int order = executionOrder.fetch_add(1) + 1;
  callbackCount.fetch_add(1);
  
  std::cout << "testHostFunc called" << std::endl;
  std::cout << "order: " << order << std::endl;
  std::cout << "hostFuncCallCount: " << callbackCount.load() << std::endl;
  std::cout << "executionOrder: " << executionOrder.load() << std::endl;
  
  if (Cb->Order) {
    *Cb->Order = order;
  }
  
  if (Cb->CallbackFinishEvent != nullptr) {
    cl_int status = clSetUserEventStatus(Cb->CallbackFinishEvent, CL_COMPLETE);
    if (status != CL_SUCCESS) {
      std::cerr << "ERROR: clSetUserEventStatus failed with " << status << std::endl;
    }
  }
  
  delete Cb;
}

int main() {
  cl_int err;
  cl_uint numPlatforms = 0;
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  
  err = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms == 0) {
    std::cerr << "Failed to get platforms" << std::endl;
    return 1;
  }
  
  std::vector<cl_platform_id> platforms(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get platform IDs" << std::endl;
    return 1;
  }
  
  platform = platforms[0];
  
  cl_uint numDevices = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
  if (err != CL_SUCCESS || numDevices == 0) {
    std::cerr << "Failed to get GPU devices" << std::endl;
    return 1;
  }
  
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to get device ID" << std::endl;
    return 1;
  }
  
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create context" << std::endl;
    return 1;
  }
  
  queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create command queue" << std::endl;
    clReleaseContext(context);
    return 1;
  }
  
  callbackCount = 0;
  executionOrder = 0;
  
  int order1 = 0, order2 = 0;
  
  cl_event barrier1 = nullptr;
  err = clEnqueueBarrierWithWaitList(queue, 0, nullptr, &barrier1);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue barrier 1" << std::endl;
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  cl_event callbackEvent1 = clCreateUserEvent(context, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create user event 1" << std::endl;
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  CallbackData *cbData1 = new CallbackData{callbackEvent1, &order1, 1};
  
  err = clSetEventCallback(barrier1, CL_COMPLETE, pfn_notify, cbData1);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to set event callback 1" << std::endl;
    delete cbData1;
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  cl_event waitEvents1[] = {callbackEvent1};
  cl_event barrier2 = nullptr;
  err = clEnqueueBarrierWithWaitList(queue, 1, waitEvents1, &barrier2);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue barrier 2" << std::endl;
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  cl_event barrier3 = nullptr;
  err = clEnqueueBarrierWithWaitList(queue, 0, nullptr, &barrier3);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue barrier 3" << std::endl;
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  cl_event callbackEvent2 = clCreateUserEvent(context, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create user event 2" << std::endl;
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  CallbackData *cbData2 = new CallbackData{callbackEvent2, &order2, 2};
  
  err = clSetEventCallback(barrier3, CL_COMPLETE, pfn_notify, cbData2);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to set event callback 2" << std::endl;
    delete cbData2;
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  cl_event waitEvents2[] = {callbackEvent2};
  cl_event barrier4 = nullptr;
  err = clEnqueueBarrierWithWaitList(queue, 1, waitEvents2, &barrier4);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue barrier 4" << std::endl;
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  std::cout << "About to call clFinish immediately - this may hang..." << std::endl;
  std::cout << "Callback count before clFinish: " << callbackCount.load() << std::endl;
  std::cout << "NOTE: If callbacks haven't executed yet, clFinish will wait for barrier4," << std::endl;
  std::cout << "      which waits for callbackEvent2, which is only set by the callback." << std::endl;
  std::cout << "      If the callback doesn't execute, this will hang!" << std::endl;
  
  err = clFinish(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to finish queue" << std::endl;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  std::cout << "clFinish completed (callbacks should have executed)" << std::endl;
  std::cout << "Final callback count: " << callbackCount.load() << std::endl;
  
  if (callbackCount.load() != 2) {
    std::cerr << "FAIL: Expected 2 callbacks, got " << callbackCount.load() << std::endl;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  std::cout << "Callbacks completed successfully. Now attempting memory copy..." << std::endl;
  std::cout << "This is where the hang occurs in chipStar!" << std::endl;
  
  cl_int eventStatus1, eventStatus2, eventStatus3, eventStatus4;
  clGetEventInfo(barrier1, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus1, nullptr);
  clGetEventInfo(barrier2, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus2, nullptr);
  clGetEventInfo(barrier3, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus3, nullptr);
  clGetEventInfo(barrier4, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus4, nullptr);
  
  std::cout << "Event statuses:" << std::endl;
  std::cout << "  barrier1 (callback trigger): " << eventStatus1 << " (CL_COMPLETE=" << CL_COMPLETE << ")" << std::endl;
  std::cout << "  barrier2 (waits for callbackEvent1): " << eventStatus2 << " (CL_COMPLETE=" << CL_COMPLETE << ")" << std::endl;
  std::cout << "  barrier3 (callback trigger): " << eventStatus3 << " (CL_COMPLETE=" << CL_COMPLETE << ")" << std::endl;
  std::cout << "  barrier4 (waits for callbackEvent2): " << eventStatus4 << " (CL_COMPLETE=" << CL_COMPLETE << ")" << std::endl;
  
  cl_int userEvent1Status, userEvent2Status;
  clGetEventInfo(callbackEvent1, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &userEvent1Status, nullptr);
  clGetEventInfo(callbackEvent2, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &userEvent2Status, nullptr);
  std::cout << "  callbackEvent1 (user event): " << userEvent1Status << " (CL_COMPLETE=" << CL_COMPLETE << ")" << std::endl;
  std::cout << "  callbackEvent2 (user event): " << userEvent2Status << " (CL_COMPLETE=" << CL_COMPLETE << ")" << std::endl;
  
  void* dev_ptr = clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(int), 0);
  if (dev_ptr == nullptr) {
    std::cerr << "Failed to allocate SVM memory" << std::endl;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  int* host_ptr = new int;
  *host_ptr = 0;
  int* dev_data = static_cast<int*>(dev_ptr);
  *dev_data = 42;
  
  std::cout << "About to enqueue memory copy from device to host..." << std::endl;
  std::cout << "This may hang if barriers (barrier2/barrier4) haven't properly completed!" << std::endl;
  
  cl_event memcpyEvent = nullptr;
  cl_event waitForBarriers[] = {barrier2, barrier4};
  err = clEnqueueSVMMemcpy(queue, CL_FALSE, host_ptr, dev_data, sizeof(int), 
                           2, waitForBarriers, &memcpyEvent);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue memory copy" << std::endl;
    clSVMFree(context, dev_ptr);
    delete host_ptr;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  std::cout << "Memory copy enqueued. About to call clFinish - THIS MAY HANG!" << std::endl;
  
  err = clFinish(queue);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to finish queue after memory copy" << std::endl;
    clReleaseEvent(memcpyEvent);
    clSVMFree(context, dev_ptr);
    delete host_ptr;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  std::cout << "Memory copy completed successfully!" << std::endl;
  
  if (*host_ptr != 42) {
    std::cerr << "FAIL: Memory copy failed. Expected 42, got " << *host_ptr << std::endl;
    clReleaseEvent(memcpyEvent);
    clSVMFree(context, dev_ptr);
    delete host_ptr;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  clReleaseEvent(memcpyEvent);
  clSVMFree(context, dev_ptr);
  delete host_ptr;
  
  if (callbackCount.load() != 2) {
    std::cerr << "FAIL: Expected 2 callbacks, got " << callbackCount.load() << std::endl;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  if (order1 != 1 || order2 != 2) {
    std::cerr << "FAIL: Execution order incorrect. order1=" << order1 << ", order2=" << order2 << std::endl;
    clReleaseEvent(barrier4);
    clReleaseEvent(callbackEvent2);
    clReleaseEvent(barrier3);
    clReleaseEvent(barrier2);
    clReleaseEvent(callbackEvent1);
    clReleaseEvent(barrier1);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 1;
  }
  
  clReleaseEvent(barrier4);
  clReleaseEvent(callbackEvent2);
  clReleaseEvent(barrier3);
  clReleaseEvent(barrier2);
  clReleaseEvent(callbackEvent1);
  clReleaseEvent(barrier1);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  
  std::cout << "PASS" << std::endl;
  return 0;
}