#define ZE_CHECK(myZeCall)                                            \
  if (myZeCall != ZE_RESULT_SUCCESS) {                                    \
    std::cout << "Error at " << #myZeCall << ": " << __FUNCTION__ << ": " \
              << __LINE__ << std::endl;                                   \
    std::cout << "Exit with Error Code: "                                 \
              << "0x" << std::hex << myZeCall << std::dec << std::endl;   \
    std::terminate();                                                     \
  }
