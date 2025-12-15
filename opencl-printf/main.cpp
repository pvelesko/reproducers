// IGC bug: SPIR-V printf with UniformConstant pointer as %s argument fails
// Error: "Invalid record (Producer: 'LLVM16.0.6' Reader: 'LLVM 16.0.6')"

#include <CL/cl.h>
#include <cstdio>
#include <fstream>
#include <vector>

std::vector<char> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<char> buf(f.tellg());
    f.seekg(0);
    f.read(buf.data(), buf.size());
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <spirv-file>\n", argv[0]);
        printf("  strings_bad.spv  - FAILS (printf with %%s and UniformConstant ptr)\n");
        printf("  dynamic_good.spv - WORKS (printf without UniformConstant ptr args)\n");
        return 1;
    }

    cl_platform_id plat;
    cl_device_id dev;
    clGetPlatformIDs(1, &plat, nullptr);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);

    char name[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    printf("Device: %s\n", name);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);

    auto spv = readFile(argv[1]);
    if (spv.empty()) { printf("Cannot read %s\n", argv[1]); return 1; }

    cl_program prog = clCreateProgramWithIL(ctx, spv.data(), spv.size(), &err);
    if (err) { printf("clCreateProgramWithIL failed: %d\n", err); return 1; }

    err = clCompileProgram(prog, 1, &dev, "-cl-std=CL3.0", 0, nullptr, nullptr, nullptr, nullptr);
    if (err) { printf("clCompileProgram failed: %d\n", err); return 1; }

    cl_program linked = clLinkProgram(ctx, 1, &dev, "", 1, &prog, nullptr, nullptr, &err);
    if (err) {
        size_t len;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::vector<char> log(len);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        printf("clLinkProgram FAILED: %d\n%s\n", err, log.data());
        return 1;
    }

    printf("SUCCESS\n");
    clReleaseProgram(linked);
    clReleaseProgram(prog);
    clReleaseContext(ctx);
    return 0;
}
