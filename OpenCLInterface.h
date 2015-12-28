/*
 * OpenCLInterface.h
 *
 *  Created on: Nov 26, 2015
 *      Author: jonas
 */

#ifndef OPENCLINTERFACE_H_
#define OPENCLINTERFACE_H_

#include <CL/cl.hpp>
#include <memory>
#include <vector>
#include <unordered_set>

class OpenCLInterface {
private:
	OpenCLInterface();
	cl::Context context;
	cl::Device device;
	cl::CommandQueue queue;
	std::vector<cl::Buffer> memory_objects;
	std::unordered_set<int> free_memids;
	std::vector<cl::Kernel> kernel_objects;
	std::unordered_set<int> free_kids;
	bool initialized = false;
	static std::shared_ptr<OpenCLInterface> instance;
	void printOCLDeviceInfo(const cl::Device &device) const;
	void printOCLPlatformInfo(const cl::Platform &platform) const;
public:
	enum OpenCLError {
		NO_PLATFORM_FOUND = -1,
		NO_DEVICE_FOUND = -2,
		CONTEXT_ERROR = -3,
		COMPILATION_ERROR = -4,
		INVALID_MEMORY_ID = -5,
		INVALID_DIMENSION = -6,
		KERNEL_ERROR = -7,
		PROGRAM_ERROR = -8,
		NOT_INITIALIZED = -9,
		BUFFER_ERROR = -10,
		INVALID_KERNEL_ID = -11,
		NO_WORKITEMS = -12,
		SUCCESS = 0
	};
	struct Dimension {
		size_t x = 0;
		size_t y = 0;
		size_t z = 0;
	};
	static std::shared_ptr<OpenCLInterface> getInstance();
	OpenCLError initialize(cl_device_type device_type);
	bool isInitialized() const;
	int allocateMemoryObject(void *data, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
	OpenCLError getMemoryContent(int memid, void *data, size_t size) const;
	OpenCLError writeMemoryContent(int memid, void *data, size_t size) const;
	OpenCLError freeMemoryObject(int memid);
	int createKernelFromSource(std::string code, std::string name);
	OpenCLError deleteKernel(int kid);
	OpenCLError callKernel(int kid, Dimension range, const std::vector<int> &memids,
						const std::vector<std::pair<void *, size_t>> &args);
	virtual ~OpenCLInterface();
};

#endif /* OPENCLINTERFACE_H_ */
