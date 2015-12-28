/*
 * OpenCLInterface.cpp
 *
 *  Created on: Nov 26, 2015
 *      Author: jonas
 */

#include "OpenCLInterface.h"
#include "Logger.h"

std::shared_ptr<OpenCLInterface> OpenCLInterface::instance = nullptr;

std::shared_ptr<OpenCLInterface> OpenCLInterface::getInstance() {
	if (instance == nullptr) {
		instance = std::shared_ptr<OpenCLInterface>(new OpenCLInterface());
	}
	return instance;
}

bool OpenCLInterface::isInitialized() const {
	return initialized;
}

OpenCLInterface::OpenCLError OpenCLInterface::initialize(cl_device_type device_type) {
	if (!initialized) {
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() < 1) {
			Logger::writeLine("OpenCLInterface::initialize(): Error: No OpenCL platform found.");
			return OpenCLInterface::OpenCLError::NO_PLATFORM_FOUND;
		} else {
			cl_int error = CL_SUCCESS;
			Logger::writeLine("OpenCLInterface::initialize(): Found " + std::to_string(platforms.size()) + " OpenCL platforms. Using the first with requested devices.");
			unsigned int selected_platform_index = 0;
			std::vector<cl::Device> devices;
			while ((devices.size() < 1) && (selected_platform_index < platforms.size())) {
				Logger::writeLine("OpenCLInterface::initialize(): Trying platform " + std::to_string(selected_platform_index) + ".");
				printOCLPlatformInfo(platforms[selected_platform_index]);
				error = platforms[selected_platform_index].getDevices(device_type, &devices);
				if (error != CL_SUCCESS) {
					Logger::writeLine("OpenCLInterface::initialize(): Error while getting device IDs for platform " + std::to_string(selected_platform_index) + ": " + std::to_string(error));
					selected_platform_index++;
				} else if (devices.size() < 1) {
					Logger::writeLine("OpenCLInterface::initialize(): No devices of the requested type on platform" + std::to_string(selected_platform_index) + ".");
					selected_platform_index++;
				}
			}
			if (devices.size() < 1) {
				Logger::writeLine("OpenCLInterface::initialize(): Error: No OpenCL devices of the requested type found.");
				return OpenCLInterface::OpenCLError::NO_DEVICE_FOUND;
			} else {
				Logger::writeLine("OpenCLInterface::initialize(): Found " + std::to_string(devices.size()) + " OpenCL devices. Using the first.");
				printOCLDeviceInfo(devices[0]);
				device = devices[0];
				devices.resize(1);
				context = cl::Context(devices);
				if (error != CL_SUCCESS) {
					Logger::writeLine("OpenCLInterface::initialize(): Unable to initialize OpenCL context.");
				} else {
					queue = cl::CommandQueue(context, device, 0, &error);
					if (error != CL_SUCCESS) {
						Logger::writeLine("OpenCLInterface::initialize(): Unable to initialize OpenCL command queue.");
					} else {
						initialized = true;
						Logger::writeLine("OpenCLInterface::initialize(): OpenCL system successfully initialized.");
					}
				}
			}
		}
	}
	return OpenCLInterface::OpenCLError::SUCCESS;
}

void OpenCLInterface::printOCLPlatformInfo(const cl::Platform &platform) const {
	std::string platform_name;
	std::string platform_vendor;
	cl_int error;
	error = platform.getInfo(CL_PLATFORM_NAME, &platform_name);
	if (error != CL_SUCCESS) {
		Logger::writeLine("OpenCLInterface::printOCLPlatformInfo(): Error during query for CL_PLATFORM_NAME: " + std::to_string(error));
	}
	error = platform.getInfo(CL_PLATFORM_VENDOR, &platform_vendor);
	if (error != CL_SUCCESS) {
		Logger::writeLine("OpenCLInterface::printOCLPlatformInfo(): Error during query for CL_PLATFORM_VENDOR: " + std::to_string(error));
	}
	Logger::writeLine("OpenCLInterface::printOCLPlatformInfo(): OpenCL platform info: " + platform_name + ", vendor " + platform_vendor + ".");
}

void OpenCLInterface::printOCLDeviceInfo(const cl::Device &device) const {
	std::string device_name;
	std::string device_vendor;
	cl_int error;
	error = device.getInfo(CL_DEVICE_NAME, &device_name);
	if (error != CL_SUCCESS) {
		Logger::writeLine("OpenCLInterface::printOCLDeviceInfo(): Error during query for CL_DEVICE_NAME: " + std::to_string(error));
	}
	error = device.getInfo(CL_DEVICE_VENDOR, &device_vendor);
	if (error != CL_SUCCESS) {
		Logger::writeLine("OpenCLInterface::printOCLDeviceInfo(): Error during query for CL_DEVICE_VENDOR: " + std::to_string(error));
	}
	Logger::writeLine("OpenCLInterface::printOCLDeviceInfo(): OpenCL device info: " + device_name + ", vendor " + device_vendor + ".");
}

int OpenCLInterface::allocateMemoryObject(void *data, size_t size, cl_mem_flags flags) {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::allocateMemoryObject(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		cl_int error;
		cl::Buffer newbuffer = cl::Buffer(context, flags, size, data, &error);
		if (error != CL_SUCCESS) {
			Logger::writeLine("OpenCLInterface::allocateMemoryObject(): Unable to allocate memory object: " + std::to_string(error));
			return OpenCLInterface::OpenCLError::BUFFER_ERROR;
		} else {
			if (free_memids.size() < 1) {
				memory_objects.push_back(newbuffer);
				return (memory_objects.size() - 1);
			} else {
				int memid = *(free_memids.begin());
				free_memids.erase(free_memids.begin());
				memory_objects[memid] = newbuffer;
				return memid;
			}
		}
	}
}

OpenCLInterface::OpenCLError OpenCLInterface::freeMemoryObject(int memid) {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::freeMemoryObject(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		if ((memid < memory_objects.size()) && (free_memids.find(memid) == free_memids.end())) {
			if (memid == memory_objects.size() - 1) memory_objects.pop_back();
			else {
				memory_objects[memid] = cl::Buffer();
				free_memids.insert(memid);
			}
			return OpenCLInterface::OpenCLError::SUCCESS;
		} else {
			Logger::writeLine("OpenCLInterface::freeMemoryObject(): Invalid memory id.");
			return OpenCLInterface::OpenCLError::INVALID_MEMORY_ID;
		}
	}
}

OpenCLInterface::OpenCLError OpenCLInterface::getMemoryContent(int memid, void *data, size_t size) const {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::getMemoryContent(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		if ((memid < memory_objects.size()) && (free_memids.find(memid) == free_memids.end())) {
			cl_int error = queue.enqueueReadBuffer(memory_objects[memid], true, 0, size, data, NULL, NULL);
			if (error != CL_SUCCESS) {
				Logger::writeLine("OpenCLInterface::getMemoryContent(): Unable to get memory content: " + std::to_string(error));
			} else {
				return OpenCLInterface::OpenCLError::SUCCESS;
			}
		} else {
			Logger::writeLine("OpenCLInterface::getMemoryContent(): Invalid memory id.");
			return OpenCLInterface::OpenCLError::INVALID_MEMORY_ID;
		}
	}
}

OpenCLInterface::OpenCLError OpenCLInterface::writeMemoryContent(int memid, void *data, size_t size) const {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::writeMemoryContent(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		if ((memid < memory_objects.size()) && (free_memids.find(memid) == free_memids.end())) {
			cl_int error = queue.enqueueWriteBuffer(memory_objects[memid], true, 0, size, data, NULL, NULL);
			if (error != CL_SUCCESS) {
				Logger::writeLine("OpenCLInterface::writeMemoryContent(): Unable to write memory content.");
			} else {
				return OpenCLInterface::OpenCLError::SUCCESS;
			}
		} else {
			Logger::writeLine("OpenCLInterface::writeMemoryContent(): Invalid memory id.");
			return OpenCLInterface::OpenCLError::INVALID_MEMORY_ID;
		}
	}
}

int OpenCLInterface::createKernelFromSource(std::string code, std::string name) {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::createKernelFromSource(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		cl_int error;
		std::vector<std::pair<const char *, size_t>> source;
		source.push_back(std::make_pair(code.c_str(), code.length()));
		cl::Program program(context, source, &error);
		if (error != CL_SUCCESS) {
			Logger::writeLine("OpenCLInterface::createKernelFromSource(): Unable to create program.");
			return OpenCLInterface::OpenCLError::PROGRAM_ERROR;
		} else {
			error = program.build(std::vector<cl::Device>(1, device), NULL, NULL, NULL);
			if (error != CL_SUCCESS) {
				std::string build_log;
				program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
				Logger::writeLine("OpenCLInterface::createKernelFromSource(): Failed to build program: " + build_log);
				return OpenCLInterface::OpenCLError::COMPILATION_ERROR;
			} else {
				cl::Kernel kernel(program, name.c_str(), &error);
				if (error != CL_SUCCESS) {
					Logger::writeLine("OpenCLInterface::createKernelFromSource(): Unable to create kernel.");
					return OpenCLInterface::OpenCLError::KERNEL_ERROR;
				} else {
					if (free_kids.size() < 1) {
						kernel_objects.push_back(kernel);
						return (kernel_objects.size() - 1);
					} else {
						int kid = *(free_kids.begin());
						free_kids.erase(free_kids.begin());
						kernel_objects[kid] = kernel;
						return kid;
					}
				}
			}
		}
	}
}

OpenCLInterface::OpenCLError OpenCLInterface::deleteKernel(int kid) {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::deleteKernel(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		if ((kid < kernel_objects.size()) && (free_kids.find(kid) == free_kids.end())) {
			if (kid == kernel_objects.size() - 1) kernel_objects.pop_back();
			else {
				kernel_objects[kid] = cl::Kernel();
				free_kids.insert(kid);
			}
			return OpenCLInterface::OpenCLError::SUCCESS;
		} else {
			Logger::writeLine("OpenCLInterface::deleteKernel(): Invalid kernel id.");
			return OpenCLInterface::OpenCLError::INVALID_KERNEL_ID;
		}
	}
}

OpenCLInterface::OpenCLError OpenCLInterface::callKernel(int kid, OpenCLInterface::Dimension range, const std::vector<int> &memids,
						const std::vector<std::pair<void *, size_t>> &args) {
	if (!initialized) {
		Logger::writeLine("OpenCLInterface::callKernel(): OpenCL system was not initialized.");
		return OpenCLInterface::OpenCLError::NOT_INITIALIZED;
	} else {
		if (kid < kernel_objects.size() && (free_kids.find(kid) == free_kids.end())) {
			cl::NDRange ndrange;
			cl::Event finish;
			if (range.z != 0) {
				ndrange = cl::NDRange(range.x, range.y, range.z);
			} else if (range.y != 0) {
				ndrange = cl::NDRange(range.x, range.y);
			} else if (range.x != 0) {
				ndrange = cl::NDRange(range.x);
			} else {
				Logger::writeLine("OpenCLInterface::callKernel(): NDRange contains no work-items.");
				return OpenCLInterface::OpenCLError::NO_WORKITEMS;
			}
			for (unsigned int i = 0; i < memids.size(); i++) {
				if ((memids[i] < memory_objects.size()) && (free_memids.find(memids[i]) == free_memids.end())) {
					kernel_objects[kid].setArg<cl::Buffer>(i, memory_objects[memids[i]]);
				} else {
					Logger::writeLine("OpenCLInterface::callKernel(): Invalid memory id in kernel arguments.");
					return OpenCLInterface::OpenCLError::INVALID_MEMORY_ID;
				}
			}
			for (unsigned int i = 0; i < args.size(); i++) {
				kernel_objects[kid].setArg(memids.size() + i, args[i].second, args[i].first);
			}
			cl_int error = queue.enqueueNDRangeKernel(kernel_objects[kid], cl::NullRange, ndrange, cl::NullRange, NULL, &finish);
			if (error != CL_SUCCESS) {
				Logger::writeLine("OpenCLInterface::callKernel(): Could not enqueue NDRange kernel: " + std::to_string(error));
				return OpenCLInterface::OpenCLError::KERNEL_ERROR;
			} else {
				cl::WaitForEvents(std::vector<cl::Event>(1, finish));
				return OpenCLInterface::OpenCLError::SUCCESS;
			}
		} else {
			Logger::writeLine("OpenCLInterface::callKernel(): Invalid kernel ID.");
			return OpenCLInterface::OpenCLError::INVALID_KERNEL_ID;
		}
	}
}

OpenCLInterface::OpenCLInterface() {
}

OpenCLInterface::~OpenCLInterface() {
	// TODO Auto-generated destructor stub
}

