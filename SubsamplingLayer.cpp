/*
 * SubsamplingLayer.cpp
 *
 *  Created on: Dec 26, 2015
 *      Author: jonas
 */

#include "SubsamplingLayer.h"
#include "RandomGenerator.h"
#include "Logger.h"

namespace clneural {

const std::string SubsamplingLayer::fwclcode = "__kernel void computeOutput(__global const float *inputs, __global const float *weights, \n"
		"__global float *outputs, __global float *netsums,\n"
		"unsigned int inp_width, unsigned int inp_height, unsigned int filter_width, unsigned int filter_height) {\n"
		"unsigned int output_id = get_global_id(0);\n"
		"unsigned int output_feature_map_size = ((inp_width + filter_width - 1) / filter_width) * ((inp_height + filter_height - 1) / filter_height);\n"
		"unsigned int input_feature_map_size = inp_width * inp_height;\n"
		"unsigned int feature_map_id = output_id / output_feature_map_size;\n"
		"unsigned int output_x = (output_id % output_feature_map_size) % ((inp_width + filter_width - 1) / filter_width);\n"
		"unsigned int output_y = (output_id % output_feature_map_size) / ((inp_width + filter_width - 1) / filter_width);\n"
		"float sum = 0.0f;\n"
		"for (unsigned int inp_y = output_y * filter_height; inp_y < (output_y + 1) * filter_height && inp_y < inp_width; inp_y++) {\n"
		"for (unsigned int inp_x = output_x * filter_width; inp_x < (output_x + 1) * filter_width && inp_x < inp_width; inp_x++) {\n"
		"sum += inputs[feature_map_id * input_feature_map_size + inp_y * inp_width + inp_x];\n"
		"}\n"
		"}\n"
		"sum /= filter_width * filter_height;\n"
		"netsums[output_id] = sum;\n"
		"sum *= weights[2 * feature_map_id];\n"
		"sum += weights[2 * feature_map_id + 1];\n"
		"outputs[output_id] = activationFunction(sum);\n"
		"}\n";

const std::string SubsamplingLayer::fberrorclcode = "__kernel void computeNextError(__global const float *error, __global const float *netsums,\n"
		"__global const float *weights, __global float *nexterror, \n"
		"unsigned int inp_width, unsigned int inp_height, unsigned int filter_width, unsigned int filter_height) {\n"
		"unsigned int input_id = get_global_id(0);\n"
		"unsigned int input_feature_map_size = inp_width * inp_height;\n"
		"unsigned int output_feature_map_size = ((inp_width + filter_width - 1) / filter_width) * ((inp_height + filter_height - 1) / filter_height);\n"
		"unsigned int feature_map_id = input_id / input_feature_map_size;\n"
		"unsigned int inp_x = (input_id % input_feature_map_size) % inp_width;\n"
		"unsigned int inp_y = (input_id % input_feature_map_size) / inp_width;\n"
		"unsigned int output_id = feature_map_id * output_feature_map_size + (inp_y / filter_height) * ((inp_width + filter_width - 1) / filter_width) + inp_x / filter_width;"
		"float delta = activationDerivate(netsums[output_id] * weights[2 * feature_map_id] + weights[2 * feature_map_id + 1]) * error[output_id];"
		"nexterror[input_id] = weights[2 * feature_map_id] * delta;\n"
		"}\n";

const std::string SubsamplingLayer::fbweightsclcode = "__kernel void computeWeights(__global const float *error, \n"
		"__global const float *netsums, __global float *weights, \n"
		"unsigned int inp_width, unsigned int inp_height, unsigned int filter_width, unsigned int filter_height, float learning_rate) {\n"
		"unsigned int feature_map_id = get_global_id(0);\n"
		"unsigned int output_feature_map_size = ((inp_width + filter_width - 1)/filter_width) * ((inp_height + filter_height - 1)/filter_height);\n"
		"float delta = 0.0f;\n"
		"float delta_bias = 0.0f;\n"
		"for (unsigned int output_y = 0; output_y < ((inp_height + filter_height - 1)/filter_height); output_y++) {\n"
		"for (unsigned int output_x = 0; output_x < ((inp_width + filter_width - 1)/filter_width); output_x++) {\n"
		"unsigned int output_id = feature_map_id * output_feature_map_size + output_y * ((inp_width + filter_width - 1)/filter_width) + output_x;\n"
		"delta += learning_rate * error[output_id] * activationDerivate(netsums[output_id] * weights[2 * feature_map_id] + weights[2*feature_map_id + 1]) * netsums[output_id];\n"
		"delta_bias += learning_rate * error[output_id] * activationDerivate(netsums[output_id] * weights[2 * feature_map_id] + weights[2*feature_map_id + 1]);\n"
		"}\n"
		"}\n"
		"weights[2 * feature_map_id] += delta;\n"
		"weights[2 * feature_map_id + 1] += delta_bias;\n"
		"}\n";

const NeuralNetworkLayerRegisterHelper<SubsamplingLayer> SubsamplingLayer::reg("SubsamplingLayer");

SubsamplingLayer::SubsamplingLayer(Dimension input_maps, Dimension filter, unsigned int num_feature_maps, std::shared_ptr<ActivationFunction> act, float learning) :
	learning(learning),
	act(act),
	input_maps(input_maps),
	filter(filter),
	num_feature_maps(num_feature_maps) {
	unsigned int output_feature_map_size = ((input_maps.width + filter.width - 1)/filter.width) * ((input_maps.height + filter.height - 1)/filter.height);
	num_inputs = num_feature_maps * input_maps.width * input_maps.height;
	num_outputs = num_feature_maps * output_feature_map_size;
	for (unsigned int i = 0; i < 2*num_feature_maps; i++) {
		weights.push_back(RandomGenerator::getRandomNumber(-0.5f, 0.5f));
	}
}

bool SubsamplingLayer::initializeMemoryObjects(std::shared_ptr<OpenCLInterface> ocl) {
	if (wmemid < 0) {
		wmemid = ocl->allocateMemoryObject((void *) &weights[0], weights.size() * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
	}
	if (imemid < 0) {
		imemid = ocl->allocateMemoryObject(NULL, num_inputs * sizeof(float), CL_MEM_READ_WRITE);
	}
	if (oememid < 0) {
		oememid = ocl->allocateMemoryObject(NULL, num_outputs * sizeof(float), CL_MEM_WRITE_ONLY);
	}
	if (smemid < 0) {
		smemid = ocl->allocateMemoryObject(NULL, num_outputs * sizeof(float), CL_MEM_READ_WRITE);
	}
	if (nememid < 0) {
		nememid = ocl->allocateMemoryObject(NULL, num_inputs * sizeof(float), CL_MEM_WRITE_ONLY);
	}
	if ((wmemid < 0) || (imemid < 0) || (oememid < 0) || (smemid < 0) || (nememid < 0)) {
		return false;
	}
	return true;
}

bool SubsamplingLayer::initializeKernelObjects(std::shared_ptr<OpenCLInterface> ocl) {
	if (okid < 0) {
		std::string code = act->getCode() + fwclcode;
		okid = ocl->createKernelFromSource(code, "computeOutput");
	}
	if (fberrorkid < 0) {
		std::string code = act->getDerivCode() + fberrorclcode;
		fberrorkid = ocl->createKernelFromSource(code, "computeNextError");
	}
	if (fbweightskid < 0) {
		std::string code = act->getDerivCode() + fbweightsclcode;
		fbweightskid = ocl->createKernelFromSource(code, "computeWeights");
	}
	if ((okid < 0) || (fberrorkid < 0) || (fbweightskid < 0)) {
		return false;
	}
	return true;
}

std::vector<float> SubsamplingLayer::computeOutput(const std::vector<float> &input) {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (ocl->isInitialized()) {
		if (!initializeKernelObjects(ocl)) {
			Logger::writeLine("SubsamplingLayer::computeOutput(): Can't initialize kernel. Unable to compute anything.");
			return input;
		} else if (!initializeMemoryObjects(ocl)) {
			Logger::writeLine("SubsamplingLayer::computeOutput(): Can't initialize memory objects. Unable to compute anything.");
			return input;
		} else {
			std::vector<float> output(num_outputs);
			ocl->writeMemoryContent(imemid, (void*) &input[0], num_inputs * sizeof(float));
			OpenCLInterface::Dimension dim;
			dim.x = num_outputs;
			std::vector<int> memargs({imemid, wmemid, oememid, smemid});
			std::vector<std::pair<void *, size_t>> constargs({std::make_pair((void *) &input_maps.width, sizeof(unsigned int)),
															std::make_pair((void*) &input_maps.height, sizeof(unsigned int)),
															std::make_pair((void*) &filter.width, sizeof(unsigned int)),
															std::make_pair((void*) &filter.height, sizeof(unsigned int))});
			OpenCLInterface::OpenCLError err = ocl->callKernel(okid, dim, memargs, constargs);
			if (err != OpenCLInterface::OpenCLError::SUCCESS) {
				Logger::writeLine("SubsamplingLayer::computeOutput(): Error when calling the OpenCL kernel.");
				return input;
			} else {
				ocl->getMemoryContent(oememid, (void *) &output[0], num_outputs * sizeof(float));
				return output;
			}
		}
	} else {
		Logger::writeLine("SubsamplingLayer::computeOutput(): OpenCLInterface not initialized. Unable to compute anything.");
		return input;
	}
}

std::vector<float> SubsamplingLayer::computeError(const std::vector<float> &input) {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (ocl->isInitialized()) {
		if (!initializeKernelObjects(ocl)) {
			Logger::writeLine("SubsamplingLayer::computeError(): Can't initialize kernel. Unable to compute anything.");
			return input;
		} else if (!initializeMemoryObjects(ocl)) {
			Logger::writeLine("SubsamplingLayer::computeError(): Can't initialize memory objects. Unable to compute anything.");
			return input;
		} else {
			std::vector<float> newerror(num_inputs);
			ocl->writeMemoryContent(oememid, (void*) &input[0], num_outputs * sizeof(float));
			OpenCLInterface::Dimension dim;
			dim.x = num_inputs;
			std::vector<int> memargs({oememid, smemid, wmemid, nememid});
			std::vector<std::pair<void *, size_t>> constargs({std::make_pair((void *) &input_maps.width, sizeof(unsigned int)),
															std::make_pair((void*) &input_maps.height, sizeof(unsigned int)),
															std::make_pair((void*) &filter.width, sizeof(unsigned int)),
															std::make_pair((void*) &filter.height, sizeof(unsigned int))});
			OpenCLInterface::OpenCLError err = ocl->callKernel(fberrorkid, dim, memargs, constargs);
			if (err != OpenCLInterface::OpenCLError::SUCCESS) {
				Logger::writeLine("SubsamplingLayer::computeError(): Error when calling the OpenCL kernel for next error computation.");
				return input;
			} else {
				ocl->getMemoryContent(nememid, (void *) &newerror[0], num_inputs * sizeof(float));
				constargs.push_back(std::make_pair((void *) &learning, sizeof(float)));
				memargs.pop_back();
				dim.x = num_feature_maps;
				err = ocl->callKernel(fbweightskid, dim, memargs, constargs);
				if (err != OpenCLInterface::OpenCLError::SUCCESS) {
					Logger::writeLine("SubsamplingLayer::computeError(): Error when calling the OpenCL kernel for weights computation.");
					return input;
				}
				ocl->getMemoryContent(wmemid, (void *) &weights[0], weights.size() * sizeof(float));
				return newerror;
			}
		}
	} else {
		Logger::writeLine("SubsamplingLayer::computeError(): OpenCLInterface not initialized. Unable to compute anything.");
		return input;
	}
}

std::string SubsamplingLayer::getName() const {
	return "SubsamplingLayer";
}

std::string SubsamplingLayer::getDatastring() const {
	std::string datastring = act->getName() + ":";
	datastring += std::to_string(learning) + ":" + std::to_string(num_feature_maps) + ":";
	datastring += std::to_string(input_maps.width) + ":" + std::to_string(input_maps.height) + ":";
	datastring += std::to_string(filter.width) + ":" + std::to_string(filter.height) + ":";
	datastring += getVectorRepresentation<float>(weights, ';');
	return datastring;
}

bool SubsamplingLayer::parseDatastring(std::string datastring) {
	std::vector<std::string> data = parseVectorRepresentation<std::string> (datastring, ':');
	if (data.size() != 8) {
		Logger::writeLine("SubsamplingLayer::parseDatastring(): Invalid number of parameters." + std::to_string(data.size()));
		return false;
	} else {
		act = ActivationFunction::getObjectFromString(data[0]);
		if (act == nullptr) {
			Logger::writeLine("SubsamplingLayer::parseDatastring(): Invalid activation function identifier: " + data[0]);
			return false;
		} else {
			learning = std::stof(data[1]);
			num_feature_maps = std::stoul(data[2]);
			input_maps.width = std::stoul(data[3]);
			input_maps.height = std::stoul(data[4]);
			num_inputs = input_maps.width * input_maps.height * num_feature_maps;
			filter.width = std::stoul(data[5]);
			filter.height = std::stoul(data[6]);
			num_outputs = ((input_maps.width + filter.width - 1) / filter.width) * ((input_maps.height + filter.height - 1) / filter.height) * num_feature_maps;
			weights = parseVectorRepresentation<float>(data[7], ';');
		}
	}
	return true;
}

SubsamplingLayer::~SubsamplingLayer() {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (wmemid > 0) {
		ocl->freeMemoryObject(wmemid);
	}
	if (imemid > 0) {
		ocl->freeMemoryObject(imemid);
	}
	if (oememid > 0) {
		ocl->freeMemoryObject(oememid);
	}
	if (smemid > 0) {
		ocl->freeMemoryObject(smemid);
	}
	if (nememid > 0) {
		ocl->freeMemoryObject(nememid);
	}
	if (okid > 0) {
		ocl->deleteKernel(okid);
	}
}

} /* namespace clneural */
