/*
 * ConvolutionalLayer.cpp
 *
 *  Created on: Dec 12, 2015
 *      Author: jonas
 */

#include "ConvolutionalLayer.h"
#include "RandomGenerator.h"
#include "OpenCLInterface.h"
#include "Logger.h"
#include <algorithm>
#include <iostream>

namespace clneural {

const std::string ConvolutionalLayer::fwclcode = "__kernel void computeOutput(__global const float *inputs, __global const float *weights, \n"
		"__global float *outputs, __global float *netsums, __global const unsigned int *input_connections, __global const unsigned int *input_connection_indices, \n"
		"unsigned int inp_width, unsigned int inp_height, unsigned int filter_width, unsigned int filter_height) {\n"
		"unsigned int output_id = get_global_id(0);\n"
		"unsigned int output_feature_map_size = (inp_width - filter_width + 1) * (inp_height - filter_height + 1);\n"
		"unsigned int input_feature_map_size = inp_width * inp_height;\n"
		"unsigned int output_feature_map_id = output_id / output_feature_map_size;\n"
		"unsigned int output_x = (output_id % output_feature_map_size) % (inp_width - filter_width + 1);\n"
		"unsigned int output_y = (output_id % output_feature_map_size) / (inp_width - filter_width + 1);\n"
		"float sum = 0.0f;\n"
		"for (unsigned int i = 0; i < input_connection_indices[output_feature_map_id + 1] - input_connection_indices[output_feature_map_id]; i++) {\n"
		"unsigned int input_feature_map_id = input_connections[input_connection_indices[output_feature_map_id] + i];\n"
		"for (unsigned int y = 0; y < filter_height; y++) {\n"
		"for (unsigned int x = 0; x < filter_width; x++) {\n"
		"unsigned int inp_x = output_x + x;\n"
		"unsigned int inp_y = output_y + y;\n"
		"sum += inputs[input_feature_map_id * input_feature_map_size + inp_y * inp_width + inp_x] * weights[(input_connection_indices[output_feature_map_id] + i) * (filter_width * filter_height) + output_feature_map_id + y*filter_width + x];\n"
		"}\n"
		"}\n"
		"}\n"
		"sum += weights[input_connection_indices[output_feature_map_id + 1]*(filter_width*filter_height) + output_feature_map_id];\n"
		"netsums[output_id] = sum;\n"
		"outputs[output_id] = activationFunction(sum);\n"
		"}\n";

const std::string ConvolutionalLayer::fberrorclcode = "__kernel void computeNextError(__global const float *error, __global const float *netsums,\n"
		"__global const float *weights, __global float *nexterror, __global const unsigned int *output_connections, \n"
		"__global const unsigned int *output_connection_indices, __global const unsigned int *output_weight_indices, \n"
		"unsigned int inp_width, unsigned int inp_height, unsigned int filter_width, unsigned int filter_height) {\n"
		"unsigned int input_id = get_global_id(0);\n"
		"unsigned int input_feature_map_size = inp_width * inp_height;\n"
		"unsigned int output_feature_map_size = (inp_width - filter_width + 1) * (inp_height - filter_height + 1);\n"
		"unsigned int input_feature_map_id = input_id / input_feature_map_size;\n"
		"unsigned int inp_x = (input_id % input_feature_map_size) % inp_width;\n"
		"unsigned int inp_y = (input_id % input_feature_map_size) / inp_width;\n"
		"float sum = 0.0f;\n"
		"for (unsigned int i = 0; i < output_connection_indices[input_feature_map_id + 1] - output_connection_indices[input_feature_map_id]; i++) {\n"
		"unsigned int output_feature_map_id = output_connections[output_connection_indices[input_feature_map_id] + i];\n"
		"for (int y = filter_height - 1; y >= 0; y--) {\n"
		"for (int x = filter_width - 1; x >= 0; x--) {\n"
		"int output_x = inp_x - x;\n"
		"int output_y = inp_y - y;\n"
		"if ((output_x >= 0) && (output_y >= 0) && (output_x < (inp_width - filter_width + 1)) && (output_y < (inp_height - filter_height + 1))) {\n"
		"unsigned int output_id = output_feature_map_id * output_feature_map_size + output_y * (inp_width - filter_width + 1) + output_x;"
		"float delta = activationDerivate(netsums[output_id]) * error[output_id];\n"
		"sum += weights[output_weight_indices[output_connection_indices[input_feature_map_id] + i] + y * filter_width + x] * delta;\n"
		"}\n"
		"}\n"
		"}\n"
		"}\n"
		"nexterror[input_id] = sum;\n"
		"}\n";

const std::string ConvolutionalLayer::fbweightsclcode = "__kernel void computeWeights(__global const float *error, __global const float *last_inputs, \n"
		"__global const float *netsums, __global float *weights, __global const unsigned int *weight_output_maps, __global const unsigned int *input_connections, \n"
		" __global const unsigned int *input_connection_indices, unsigned int inp_width, unsigned int inp_height, unsigned int filter_width, \n"
		"unsigned int filter_height, float learning_rate) {\n"
		"unsigned int weight_id = get_global_id(0);\n"
		"unsigned int input_feature_map_size = inp_width * inp_height;\n"
		"unsigned int output_feature_map_size = (inp_width - filter_width + 1) * (inp_height - filter_height + 1);\n"
		"unsigned int output_feature_map_id = weight_output_maps[weight_id];\n"
		"int weight_x = -1;\n"
		"int weight_y = -1;\n"
		"unsigned int weight_startindex = filter_height * filter_width * input_connection_indices[output_feature_map_id] + output_feature_map_id;\n"
		"unsigned int input_map_offset = (weight_id % weight_startindex) / (filter_height * filter_width);\n"
		"if (weight_id != input_connection_indices[output_feature_map_id + 1] * filter_height * filter_width + output_feature_map_id){\n"
		"weight_y = ((weight_id - weight_startindex) % (filter_height * filter_width)) / filter_width;\n"
		"weight_x = ((weight_id - weight_startindex) % (filter_height * filter_width)) % filter_width;\n"
		"}\n"
		"float delta = 0.0f;\n"
		"for (unsigned int output_y = 0; output_y < (inp_height - filter_height + 1); output_y++) {\n"
		"for (unsigned int output_x = 0; output_x < (inp_height - filter_height + 1); output_x++) {\n"
		"unsigned int output_id = output_feature_map_id * output_feature_map_size + output_y * (inp_width - filter_width + 1) + output_x;\n"
		"float last_input = 1.0f;\n"
		"if (weight_x > 0) {\n"
		"unsigned int input_id = input_connections[input_connection_indices[output_feature_map_id] + input_map_offset] * input_feature_map_size + (output_y + weight_y) * inp_width + (output_x + weight_x);\n"
		"last_input = last_inputs[input_id];\n"
		"}\n"
		"delta += learning_rate * error[output_id] * activationDerivate(netsums[output_id]) * last_input;\n"
		"}\n"
		"}\n"
		"weights[weight_id] += delta;\n"
		"}\n";


ConvolutionalLayer::ConvolutionalLayer(Dimension input_maps, Dimension filter, const std::vector<std::list<unsigned int>> &input_to_output,
		std::shared_ptr<ActivationFunction> act, float learning) :
			act(act),
			learning(learning),
			input_maps(input_maps),
			filter(filter) {
	num_outputs = (input_maps.width - filter.width + 1) * (input_maps.height - filter.height + 1) * input_to_output.size();
	unsigned int max_index = 0;
	unsigned int counter = 0;
	std::vector<std::list<unsigned int>> output_to_input(1);
	num_output_maps = input_to_output.size();
	for (unsigned int i = 0; i < input_to_output.size(); i++) {
		input_connection_indices.push_back(counter);
		for (std::list<unsigned int>::const_iterator it = input_to_output[i].begin(); it != input_to_output[i].end(); it++) {
			input_connections.push_back(*it);
			if (*it > max_index) {
				max_index = *it;
				output_to_input.resize(*it + 1, std::list<unsigned int>());
			}
			output_to_input[*it].push_back(i);
			counter++;
			for (unsigned int j = 0; j < (filter.height * filter.width); j++) {
				weights.push_back(RandomGenerator::getRandomNumber(-1.0f/(input_to_output[i].size() * filter.width * filter.height + 1), 1.0f/(input_to_output[i].size() * filter.width * filter.height + 1)));
				weight_output_maps.push_back(i);
			}
		}
		weights.push_back(RandomGenerator::getRandomNumber(-1.0f/(input_to_output[i].size() * filter.width * filter.height + 1), 1.0f/(input_to_output[i].size() * filter.width * filter.height + 1))); //For bias
		weight_output_maps.push_back(i);
	}
	input_connection_indices.push_back(counter);
	counter = 0;
	num_input_maps = output_to_input.size();
	for (unsigned int i = 0; i < output_to_input.size(); i++) {
		output_connection_indices.push_back(counter);
		for (std::list<unsigned int>::iterator it = output_to_input[i].begin(); it != output_to_input[i].end(); it++) {
			output_connections.push_back(*it);
			unsigned int windex = input_connection_indices[*it] + std::distance(input_to_output[*it].begin(), std::find(input_to_output[*it].begin(), input_to_output[*it].end(), i));
			output_weight_indices.push_back(windex * (filter.height * filter.width) + *it);
			counter++;
		}
	}
	output_connection_indices.push_back(counter);
	num_inputs = (max_index + 1) * input_maps.width * input_maps.height;
	num_input_maps = output_to_input.size();
}

bool ConvolutionalLayer::initializeMemoryObjects(std::shared_ptr<OpenCLInterface> ocl) {
	if (wmemid < 0) {
		wmemid = ocl->allocateMemoryObject((void *) &weights[0], weights.size() * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
	}
	if (womemid < 0) {
		womemid = ocl->allocateMemoryObject((void *) &weight_output_maps[0],  weight_output_maps.size() * sizeof(unsigned int), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
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
	if (icmemid < 0) {
		icmemid = ocl->allocateMemoryObject((void *) &input_connections[0], input_connections.size() * sizeof(unsigned int), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
	}
	if (ocmemid < 0) {
		ocmemid = ocl->allocateMemoryObject((void *) &output_connections[0], output_connections.size() * sizeof(unsigned int), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
	}
	if (icimemid < 0) {
		icimemid = ocl->allocateMemoryObject((void *) &input_connection_indices[0], input_connection_indices.size() * sizeof(unsigned int), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
	}
	if (ocimemid < 0) {
		ocimemid = ocl->allocateMemoryObject((void *) &output_connection_indices[0], output_connection_indices.size() * sizeof(unsigned int), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
	}
	if (owimemid < 0) {
		owimemid = ocl->allocateMemoryObject((void *) &output_weight_indices[0], output_weight_indices.size() * sizeof(unsigned int), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
	}
	if ((wmemid < 0) || (womemid < 0) || (imemid < 0) || (oememid < 0) || (smemid < 0) || (nememid < 0) || (icmemid < 0) || (ocmemid < 0) || (icimemid < 0) || (ocimemid < 0) || (owimemid < 0)) {
		return false;
	}
	return true;
}

bool ConvolutionalLayer::initializeKernelObjects(std::shared_ptr<OpenCLInterface> ocl) {
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

std::vector<float> ConvolutionalLayer::computeOutput(const std::vector<float> &input) {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (ocl->isInitialized()) {
		if (!initializeKernelObjects(ocl)) {
			Logger::writeLine("ConvolutionalLayer::computeOutput(): Can't initialize kernel. Unable to compute anything.");
			return input;
		} else if (!initializeMemoryObjects(ocl)) {
			Logger::writeLine("ConvolutionalLayer::computeOutput(): Can't initialize memory objects. Unable to compute anything.");
			return input;
		} else {
			std::vector<float> output(num_outputs);
			ocl->writeMemoryContent(imemid, (void*) &input[0], num_inputs * sizeof(float));
			OpenCLInterface::Dimension dim;
			dim.x = num_outputs;
			std::vector<int> memargs({imemid, wmemid, oememid, smemid, icmemid, icimemid});
			std::vector<std::pair<void *, size_t>> constargs({std::make_pair((void *) &input_maps.width, sizeof(unsigned int)),
															std::make_pair((void*) &input_maps.height, sizeof(unsigned int)),
															std::make_pair((void*) &filter.width, sizeof(unsigned int)),
															std::make_pair((void*) &filter.height, sizeof(unsigned int))});
			OpenCLInterface::OpenCLError err = ocl->callKernel(okid, dim, memargs, constargs);
			if (err != OpenCLInterface::OpenCLError::SUCCESS) {
				Logger::writeLine("ConvolutionalLayer::computeOutput(): Error when calling the OpenCL kernel.");
				return input;
			}
			ocl->getMemoryContent(oememid, (void *) &output[0], num_outputs * sizeof(float));
			return output;
		}
	} else {
		Logger::writeLine("ConvolutionalLayer::computeOutput(): OpenCLInterface not initialized. Unable to compute anything.");
		return input;
	}
}

std::vector<float> ConvolutionalLayer::computeError(const std::vector<float> &input) {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (ocl->isInitialized()) {
		if (!initializeKernelObjects(ocl)) {
			Logger::writeLine("ConvolutionalLayer::computeError(): Can't initialize kernel. Unable to compute anything.");
			return input;
		} else if (!initializeMemoryObjects(ocl)) {
			Logger::writeLine("ConvolutionalLayer::computeError(): Can't initialize memory objects. Unable to compute anything.");
			return input;
		} else {
			std::vector<float> newerror(num_inputs);
			ocl->writeMemoryContent(oememid, (void*) &input[0], num_outputs * sizeof(float));
			OpenCLInterface::Dimension dim;
			dim.x = num_inputs;
			std::vector<int> memargs({oememid, smemid, wmemid, nememid, ocmemid, ocimemid, owimemid});
			std::vector<std::pair<void *, size_t>> constargs({std::make_pair((void *) &input_maps.width, sizeof(unsigned int)),
															std::make_pair((void*) &input_maps.height, sizeof(unsigned int)),
															std::make_pair((void*) &filter.width, sizeof(unsigned int)),
															std::make_pair((void*) &filter.height, sizeof(unsigned int))});
			OpenCLInterface::OpenCLError err = ocl->callKernel(fberrorkid, dim, memargs, constargs);
			if (err != OpenCLInterface::OpenCLError::SUCCESS) {
				Logger::writeLine("ConvolutionalLayer::computeError(): Error when calling the OpenCL kernel for next error calculation.");
				return input;
			} else {
				dim.x = weights.size();
				ocl->getMemoryContent(nememid, (void *) &newerror[0], num_inputs * sizeof(float));
				memargs = std::vector<int>({oememid, imemid, smemid, wmemid, womemid, icmemid, icimemid});
				constargs.push_back(std::make_pair((void*) &learning, sizeof(float)));
				err = ocl->callKernel(fbweightskid, dim, memargs, constargs);
				if (err != OpenCLInterface::OpenCLError::SUCCESS) {
					Logger::writeLine("ConvolutionalLayer::computeError(): Error when calling the OpenCL kernel for weight calculation.");
					return input;
				}
				ocl->getMemoryContent(wmemid, (void *) &weights[0], weights.size() * sizeof(float));
				return newerror;
			}
		}
	} else {
		Logger::writeLine("ConvolutionalLayer::computeError(): OpenCLInterface not initialized. Unable to compute anything.");
		return input;
	}
}

std::string ConvolutionalLayer::getName() const {
	return "ConvolutionalLayer";
}

std::string ConvolutionalLayer::getDatastring() const {
	std::string datastring = act->getName() + ":";
	datastring += std::to_string(learning) + ":" + std::to_string(num_input_maps) + ":" + std::to_string(num_output_maps) + ":";
	datastring += std::to_string(input_maps.width) + ":" + std::to_string(input_maps.height) + ":";
	datastring += std::to_string(filter.width) + ":" + std::to_string(filter.height) + ":";
	datastring += getVectorRepresentation<unsigned int>(input_connections, ';') + ":";
	datastring += getVectorRepresentation<unsigned int>(input_connection_indices, ';') + ":";
	datastring += getVectorRepresentation<unsigned int>(output_connections, ';') + ":";
	datastring += getVectorRepresentation<unsigned int>(output_connection_indices, ';') + ":";
	datastring += getVectorRepresentation<unsigned int>(output_weight_indices, ';') + ":";
	datastring += getVectorRepresentation<unsigned int>(weight_output_maps, ';') + ":";
	datastring += getVectorRepresentation<float>(weights, ';');
	return datastring;
}

bool ConvolutionalLayer::parseDatastring(std::string datastring) {
	std::vector<std::string> data = parseVectorRepresentation<std::string> (datastring, ':');
	if (data.size() != 15) {
		Logger::writeLine("ConvolutionalLayer::parseDatastring(): Invalid number of parameters.");
		return false;
	} else {
		act = ActivationFunction::getObjectFromString(data[0]);
		if (act == nullptr) {
			Logger::writeLine("ConvolutionalLayer::parseDatastring(): Invalid activation function identifier: " + data[0]);
			return false;
		} else {
			learning = std::stof(data[1]);
			num_input_maps = std::stoul(data[2]);
			num_output_maps = std::stoul(data[3]);
			input_maps.width = std::stoul(data[4]);
			input_maps.height = std::stoul(data[5]);
			filter.width = std::stoul(data[6]);
			filter.height = std::stoul(data[7]);
			input_connections = parseVectorRepresentation<unsigned int>(data[8], ';');
			input_connection_indices = parseVectorRepresentation<unsigned int>(data[9], ';');
			output_connections = parseVectorRepresentation<unsigned int>(data[10], ';');
			output_connection_indices = parseVectorRepresentation<unsigned int>(data[11], ';');
			output_weight_indices = parseVectorRepresentation<unsigned int>(data[12], ';');
			weight_output_maps = parseVectorRepresentation<unsigned int>(data[13], ';');
			weights = parseVectorRepresentation<float>(data[14], ';');
		}
	}
	return true;
}

unsigned int ConvolutionalLayer::getNumInputFeatureMaps() const {
	return num_input_maps;
}

unsigned int ConvolutionalLayer::getNumOutputFeatureMaps() const {
	return num_output_maps;
}

ConvolutionalLayer::~ConvolutionalLayer() {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (wmemid > 0) {
		ocl->freeMemoryObject(wmemid);
	}
	if (womemid > 0) {
		ocl->freeMemoryObject(womemid);
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
	if (icmemid > 0) {
		ocl->freeMemoryObject(icmemid);
	}
	if (ocmemid > 0) {
		ocl->freeMemoryObject(ocmemid);
	}
	if (icimemid > 0) {
		ocl->freeMemoryObject(icimemid);
	}
	if (ocimemid > 0) {
		ocl->freeMemoryObject(ocimemid);
	}
	if (owimemid > 0) {
		ocl->freeMemoryObject(owimemid);
	}
	if (okid > 0) {
		ocl->deleteKernel(okid);
	}
	if (fberrorkid > 0) {
		ocl->deleteKernel(fberrorkid);
	}
	if (fbweightskid > 0) {
		ocl->deleteKernel(fbweightskid);
	}
}

} /* namespace clneural */
