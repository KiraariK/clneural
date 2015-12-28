/*
 * FullFeedforwardLayer.cpp
 *
 *  Created on: Dec 4, 2015
 *      Author: jonas
 */

#include "FullFeedforwardLayer.h"
#include "RandomGenerator.h"
#include "OpenCLInterface.h"
#include "Logger.h"

namespace clneural {

const std::string FullFeedforwardLayer::fwclcode = "__kernel void computeOutput(__global const float *inputs, __global const float *weights, __global float *outputs, __global float *netsums, unsigned int num_inputs) {\n"
												 "unsigned int neuron_id = get_global_id(0);\n"
												 "float sum = 0.0f;\n"
												 "for (unsigned int i = 0; i < num_inputs; i++) {\n"
												 "sum += inputs[i] * weights[neuron_id*(num_inputs+1) + i];\n"
												 "}\n"
												 "sum += weights[neuron_id*(num_inputs+1)+num_inputs];\n"
												 "netsums[neuron_id] = sum;"
												 "outputs[neuron_id] = activationFunction(sum);\n"
												 "}\n";

const std::string FullFeedforwardLayer::fbclcode = "__kernel void computeError(__global const float *error, __global const float *last_inputs, __global const float *netsums, __global float *weights, __global float *nexterror, unsigned int num_outputs, float learning_rate) {\n"
												 "unsigned int input_id = get_global_id(0);\n"
												 "unsigned int num_inputs = get_global_size(0) - 1;\n"
												 "float sum = 0.0f;\n"
												 "float last_input = 1.0f;\n"
												 "if (input_id != num_inputs) last_input = last_inputs[input_id];\n"
												 "for (unsigned int i = 0; i < num_outputs; i++) {\n"
												 "float delta = error[i] * activationDerivate(netsums[i]);\n"
												 "sum += weights[i*(num_inputs+1) + input_id] * delta;"
												 "weights[i*(num_inputs+1) + input_id] += learning_rate * delta * last_input;\n"
												 "}\n"
												 "if (input_id != num_inputs) nexterror[input_id] = sum;\n"
												 "}\n";

const NeuralNetworkLayerRegisterHelper<FullFeedforwardLayer> FullFeedforwardLayer::reg("FullFeedforwardLayer");

FullFeedforwardLayer::FullFeedforwardLayer(unsigned int num_inputs, unsigned int num_outputs, std::shared_ptr<ActivationFunction> act, float learning) :
	NeuralNetworkLayer(num_inputs, num_outputs), act(act), learning(learning)
{
	weights = std::vector<float>((num_inputs + 1) * num_outputs);
	for (unsigned int i = 0; i < weights.size(); i++) {
		weights[i] = RandomGenerator::getRandomNumber(-1.0f/((float) (num_inputs+ 1)), 1.0f/((float) (num_inputs+ 1)));
	}
}

bool FullFeedforwardLayer::initializeMemoryObjects(std::shared_ptr<OpenCLInterface> ocl) {
	if (wmemid < 0) {
		wmemid = ocl->allocateMemoryObject((void *) &weights[0], (num_inputs + 1) * num_outputs * sizeof(float), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
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
		nememid = ocl->allocateMemoryObject(NULL, num_inputs * sizeof(float), CL_MEM_READ_ONLY);
	}
	if ((oememid < 0) || (imemid < 0) || (wmemid < 0) ||(smemid < 0) || (nememid < 0)) {
		return false;
	}
	return true;
}

bool FullFeedforwardLayer::initializeKernelObjects(std::shared_ptr<OpenCLInterface> ocl) {
	if (okid < 0) {
		std::string code = act->getCode() + fwclcode;
		okid = ocl->createKernelFromSource(code, "computeOutput");
	}
	if (fbkid < 0) {
		std::string code = act->getDerivCode() + fbclcode;
		fbkid = ocl->createKernelFromSource(code, "computeError");
	}
	if ((okid < 0) || (fbkid < 0)) {
		return false;
	}
	return true;
}

std::vector<float> FullFeedforwardLayer::computeOutput(const std::vector<float> &input) {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (ocl->isInitialized()) {
		if (!initializeKernelObjects(ocl)) {
			Logger::writeLine("FullFeedforwardLayer::computeOutput(): Can't initialize kernel. Unable to compute anything.");
			return input;
		} else if (!initializeMemoryObjects(ocl)) {
			Logger::writeLine("FullFeedforwardLayer::computeOutput(): Can't initialize memory objects. Unable to compute anything.");
			return input;
		} else {
			std::vector<float> output(num_outputs);
			ocl->writeMemoryContent(imemid, (void*) &input[0], num_inputs * sizeof(float));
			OpenCLInterface::Dimension dim;
			dim.x = num_outputs;
			std::vector<int> memargs({imemid, wmemid, oememid, smemid});
			std::vector<std::pair<void *, size_t>> constargs({std::make_pair((void *) &num_inputs, sizeof(unsigned int))});
			OpenCLInterface::OpenCLError err = ocl->callKernel(okid, dim, memargs, constargs);
			if (err != OpenCLInterface::OpenCLError::SUCCESS) {
				Logger::writeLine("FullFeedforwardLayer::computeOutput(): Error when calling the OpenCL kernel.");
				return input;
			} else {
				ocl->getMemoryContent(oememid, (void *) &output[0], num_outputs * sizeof(float));
				return output;
			}
		}
	} else {
		Logger::writeLine("FullFeedforwardLayer::computeOutput(): OpenCLInterface not initialized. Unable to compute anything.");
		return input;
	}
}

std::vector<float> FullFeedforwardLayer::computeError(const std::vector<float> &input) {
	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	if (ocl->isInitialized()) {
		if (!initializeKernelObjects(ocl)) {
			Logger::writeLine("FullFeedforwardLayer::computeError(): Can't initialize kernel. Unable to compute anything.");
			return input;
		} else if (!initializeMemoryObjects(ocl)) {
			Logger::writeLine("FullFeedforwardLayer::computeError(): Can't initialize memory objects. Unable to compute anything.");
			return input;
		} else {
			std::vector<float> newerror(num_inputs);
			ocl->writeMemoryContent(oememid, (void*) &input[0], num_outputs * sizeof(float));

			OpenCLInterface::Dimension dim;
			dim.x = num_inputs + 1;
			std::vector<int> memargs({oememid, imemid, smemid, wmemid, nememid});
			std::vector<std::pair<void *, size_t>> constargs({std::make_pair((void *) &num_outputs, sizeof(unsigned int)),
																std::make_pair((void *) &learning, sizeof(float))});
			OpenCLInterface::OpenCLError err = ocl->callKernel(fbkid, dim, memargs, constargs);
			if (err != OpenCLInterface::OpenCLError::SUCCESS) {
				Logger::writeLine("FullFeedforwardLayer::computeError(): Error when calling the OpenCL kernel.");
				return input;
			} else {
				ocl->getMemoryContent(nememid, (void *) &newerror[0], num_inputs * sizeof(float));
				ocl->getMemoryContent(wmemid, (void *) &weights[0], num_inputs * num_outputs * sizeof(float));
				return newerror;
			}
		}
	} else {
		Logger::writeLine("FullFeedforwardLayer::computeError(): OpenCLInterface not initialized. Unable to compute anything.");
		return input;
	}
}

std::string FullFeedforwardLayer::getName() const {
	return "FullFeedforwardLayer";
}

std::string FullFeedforwardLayer::getDatastring() const {
	std::string repr = act->getName() + ":" + std::to_string(learning) + ":" + getVectorRepresentation<float>(weights, ';');
	return repr;
}

bool FullFeedforwardLayer::parseDatastring(std::string datastring) {
	std::vector<std::string> data = parseVectorRepresentation<std::string>(datastring, ':');
	if (data.size() != 3) {
		Logger::writeLine("FullFeedforwardLayer::parseDatastring(): Invalid number of parameters.");
		return false;
	} else {
		act = ActivationFunction::getObjectFromString(data[0]);
		if (act == nullptr) {
			Logger::writeLine("FullFeedforwardLayer::parseDatastring(): Invalid activation function identifier: " + data[0]);
			return false;
		}
		learning = std::stof(data[1]);
		weights = parseVectorRepresentation<float>(data[2], ';');
		if(weights.size() != (num_inputs + 1) * num_outputs) {
			Logger::writeLine("FullFeedforwardLayer::parseDatastring(): Invalid number of weights.");
			return false;
		}
	}
	return true;
}

FullFeedforwardLayer::~FullFeedforwardLayer() {
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
	if (nememid > 0) {
		ocl->freeMemoryObject(nememid);
	}
	if (smemid > 0) {
		ocl->freeMemoryObject(smemid);
	}
	if(okid > 0) {
		ocl->deleteKernel(okid);
	}
	if(fbkid > 0) {
		ocl->deleteKernel(okid);
	}
}

} /* namespace clneural */
