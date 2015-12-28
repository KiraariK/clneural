/*
 * NeuralNetworkLayer.cpp
 *
 *  Created on: Nov 15, 2015
 *      Author: jonas
 */

#include "Logger.h"
#include <exception>
#include "NeuralNetworkLayer.h"

namespace clneural {

std::unordered_map<std::string, std::shared_ptr<NeuralNetworkLayer> (*)()> *NeuralNetworkLayerRegister::typemap = nullptr;

NeuralNetworkLayer::NeuralNetworkLayer(unsigned int num_inputs, unsigned int num_outputs) :
	num_inputs(num_inputs),
	num_outputs(num_outputs)
{
}

std::shared_ptr<NeuralNetworkLayer> NeuralNetworkLayer::getObjectFromString(std::string name) {
	std::unordered_map<std::string, std::shared_ptr<NeuralNetworkLayer> (*)()>::iterator it = NeuralNetworkLayerRegister::getMap()->find(name);
	if (it == NeuralNetworkLayerRegister::getMap()->end()) {
		return nullptr;
	}
	return it->second();
}

unsigned int NeuralNetworkLayer::getNumInputs() const {
	return num_inputs;
}

unsigned int NeuralNetworkLayer::getNumOutputs() const {
	return num_outputs;
}

std::vector<float> NeuralNetworkLayer::getLastInput() const {
	return last_input;
}

std::vector<float> NeuralNetworkLayer::getLastOutput() const {
	return last_output;
}

std::shared_ptr<NeuralNetworkLayer> NeuralNetworkLayer::getNextLayer() const {
	return next_layer;
}

std::shared_ptr<NeuralNetworkLayer> NeuralNetworkLayer::getPreviousLayer() const {
	return previous_layer;
}

void NeuralNetworkLayer::processAndForwardInput(const std::vector<float> &input) {
	std::vector<float> output = computeOutput(input);
	if (output.size() != num_outputs) {
		throw new std::runtime_error("NeuralNetworkLayer::processAndForwardInput(): Invalid output vector length.");
	}
	last_output = output;
	last_input = input;
	if (next_layer != nullptr) next_layer->processAndForwardInput(output);
}

void NeuralNetworkLayer::processAndForwardError(const std::vector<float> &error) {
	std::vector<float> newerror = computeError(error);
	if (newerror.size() != num_inputs) {
		throw new std::runtime_error("NeuralNetworkLayer::processAndForwardError(): Invalid error vector length.");
	}
	if (previous_layer != nullptr) previous_layer->processAndForwardError(newerror);
}

bool NeuralNetworkLayer::setNextLayer(std::shared_ptr<NeuralNetworkLayer> newNextLayer) {
	if (newNextLayer == nullptr) {
		return false;
	}
	if (newNextLayer->getNumInputs() != num_outputs) {
		return false;
	}
	next_layer = newNextLayer;
	return true;
}

bool NeuralNetworkLayer::setPreviousLayer(std::shared_ptr<NeuralNetworkLayer> newPreviousLayer) {
	if (newPreviousLayer == nullptr) {
		return false;
	}
	if (newPreviousLayer->getNumOutputs() != num_inputs) {
		return false;
	}
	previous_layer = newPreviousLayer;
	return true;
}

std::string NeuralNetworkLayer::getStringRepresentation() const {
	return getName() + ":" + std::to_string(num_inputs) + ":" + std::to_string(num_outputs) + ":" + getDatastring();
}

std::shared_ptr<NeuralNetworkLayer> NeuralNetworkLayer::createFromStringRepresentation(std::string repr) {
	size_t newpos = repr.find_first_of(':', 0);
	std::string name = repr.substr(0, newpos);
	size_t lastpos = newpos + 1;
	std::shared_ptr<NeuralNetworkLayer> layer = getObjectFromString(name);
	if (layer != nullptr) {
		newpos = repr.find_first_of(':', lastpos);
		layer->num_inputs = std::stoi(repr.substr(lastpos, newpos - lastpos));
		lastpos = newpos + 1;
		newpos = repr.find_first_of(':', lastpos);
		layer->num_outputs = std::stoi(repr.substr(lastpos, newpos - lastpos));
		lastpos = newpos + 1;
		bool res = layer->parseDatastring(repr.substr(lastpos, repr.length() - lastpos));
		if (!res) {
			Logger::writeLine("NeuralNetworkLayer::createFromStringRepresentation(): Error while parsing datastring for " + name + ".");
			return nullptr;
		}
	} else {
		Logger::writeLine("NeuralNetworkLayer::createFromStringRepresentation(): Unable to create a layer for the given class name: " + name);
	}
	return layer;
}

NeuralNetworkLayer::~NeuralNetworkLayer() {
	next_layer = nullptr;
	previous_layer = nullptr;
}

} /* namespace clneuronal */

