/*
 * NeuralNetwork.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: jonas
 */

#include "Logger.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include "NeuralNetwork.h"

namespace clneural {

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::~NeuralNetwork() {
}

bool NeuralNetwork::addLayer(std::shared_ptr<NeuralNetworkLayer> layer) {
	if ((first_layer == nullptr) && (last_layer == nullptr)) {
		first_layer = layer;
		last_layer = layer;
		return true;
	}
	if (last_layer->getNumOutputs() != layer->getNumInputs()) {
		Logger::writeLine("NeuralNetwork::addLayer(): Inputs not matching outputs for layer to be added.");
		return false;
	}
	last_layer->setNextLayer(layer);
	layer->setPreviousLayer(last_layer);
	last_layer = layer;
	return true;
}

std::vector<float> NeuralNetwork::getLastOutput() const {
	if (last_layer == nullptr) {
		return std::vector<float>();
	}
	return last_layer->getLastOutput();
}

void NeuralNetwork::processInput(const std::vector<float> &input) {
	if (first_layer != nullptr) {
		first_layer->processAndForwardInput(input);
	}
}

float NeuralNetwork::trainNetwork(const std::vector<float> &input, const std::vector<float> &desired_output) {
	if (first_layer == nullptr) {
		return 0.0f;
	}
	processInput(input);
	std::vector<float> out = getLastOutput();
	std::vector<float> dif(out.size());
	float dist = 0.0f;
	for (unsigned int i = 0; i < out.size(); i++) {
		dif[i] = desired_output[i] - out[i];
		dist += dif[i]*dif[i];
	}
	last_layer->processAndForwardError(dif);
	return sqrt(dist);
}

std::string NeuralNetwork::getStringRepresentation() const {
	std::string repr;
	std::shared_ptr<NeuralNetworkLayer> iterator = first_layer;
	while (iterator != nullptr) {
		repr += iterator->getStringRepresentation() + "\n";
		iterator = iterator->getNextLayer();
	}
	return repr;
}

bool NeuralNetwork::parseStringRepresentation(std::string repr) {
	bool res = true;
	size_t lastpos = 0;
	size_t newpos = repr.find_first_of('\n', 0);
	while ((newpos != std::string::npos) && res) {
		std::shared_ptr<NeuralNetworkLayer> templayer = NeuralNetworkLayer::createFromStringRepresentation(repr.substr(lastpos, newpos - lastpos));
		if (templayer != nullptr) {
			if (!addLayer(templayer)) {
				Logger::writeLine("NeuralNetwork::parseStringRepresentation(): Unable to add layer in string representation.");
				res = false;
			}
		} else {
			Logger::writeLine("NeuralNetwork::parseStringRepresentation(): Unable to parse layer string representation.");
			res = false;
		}
		lastpos = newpos + 1;
		newpos = repr.find_first_of('\n', 0);
	}
	return res;
}

bool NeuralNetwork::saveToFile(std::string filename) const {
	std::ofstream file(filename, std::ios::out | std::ios::trunc);
	if (!file.is_open()) {
		Logger::writeLine("NeuralNetwork::saveToFile(): Unable to open file: " + filename);
		return false;
	}
	file << getStringRepresentation();
	file.close();
	return true;
}

bool NeuralNetwork::loadFromFile(std::string filename) {
	std::ifstream file(filename, std::ios::in);
	if (!file.is_open()) {
		Logger::writeLine("NeuralNetwork::loadFromFile(): Unable to open file: " + filename);
		return false;
	}
	std::stringstream content;
	content << file.rdbuf();
	parseStringRepresentation(content.str());
	return true;
}

} /* namespace clneural */
