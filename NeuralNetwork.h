/*
 * NeuronalNetwork.h
 *
 *  Created on: Dec 10, 2015
 *      Author: jonas
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <memory>
#include "NeuralNetworkLayer.h"

namespace clneural {

class NeuralNetwork {
private:
	std::shared_ptr<NeuralNetworkLayer> first_layer = nullptr;
	std::shared_ptr<NeuralNetworkLayer> last_layer = nullptr;

public:
	NeuralNetwork();
	bool addLayer(std::shared_ptr<NeuralNetworkLayer> layer);
	std::vector<float> getLastOutput() const;
	void processInput(const std::vector<float> &input);
	float trainNetwork(const std::vector<float> &input, const std::vector<float> &desired_output);
	bool parseStringRepresentation(std::string repr);
	std::string getStringRepresentation() const;
	bool saveToFile(std::string filename) const;
	bool loadFromFile(std::string filename);
	virtual ~NeuralNetwork();
};

} /* namespace clneural */

#endif /* NEURALNETWORK_H_ */
