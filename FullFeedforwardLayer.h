/*
 * FullFeedforwardLayer.h
 *
 *  Created on: Dec 4, 2015
 *      Author: jonas
 */

#ifndef FULLFEEDFORWARDLAYER_H_
#define FULLFEEDFORWARDLAYER_H_

#include "NeuralNetworkLayer.h"
#include "ActivationFunction.h"
#include "OpenCLInterface.h"
#include <vector>

namespace clneural {

class FullFeedforwardLayer: public NeuralNetworkLayer {
private:
	std::vector<float> weights;
	float learning = 0.5f;
	std::shared_ptr<ActivationFunction> act;
	static const std::string fwclcode;
	static const std::string fbclcode;
	int okid = -1;
	int fbkid = -1;
	int wmemid = -1; //weights
	int imemid = -1; //inputs
	int nememid = -1; //errors for previous layer (delta)
	int oememid = -1; //neuron outputs (after activation function) and error from next layer
	int smemid = -1; //neuron sums
	bool initializeMemoryObjects(std::shared_ptr<OpenCLInterface> ocl);
	bool initializeKernelObjects(std::shared_ptr<OpenCLInterface> ocl);
	static const NeuralNetworkLayerRegisterHelper<FullFeedforwardLayer> reg;
protected:
	virtual std::vector<float> computeOutput(const std::vector<float> &input);
	virtual std::vector<float> computeError(const std::vector<float> &input);
	virtual std::string getName() const;
	virtual std::string getDatastring() const;
	virtual bool parseDatastring(std::string datastring);
public:
	FullFeedforwardLayer(unsigned int num_inputs, unsigned int num_outputs, std::shared_ptr<ActivationFunction> act, float learning);
	FullFeedforwardLayer() = default;
	virtual ~FullFeedforwardLayer();
};

} /* namespace clneural */

#endif /* FULLFEEDFORWARDLAYER_H_ */
