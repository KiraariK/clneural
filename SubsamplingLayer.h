/*
 * SubsamplingLayer.h
 *
 *  Created on: Dec 26, 2015
 *      Author: jonas
 */

#ifndef SUBSAMPLINGLAYER_H_
#define SUBSAMPLINGLAYER_H_

#include "NeuralNetworkLayer.h"
#include "ActivationFunction.h"
#include "OpenCLInterface.h"

namespace clneural {

class SubsamplingLayer: public NeuralNetworkLayer {
public:
	struct Dimension {
		unsigned int width = 0;
		unsigned int height = 0;
	};
private:
	std::vector<float> weights;
	float learning = 0.5f;
	static const std::string fwclcode;
	static const std::string fberrorclcode;
	static const std::string fbweightsclcode;
	std::shared_ptr<ActivationFunction> act = nullptr;
	unsigned int num_feature_maps = 0;
	int wmemid = -1; //weights
	int imemid = -1; //inputs
	int oememid = -1; //outputs and errors from next layer
	int nememid = -1; //error to previous layer
	int smemid =-1; //input*weight sums
	int okid = -1; //kernel for output computation
	int fberrorkid = -1; //kernel for previous error computation
	int fbweightskid = -1; //kernel for weight adaption computation
	Dimension input_maps;
	Dimension filter;
	bool initializeMemoryObjects(std::shared_ptr<OpenCLInterface> ocl);
	bool initializeKernelObjects(std::shared_ptr<OpenCLInterface> ocl);
	static const NeuralNetworkLayerRegisterHelper<SubsamplingLayer> reg;
protected:
	virtual std::vector<float> computeOutput(const std::vector<float> &input);
	virtual std::vector<float> computeError(const std::vector<float> &input);
	virtual std::string getName() const;
	virtual std::string getDatastring() const;
	virtual bool parseDatastring(std::string datastring);
public:
	SubsamplingLayer() = default;
	SubsamplingLayer(Dimension input_maps, Dimension filter, unsigned int num_feature_maps, std::shared_ptr<ActivationFunction> act, float learning);
	virtual ~SubsamplingLayer();
};

} /* namespace clneural */

#endif /* SUBSAMPLINGLAYER_H_ */
