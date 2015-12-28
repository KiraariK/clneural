/*
 * ConvolutionalLayer.h
 *
 *  Created on: Dec 12, 2015
 *      Author: jonas
 */

#ifndef CONVOLUTIONALLAYER_H_
#define CONVOLUTIONALLAYER_H_

#include "NeuralNetworkLayer.h"
#include "ActivationFunction.h"
#include "OpenCLInterface.h"
#include <vector>
#include <memory>
#include <list>

namespace clneural {

class ConvolutionalLayer: public NeuralNetworkLayer {
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
	std::vector<unsigned int> input_connections;
	std::vector<unsigned int> input_connection_indices;
	std::vector<unsigned int> output_connections;
	std::vector<unsigned int> output_connection_indices;
	std::vector<unsigned int> output_weight_indices;
	std::vector<unsigned int> weight_output_maps;
	unsigned int num_input_maps = 0;
	unsigned int num_output_maps = 0;
	int wmemid = -1; //weights
	int womemid = -1; //weights to output feature maps
	int imemid = -1; //inputs
	int oememid = -1; //outputs and errors from next layer
	int nememid = -1; //error to previous layer
	int smemid =-1; //input*weight sums
	int icmemid = -1; //input feature maps per output feature map
	int icimemid = -1; //indices for every map in the above array
	int ocmemid = -1; //output feature maps per input feature map
	int ocimemid = -1; //indices for every map in the above array
	int owimemid = -1; //position in weight array for an output map assigned to an input map
	int okid = -1; //kernel for output computation
	int fberrorkid = -1; //kernel for previous error computation
	int fbweightskid = -1; //kernel for weight adaption computation
	Dimension input_maps;
	Dimension filter;
	bool initializeMemoryObjects(std::shared_ptr<OpenCLInterface> ocl);
	bool initializeKernelObjects(std::shared_ptr<OpenCLInterface> ocl);
	static const NeuralNetworkLayerRegisterHelper<ConvolutionalLayer> reg;
protected:
	virtual std::vector<float> computeOutput(const std::vector<float> &input);
	virtual std::vector<float> computeError(const std::vector<float> &input);
	virtual std::string getName() const;
	virtual std::string getDatastring() const;
	virtual bool parseDatastring(std::string datastring);
public:
	ConvolutionalLayer(Dimension input_maps, Dimension filter, const std::vector<std::list<unsigned int>> &input_to_output, std::shared_ptr<ActivationFunction> act, float learning);
	ConvolutionalLayer() = default;
	unsigned int getNumOutputFeatureMaps() const;
	unsigned int getNumInputFeatureMaps() const;
	virtual ~ConvolutionalLayer();
};

} /* namespace clneural */

#endif /* CONVOLUTIONALLAYER_H_ */
