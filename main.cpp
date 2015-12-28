/*
 * main.cpp
 *
 *  Created on: Nov 15, 2015
 *      Author: jonas
 */

#include "ImageDataset.h"
#include "FullFeedforwardLayer.h"
#include "ConvolutionalLayer.h"
#include "SubsamplingLayer.h"
#include "NeuralNetwork.h"
#include "LinearActivationFunction.h"
#include "SigmoidActivationFunction.h"
#include "TanhActivationFunction.h"
#include "OpenCLInterface.h"
#include "RandomGenerator.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>

std::vector<float> bingen() {
	std::vector<float> ex(2);
	float rand = clneural::RandomGenerator::getRandomNumber(1.0f);
	if (rand > 0.5) ex[0] = 1.0;
	else ex[0] = 0.0;
	rand = clneural::RandomGenerator::getRandomNumber(1.0f);
	if (rand > 0.5) ex[1] = 1.0;
	else ex[1] = 0.0;
	return ex;
}

std::vector<float> xorout(const std::vector<float> &in) {
	if (((in[0] > 0.5) && (in[1] < 0.5)) || ((in[0] < 0.5) && (in[1] > 0.5))) {
		return std::vector<float>({1.0f});
	}
	return std::vector<float>({0.0f});
}

void verifyNetwork(clneural::NeuralNetwork &net) {
	ImageDataset testset;
	testset.loadImagesFromFile("t10k-images-idx3-ubyte");
	testset.loadLabelsFromFile("t10k-labels-idx1-ubyte");
	std::cout << "Verifying network with " + std::to_string(testset.getSize()) + " images:" << std::endl;
	unsigned int size = testset.getSize();
	clock_t total_begin = clock();
	float avgmse = 0.0f;
	float avgedist = 0.0f;
	float maxedist = 0.0f;
	float maxmse = 0.0f;
	float avgtime = 0.0f;
	unsigned int counter = 1;
	unsigned int correct_outputs = 0;
	while (testset.getSize() > 0) {
		if ((counter % 1000) == 0) std::cout << "Computing step: " << counter << std::endl;
		std::pair<std::vector<float>, uint8_t> elem = testset.popRandomElementWithLabel();
		std::vector<float> desired(10, 0.0f);
		desired[elem.second] = 1.0f;
		clock_t begin = clock();
		net.processInput(elem.first);
		avgtime += ((float) (clock() - begin))/CLOCKS_PER_SEC;
		std::vector<float> output = net.getLastOutput();
		float tmpsum = 0.0f;
		for (unsigned int i = 0; i < output.size(); i++) {
			tmpsum += (output[i] - desired[i]) * (output[i] - desired[i]);
		}
		uint8_t maxresult = (uint8_t) std::distance(output.begin(), std::max_element(output.begin(), output.end()));
		if (maxresult == elem.second) correct_outputs++;
		float tmpedist = sqrt(tmpsum);
		float tmpmse = tmpsum/10.0f;
		if (tmpedist > maxedist) maxedist = tmpedist;
		if (tmpmse > maxmse) maxmse = tmpmse;
		avgedist += tmpedist;
		avgmse += tmpmse;
		counter++;
	}
	avgmse /= size;
	avgedist /= size;
	avgtime /= size;
	float totaltime = ((float) (clock() - total_begin)) / CLOCKS_PER_SEC;
	std::cout << "Verifycation completed, total time: " << totaltime << " seconds. Results: " << std::endl;
	std::cout << "Detected " << correct_outputs << " out of " << size << " elements correctly (" << ((float) correct_outputs)*100.0f/size << "%)." << std::endl;
	std::cout << "Average MSE: " << avgmse << std::endl;
	std::cout << "Maximum MSE: " << maxmse << std::endl;
	std::cout << "Average euclid distance: " << avgedist << std::endl;
	std::cout << "Maximum euclid distance: " << maxedist << std::endl;
	std::cout << "Average computation time for one step: " << avgtime << " seconds." << std::endl;
}

int main (int argc, char **argv) {
	ImageDataset d;
	d.loadImagesFromFile("train-images-idx3-ubyte");
	d.loadLabelsFromFile("train-labels-idx1-ubyte");
	std::shared_ptr<clneural::ActivationFunction> act(new clneural::SigmoidActivationFunction());
	std::shared_ptr<clneural::ActivationFunction> act2(new clneural::LinearActivationFunction());
	std::vector<std::list<unsigned int>> C1_connections(6, std::list<unsigned int>({0}));
	clneural::ConvolutionalLayer::Dimension C1_input;
	clneural::ConvolutionalLayer::Dimension C1_filter;
	float training_speed = 0.7f;
	C1_input.width = 32;
	C1_input.height = 32;
	C1_filter.width = 5;
	C1_filter.height = 5;
	std::shared_ptr<clneural::NeuralNetworkLayer> C1(new clneural::ConvolutionalLayer(C1_input, C1_filter, C1_connections, act, training_speed));
	clneural::SubsamplingLayer::Dimension S2_input;
	clneural::SubsamplingLayer::Dimension S2_filter;
	S2_input.width = 28;
	S2_input.height = 28;
	S2_filter.width = 2;
	S2_filter.height = 2;
	std::shared_ptr<clneural::NeuralNetworkLayer> S2(new clneural::SubsamplingLayer(S2_input, S2_filter, 6, act2, training_speed));
	std::vector<std::list<unsigned int>> C3_connections(16);
	C3_connections[0] = std::list<unsigned int>({0,1,2});
	C3_connections[1] = std::list<unsigned int>({1,2,3});
	C3_connections[2] = std::list<unsigned int>({2,3,4});
	C3_connections[3] = std::list<unsigned int>({3,4,5});
	C3_connections[4] = std::list<unsigned int>({4,5,0});
	C3_connections[5] = std::list<unsigned int>({5,0,1});
	C3_connections[6] = std::list<unsigned int>({0,1,2,3});
	C3_connections[7] = std::list<unsigned int>({1,2,3,4});
	C3_connections[8] = std::list<unsigned int>({2,3,4,5});
	C3_connections[9] = std::list<unsigned int>({3,4,5,0});
	C3_connections[10] = std::list<unsigned int>({4,5,0,1});
	C3_connections[11] = std::list<unsigned int>({5,0,1,2});
	C3_connections[12] = std::list<unsigned int>({0,1,3,4});
	C3_connections[13] = std::list<unsigned int>({1,2,4,5});
	C3_connections[14] = std::list<unsigned int>({0,2,3,5});
	C3_connections[15] = std::list<unsigned int>({0,1,2,3,4,5});
	clneural::ConvolutionalLayer::Dimension C3_input;
	clneural::ConvolutionalLayer::Dimension C3_filter;
	C3_input.width = 14;
	C3_input.height = 14;
	C3_filter.width = 5;
	C3_filter.height = 5;
	std::shared_ptr<clneural::NeuralNetworkLayer> C3(new clneural::ConvolutionalLayer(C3_input, C3_filter, C3_connections, act, training_speed));
	clneural::SubsamplingLayer::Dimension S4_input;
	clneural::SubsamplingLayer::Dimension S4_filter;
	S4_input.width = 10;
	S4_input.height = 10;
	S4_filter.width = 2;
	S4_filter.height = 2;
	std::shared_ptr<clneural::NeuralNetworkLayer> S4(new clneural::SubsamplingLayer(S4_input, S4_filter, 16, act2, training_speed));
	std::shared_ptr<clneural::NeuralNetworkLayer> N1(new clneural::FullFeedforwardLayer(400, 84, act, training_speed));
	std::shared_ptr<clneural::NeuralNetworkLayer> N2(new clneural::FullFeedforwardLayer(84, 10, act, training_speed));
	clneural::NeuralNetwork n;
	n.addLayer(C1);
	n.addLayer(S2);
	n.addLayer(C3);
	n.addLayer(S4);
	n.addLayer(N1);
	n.addLayer(N2);


	std::shared_ptr<OpenCLInterface> ocl = OpenCLInterface::getInstance();
	ocl->initialize(CL_DEVICE_TYPE_CPU);

	float dist = 0.0f;
	for (unsigned int i = 0; i < 60000; i++) {
		std::pair<std::vector<float>, uint8_t> trainelem = d.popRandomElementWithLabel();
		std::vector<float> desired(10, 0.0f);
		desired[trainelem.second] = 1.0f;
		dist += n.trainNetwork(trainelem.first, desired);
		std::vector<float> nout = n.getLastOutput();
		if ((i % 1000) == 0) {
			std::cout << "TIME: " << ((float) clock())/CLOCKS_PER_SEC << ", STEP:" << (i + 1) << ", MDIST: " << dist/1000.0f << ", OUT: (" << nout[0];
			for (unsigned int j = 1; j < nout.size(); j++) std::cout << "," << nout[j];
			std::cout << "), DESIRED: (" << desired[0];
			for (unsigned int j = 1; j < desired.size(); j++) std::cout << "," << desired[j];
			std::cout << ")" << std::endl;
			dist = 0.0f;
		}
	}
	n.saveToFile("conv_images1.net");
	verifyNetwork(n);
	return 0;
}


