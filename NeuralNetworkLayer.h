/*
 * NeuronalNetworkLayer.h
 *
 *  Created on: Nov 15, 2015
 *      Author: jonas
 */

#ifndef NEURALNETWORKLAYER_H_
#define NEURALNETWORKLAYER_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <sstream>

namespace clneural {

class NeuralNetworkLayer {
private:
	std::shared_ptr<NeuralNetworkLayer> next_layer = nullptr;
	std::shared_ptr<NeuralNetworkLayer> previous_layer = nullptr;
	static std::shared_ptr<NeuralNetworkLayer> getObjectFromString(std::string name);
protected:
	unsigned int num_inputs = 0;
	unsigned int num_outputs = 0;
	std::vector<float> last_input;
	std::vector<float> last_output;
	virtual std::vector<float> computeOutput(const std::vector<float> &input) = 0;
	virtual std::vector<float> computeError(const std::vector<float> &input) = 0;
	virtual std::string getName() const = 0;
	virtual std::string getDatastring() const = 0;
	virtual bool parseDatastring(std::string datastring) = 0;
	template<typename T> std::string getVectorRepresentation(const std::vector<T> &vector, char delim) const {
		std::string result = "";
		if (vector.size() > 0) {
			result += std::to_string(vector[0]);
			for (unsigned int i = 1; i < vector.size(); i++) {
				result += delim + std::to_string(vector[i]);
			}
		}
		return result;
	}
	template<typename T> std::vector<T> parseVectorRepresentation(std::string representation, char delim) const {
		size_t oldpos = 0;
		size_t newpos = representation.find_first_of(delim, 0);
		std::vector<T> result;
		while (newpos != std::string::npos) {
			T value;
			std::stringstream ss(representation.substr(oldpos, newpos - oldpos));
			ss >> value;
			result.push_back(value);
			oldpos = newpos + 1;
			newpos = representation.find_first_of(delim, oldpos);
		}
		T value;
		std::stringstream ss(representation.substr(oldpos, representation.length() - oldpos));
		ss >> value;
		result.push_back(value);
		return result;
	}
public:
	NeuralNetworkLayer(unsigned int num_inputs, unsigned int num_outputs);
	NeuralNetworkLayer() = default;
	bool setNextLayer(std::shared_ptr<NeuralNetworkLayer> nextLayer);
	bool setPreviousLayer(std::shared_ptr<NeuralNetworkLayer> previousLayer);
	std::shared_ptr<NeuralNetworkLayer> getNextLayer() const;
	std::shared_ptr<NeuralNetworkLayer> getPreviousLayer() const;
	unsigned int getNumInputs() const;
	unsigned int getNumOutputs() const;
	void processAndForwardInput(const std::vector<float> &input);
	void processAndForwardError(const std::vector<float> &error);
	std::vector<float> getLastInput() const;
	std::vector<float> getLastOutput() const;
	static std::shared_ptr<NeuralNetworkLayer> createFromStringRepresentation(std::string repr);
	std::string getStringRepresentation() const;
	virtual ~NeuralNetworkLayer();
};

class NeuralNetworkLayerRegister {
friend class NeuralNetworkLayer;
private:
	static std::unordered_map<std::string, std::shared_ptr<NeuralNetworkLayer> (*)()> *typemap;
protected:
	static std::unordered_map<std::string, std::shared_ptr<NeuralNetworkLayer> (*)()> *getMap() {
		if (typemap == nullptr) {
			typemap = new std::unordered_map<std::string, std::shared_ptr<NeuralNetworkLayer> (*)()>();
		}
		return typemap;
	}
	template <typename T> static std::shared_ptr<NeuralNetworkLayer> createInstance() {
		return std::shared_ptr<NeuralNetworkLayer>(new T);
	}
};

template <typename T>
class NeuralNetworkLayerRegisterHelper : public NeuralNetworkLayerRegister {
public:
	NeuralNetworkLayerRegisterHelper(std::string name) {
		getMap()->insert(std::make_pair(name, &(NeuralNetworkLayerRegister::createInstance<T>)));
	}
};

} /* namespace clneural */

#endif /* NEURALNETWORKLAYER_H_ */
