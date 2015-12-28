/*
 * SigmoidActivationFunction.cpp
 *
 *  Created on: Dec 7, 2015
 *      Author: jonas
 */

#include "SigmoidActivationFunction.h"

namespace clneural {

ActivationFunctionRegisterHelper<SigmoidActivationFunction> SigmoidActivationFunction::reg("SigmoidActivationFunction");

SigmoidActivationFunction::SigmoidActivationFunction() {
	// TODO Auto-generated constructor stub

}

std::string SigmoidActivationFunction::getCode() const {
	std::string code = "float activationFunction(float input) {\n";
	code += "return (1.0f/(1.0f + exp(-input)));\n";
	code += "}\n";
	return code;
}

std::string SigmoidActivationFunction::getDerivCode() const {
	std::string code = "float activationDerivate(float input) {\n";
	code += "return (exp(-input)/((1.0f + exp(-input)) * (1.0f + exp(-input))));\n";
	code += "}\n";
	return code;
}

std::string SigmoidActivationFunction::getName() const {
	return "SigmoidActivationFunction";
}

SigmoidActivationFunction::~SigmoidActivationFunction() {
	// TODO Auto-generated destructor stub
}

} /* namespace clneural */
