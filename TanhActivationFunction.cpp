/*
 * TanhActivationFunction.cpp
 *
 *  Created on: Dec 7, 2015
 *      Author: jonas
 */

#include "TanhActivationFunction.h"

namespace clneural {

ActivationFunctionRegisterHelper<TanhActivationFunction> TanhActivationFunction::reg("TanhActivationFunction");

TanhActivationFunction::TanhActivationFunction() {
	// TODO Auto-generated constructor stub

}

std::string TanhActivationFunction::getCode() const {
	std::string code = "float activationFunction(float input) {\n";
	code += "return (1.7159f*tanh(2.0f/3.0f * input));\n";
	code += "}\n";
	return code;
}

std::string TanhActivationFunction::getDerivCode() const {
	std::string code = "float activationDerivate(float input) {\n";
	code += "return (1.7159f * 2.0f/3.0f * (1 - tanh(2.0f/3.0f * input) * tanh(2.0f/3.0f * input)));\n";
	code += "}\n";
	return code;
}

std::string TanhActivationFunction::getName() const {
	return "TanhActivationFunction";
}

TanhActivationFunction::~TanhActivationFunction() {
	// TODO Auto-generated destructor stub
}

} /* namespace clneural */
