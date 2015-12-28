/*
 * LinearActivationFunction.cpp
 *
 *  Created on: Dec 9, 2015
 *      Author: jonas
 */

#include "LinearActivationFunction.h"

namespace clneural {

const ActivationFunctionRegisterHelper<LinearActivationFunction> LinearActivationFunction::reg("LinearActivationFunction");

LinearActivationFunction::LinearActivationFunction() {
}

std::string LinearActivationFunction::getCode() const {
	std::string code = "float activationFunction(float input) {\n";
	code += "return input;\n";
	code += "}\n";
	return code;
}

std::string LinearActivationFunction::getDerivCode() const {
	std::string code = "float activationDerivate(float input) {\n";
	code += "return 1.0f;\n";
	code += "}\n";
	return code;
}

std::string LinearActivationFunction::getName() const {
	return "LinearActivationFunction";
}

LinearActivationFunction::~LinearActivationFunction() {
}

} /* namespace clneural */
