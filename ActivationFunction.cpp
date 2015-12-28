/*
 * ActivationFunction.cpp
 *
 *  Created on: Dec 7, 2015
 *      Author: jonas
 */

#include "ActivationFunction.h"

namespace clneural {

std::unordered_map<std::string, std::shared_ptr<ActivationFunction> (*)()> *ActivationFunctionRegister::typemap = nullptr;

std::shared_ptr<ActivationFunction> ActivationFunction::getObjectFromString(std::string name) {
	std::unordered_map<std::string, std::shared_ptr<ActivationFunction>(*)()>::iterator it = ActivationFunctionRegister::getMap()->find(name);
	if (it == ActivationFunctionRegister::getMap()->end()) {
		return nullptr;
	}
	return it->second();
}

ActivationFunction::ActivationFunction() {
	// TODO Auto-generated constructor stub

}

ActivationFunction::~ActivationFunction() {
	// TODO Auto-generated destructor stub
}

} /* namespace clneural */
