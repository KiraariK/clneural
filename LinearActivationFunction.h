/*
 * LinearActivationFunction.h
 *
 *  Created on: Dec 9, 2015
 *      Author: jonas
 */

#ifndef LINEARACTIVATIONFUNCTION_H_
#define LINEARACTIVATIONFUNCTION_H_

#include "ActivationFunction.h"

namespace clneural {

class LinearActivationFunction: public ActivationFunction {
private:
	static const ActivationFunctionRegisterHelper<LinearActivationFunction> reg;
public:
	virtual std::string getCode() const;
	virtual std::string getDerivCode() const;
	virtual std::string getName() const;
	LinearActivationFunction();
	virtual ~LinearActivationFunction();
};

} /* namespace clneural */

#endif /* LINEARACTIVATIONFUNCTION_H_ */
