/*
 * SigmoidActivationFunction.h
 *
 *  Created on: Dec 7, 2015
 *      Author: jonas
 */

#ifndef SIGMOIDACTIVATIONFUNCTION_H_
#define SIGMOIDACTIVATIONFUNCTION_H_

#include "ActivationFunction.h"

namespace clneural {

class SigmoidActivationFunction: public ActivationFunction {
private:
	static ActivationFunctionRegisterHelper<SigmoidActivationFunction> reg;
public:
	SigmoidActivationFunction();
	virtual std::string getCode() const;
	virtual std::string getDerivCode() const;
	virtual std::string getName() const;
	virtual ~SigmoidActivationFunction();
};

} /* namespace clneural */

#endif /* SIGMOIDACTIVATIONFUNCTION_H_ */
