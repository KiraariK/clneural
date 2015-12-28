/*
 * TanhActivationFunction.h
 *
 *  Created on: Dec 7, 2015
 *      Author: jonas
 */

#ifndef TANHACTIVATIONFUNCTION_H_
#define TANHACTIVATIONFUNCTION_H_

#include "ActivationFunction.h"

namespace clneural {

class TanhActivationFunction: public ActivationFunction {
private:
	static ActivationFunctionRegisterHelper<TanhActivationFunction> reg;
public:
	TanhActivationFunction();
	virtual std::string getCode() const;
	virtual std::string getDerivCode() const;
	virtual std::string getName() const;
	virtual ~TanhActivationFunction();
};

} /* namespace clneural */

#endif /* TANHACTIVATIONFUNCTION_H_ */
