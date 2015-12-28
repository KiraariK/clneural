/*
 * ActivationFunction.h
 *
 *  Created on: Dec 7, 2015
 *      Author: jonas
 */

#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <utility>

namespace clneural {

class ActivationFunction {
public:
	ActivationFunction();
	static std::shared_ptr<ActivationFunction> getObjectFromString(std::string name);
	virtual std::string getCode() const = 0;
	virtual std::string getDerivCode() const = 0;
	virtual std::string getName() const = 0;
	virtual ~ActivationFunction();
};

class ActivationFunctionRegister {
friend class ActivationFunction;
private:
	static std::unordered_map<std::string, std::shared_ptr<ActivationFunction> (*)()> *typemap;
protected:
	static std::unordered_map<std::string, std::shared_ptr<ActivationFunction> (*)()> *getMap() {
		if (typemap == nullptr) {
			typemap = new std::unordered_map<std::string, std::shared_ptr<ActivationFunction> (*)()>();
		}
		return typemap;
	}
	template<typename T> static std::shared_ptr<ActivationFunction> createInstance() {
		return std::shared_ptr<ActivationFunction>(new T);
	}
};

template<typename T>
class ActivationFunctionRegisterHelper : public ActivationFunctionRegister {
public:
	ActivationFunctionRegisterHelper(std::string name) {
		getMap()->insert(std::make_pair(name, &(ActivationFunctionRegister::createInstance<T>)));
	}
};

} /* namespace clneural */

#endif /* ACTIVATIONFUNCTION_H_ */
