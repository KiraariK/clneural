/*
 * RandomGenerator.h
 *
 *  Created on: Dec 6, 2015
 *      Author: jonas
 */

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <random>

namespace clneural {

class RandomGenerator {
private:
	RandomGenerator();
	static std::default_random_engine rand;
public:
	static unsigned int getRandomNumber(unsigned int min, unsigned int max);
	static unsigned int getRandomNumber(unsigned int max);
	static float getRandomNumber(float min, float max);
	static float getRandomNumber(float max);
	virtual ~RandomGenerator();
};

} /* namespace clneural */

#endif /* RANDOMGENERATOR_H_ */
