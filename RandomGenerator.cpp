/*
 * RandomGenerator.cpp
 *
 *  Created on: Dec 6, 2015
 *      Author: jonas
 */

#include "RandomGenerator.h"

namespace clneural {

std::default_random_engine RandomGenerator::rand;

unsigned int RandomGenerator::getRandomNumber(unsigned int min, unsigned int max) {
	std::uniform_int_distribution<unsigned int> dist(min, max);
	return dist(rand);
}

unsigned int RandomGenerator::getRandomNumber(unsigned int max) {
	return getRandomNumber(0, max);
}

float RandomGenerator::getRandomNumber(float min, float max) {
	std::uniform_real_distribution<float> dist(min, max);
	return dist(rand);
}

float RandomGenerator::getRandomNumber(float max) {
	return getRandomNumber(0.0, max);
}

RandomGenerator::RandomGenerator() {
	// TODO Auto-generated constructor stub

}

RandomGenerator::~RandomGenerator() {
	// TODO Auto-generated destructor stub
}

} /* namespace neuronal */
