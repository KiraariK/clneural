/*
 * Logger.cpp
 *
 *  Created on: Sep 12, 2015
 *      Author: jonas
 */

#include "Logger.h"
#include <iostream>
#include <iomanip>
#include <time.h>

Logger::Logger() {
}

void Logger::writeLine(std::string logline) {
	clock_t curtime = clock();
	std::cout << std::setfill('0') << std::setw(10) << ((float) clock())/CLOCKS_PER_SEC;
	std::cout << ": " << logline << std::endl;
}

void Logger::writeLineNotime(std::string logline) {
	std::cout<< logline << std::endl;
}

Logger::~Logger() {
}

