/*
 * Logger.h
 *
 *  Created on: Sep 12, 2015
 *      Author: jonas
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#include <string>

class Logger {
private:
	Logger();
	virtual ~Logger();
public:
	static void writeLine(std::string logline);
	static void writeLineNotime(std::string logline);
};

#endif /* LOGGER_H_ */
