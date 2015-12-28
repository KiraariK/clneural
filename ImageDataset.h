/*
 * ImageDataset.h
 *
 *  Created on: Nov 22, 2015
 *      Author: jonas
 */

#ifndef IMAGEDATASET_H_
#define IMAGEDATASET_H_

#include <string>
#include <vector>
#include <random>

#define IMAGESIZE 28

class ImageDataset {
private:
	std::vector<std::vector<float>> data;
	std::vector<uint8_t> labels;
	std::default_random_engine randgen;
public:
	ImageDataset();
	void loadImagesFromFile(std::string imagefile);
	void loadLabelsFromFile(std::string labelfile);
	std::pair<std::vector<float>, uint8_t> popRandomElementWithLabel();
	std::vector<float> operator[](unsigned int id) const;
	uint8_t operator()(unsigned int id) const;
	unsigned int getSize() const;
	virtual ~ImageDataset();
};

#endif /* IMAGEDATASET_H_ */
