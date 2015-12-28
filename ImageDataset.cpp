/*
 * ImageDataset.cpp
 *
 *  Created on: Nov 22, 2015
 *      Author: jonas
 */

#include "ImageDataset.h"
#include "Logger.h"
#include <fstream>
#include <inttypes.h>
#include "time.h"

ImageDataset::ImageDataset() {
	randgen.seed(time(NULL));
}

void ImageDataset::loadImagesFromFile(std::string imagefile) {
	std::ifstream infile(imagefile, std::ifstream::in | std::ifstream::binary);
	if (infile.is_open()) {
		uint8_t magic_buf[4];
		uint8_t dataset_size_buf[4];
		uint32_t magic = 0;
		uint32_t dataset_size = 0;
		infile.read((char *) &magic_buf, 4);
		infile.read((char *) &dataset_size_buf, 4);
		for (unsigned int i = 0; i < 4; i++) {
			magic |= ((uint32_t) magic_buf[i]) << (8*(3-i));
			dataset_size |= ((uint32_t) dataset_size_buf[i]) << (8*(3-i));
		}
		if (magic == 0x00000803) {
			uint8_t imagebuffer[IMAGESIZE*IMAGESIZE];
			for (unsigned int i = 0; i < dataset_size; i++) {
				infile.read((char *) &imagebuffer, IMAGESIZE*IMAGESIZE);
				std::vector<float> newimage(64, -0.1f);
				for (unsigned int y = 0; y < IMAGESIZE; y++) {
					newimage.push_back(-0.1f);
					newimage.push_back(-0.1f);
					for (unsigned int x = 0; x < IMAGESIZE; x++) {
						newimage.push_back(-0.1f + 1.275f * ((float) imagebuffer[y*IMAGESIZE + x])/255.0f);
					}
					newimage.push_back(-0.1f);
					newimage.push_back(-0.1f);
				}
				for (unsigned int j = 0; j < 64; j++) newimage.push_back(-0.1f);
				data.push_back(newimage);
			}
			Logger::writeLine("ImageDataset::loadImagesFromFile(): Loaded image file with " + std::to_string(dataset_size) + " images.");
		} else {
			Logger::writeLine("ImageDataset::loadImagesFromFile(): Invalid image file, no data loaded.");
		}
	} else {
		Logger::writeLine("ImageDataset::loadImagesFromFile(): File not found: \"" + imagefile + "\".");
	}
}

void ImageDataset::loadLabelsFromFile(std::string labelfile) {
	if (data.size() > 0) {
		std::ifstream infile(labelfile, std::ifstream::in | std::ifstream::binary);
		if (infile.is_open()) {
			uint8_t magic_buf[4];
			uint8_t dataset_size_buf[4];
			uint32_t magic = 0;
			uint32_t dataset_size = 0;
			infile.read((char *) &magic_buf, 4);
			infile.read((char *) &dataset_size_buf, 4);
			for (unsigned int i = 0; i < 4; i++) {
				magic |= ((uint32_t) magic_buf[i]) << (8*(3-i));
				dataset_size |= ((uint32_t) dataset_size_buf[i]) << (8*(3-i));
			}
			if (magic == 0x00000801) {
				if (dataset_size == data.size()) {
					uint8_t label;
					for (unsigned int i = 0; i < dataset_size; i++) {
						infile.read((char *) &label, 1);
						labels.push_back(label);
					}
					Logger::writeLine("ImageDataset::loadLabelsFromFile(): Loaded label file with " + std::to_string(dataset_size) + " labels.");
				} else {
					Logger::writeLine("ImageDataset::loadLabelsFromFile(): Label file not suitable for this data set (number of labels not matching).");
				}
			} else {
				Logger::writeLine("ImageDataset::loadLabelsFromFile(): Invalid label file, no data loaded.");
			}
		} else {
			Logger::writeLine("ImageDataset::loadLabelsFromFile(): File not found: \"" + labelfile + "\".");
		}
	} else {
		Logger::writeLine("ImageDataset::loadLabelsFromFile(): No image data loaded, load image data before assigning labels.");
	}
}

unsigned int ImageDataset::getSize() const {
	return data.size();
}

std::pair<std::vector<float>, uint8_t> ImageDataset::popRandomElementWithLabel() {
	std::uniform_int_distribution<unsigned int> dist(0, data.size() - 1);
	unsigned int randindex = dist(randgen);
	std::vector<float> randelem = data[randindex];
	std::swap(data[randindex], data.back());
	data.pop_back();
	uint8_t randlabel = labels[randindex];
	std::swap(labels[randindex], labels.back());
	labels.pop_back();
	return std::make_pair(randelem, randlabel);
}

std::vector<float> ImageDataset::operator[](unsigned int id) const {
	return data[id];
}

uint8_t ImageDataset::operator()(unsigned int id) const {
	return labels[id];
}

ImageDataset::~ImageDataset() {

}

