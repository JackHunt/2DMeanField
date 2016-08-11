#ifndef MEANFIELD_DEMO_APP_HEADER
#define MEANFIELD_DEMO_APP_HEADER

#include <iostream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../lib/2DMEanField.h"

namespace MeanFieldDemo{
	int noColours = 0;
	int colourMap[255] = {0};
	const int M = 21;//VOC classes
	
//	void unariesFromLabelling(const int *inputLabelling, float *outputUnary, int W, int H, int dim)
//	void labellingFromUnaries(const float *inputUnaries, int *outputLabelling, int W, int H, int dim);
//	void labellingToImage(cv::Mat &outputImage, const int *labelling, int W, int H);
//	void readLabellingFromImage(int *outputLabelling, const cv::Mat &image, int W, int H, int dim);
	//int colourToIndex(const cv::Vec3b &colour);
	//cv::Vec3b indexToColour(int idx);
}

#endif
