#include "meanfield_cpu.h"

using namespace MeanField::CPU;

CRF::CRF(int width, int height, int dimensions) :
	width(width), height(height), dimensions(dimensions),
	Q_distribution(new float[width*height*dimensions]()),
	Q_distribution_tmp(new float[width*height*dimensions]()),
	potts_model(new float[dimensions*dimensions]())
	gaussian_out(new float[width*height*dimensions]()),
	bilateral_out(new float[width*height*dimensions]()),
	aggregated_filters(new float[width*height*dimensions]()){

	//Initialise potts model.
	for(int i=0; i<dimensions; i++){
		for(int j=0; j<dimensions; j++){
			potts_model[i*dimensions + j] = (i == j) ? -1.0 : 0.0;
		}
	}
}

CRF::~CRF(){
	//
}
			
void CRF::runInference(const unsigned char *image, const float *unaries, int iterations){
	for(int i=0; i<iterations; i++){
		runInferenceIteration(image, unaries, iterations);
	}
}

void CRF::runInferenceIteration(const unsigned char *image, const float *unaries){
	filterGaussian(unaries);
	filterBilateral(unaries, image);
	weightAndAggregate();
	applyCompatabilityTransform();
	subtractQDistribution();
	applySoftmax();
}


void CRF::reset(){
	for(int i=0; i<width*height*dimensions){
		Q_distribution[i] = 0.0;
		Q_distribution_tmp[i] = 0.0;
		gaussian_out[i] = 0.0;
		bilateral_out[i] = 0.0;
	}
}


float *CRF::getQ(){
	return Q_distribution.get();
}

void CRF::filterGaussian(const float *unaries){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			//
		}
	}
}


void CRF::filterBilateral(const float *unaries, const unsigned char *image){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			//
		}
	}
}

void CRF::weightAndAggregate(){
	//
}

void CRF::applyCompatabilityTransform(){
	//
}

void CRF::subtractQDistribution(){
	//
}

void CRF::applySoftmax(){
	//
}
