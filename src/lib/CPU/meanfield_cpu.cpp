#include "meanfield_cpu.h"

using namespace MeanField::CPU;
using namespace MeanField::Filtering;

CRF::CRF(int width, int height, int dimensions, float spatialSD,
		 float bilateralSpatialSD, float bilateralIntensitySD) :
	width(width), height(height), dimensions(dimensions),
	spatialSD(spatialSD), bilateralSpatialSD(bilateralSpatialSD),
	bilateralIntensitySD(bilateralIntensitySD),
	QDistribution(new float[width*height*dimensions]()),
	QDistributionTmp(new float[width*height*dimensions]()),
	pottsModel(new float[dimensions*dimensions]()),
	gaussianOut(new float[width*height*dimensions]()),
	bilateralOut(new float[width*height*dimensions]()),
	aggregatedFilters(new float[width*height*dimensions]()),
	spatialKernel(new float[KERNEL_SIZE]()),
	bilateralSpatialKernel(new float[KERNEL_SIZE]()),
	bilateralIntensityKernel(new float[KERNEL_SIZE]()){

	//Initialise potts model.
	for(int i=0; i<dimensions; i++){
		for(int j=0; j<dimensions; j++){
			pottsModel[i*dimensions + j] = (i == j) ? -1.0 : 0.0;
		}
	}

	//Initialise kernels.
	generateGaussianKernel(spatialKernel.get(), KERNEL_SIZE, spatialSD);
	generateGaussianKernel(bilateralSpatialKernel.get(), KERNEL_SIZE, bilateralSpatialSD);
	generateGaussianKernel(bilateralIntensityKernel.get(), KERNEL_SIZE, bilateralIntensitySD);
}

CRF::~CRF(){
	//
}
			
void CRF::runInference(const unsigned char *image, const float *unaries, int iterations){
	for(int i=0; i<iterations; i++){
		runInferenceIteration(image, unaries);
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
	for(int i=0; i<width*height*dimensions; i++){
		QDistribution[i] = 0.0;
		QDistributionTmp[i] = 0.0;
		gaussianOut[i] = 0.0;
		bilateralOut[i] = 0.0;
	}
}


float *CRF::getQ(){
	return QDistribution.get();
}

void CRF::filterGaussian(const float *unaries){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			applyGaussianKernel(spatialKernel.get(), unaries, gaussianOut.get(), spatialSD, dimensions, i, j, width, height);
		}
	}
}

void CRF::filterBilateral(const float *unaries, const unsigned char *image){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			applyBilateralKernel(bilateralSpatialKernel.get(), bilateralIntensityKernel.get(), unaries,
								 image, bilateralOut.get(), bilateralSpatialSD, bilateralIntensitySD,
								 dimensions, i, j, width, height);
		}
	}
}

void CRF::weightAndAggregate(){
	for(int i=0; i<height; i++){
		for(int i=0; i<width; i++){
			//
		}
	}
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
