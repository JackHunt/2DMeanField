#include "meanfield_cpu.h"

using namespace MeanField::CPU;
using namespace MeanField::Filtering;

CRF::CRF(int width, int height, int dimensions, float spatialSD,
		 float bilateralSpatialSD, float bilateralIntensitySD) :
	width(width), height(height), dimensions(dimensions),
	spatialWeight(1.0), bilateralWeight(1.0),
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
	runInferenceIteration(image, unaries);
	for(int i=0; i<iterations-1; i++){
		runInferenceIteration(image, QDistribution.get());
	}
}

void CRF::runInferenceIteration(const unsigned char *image, const float *unaries){
	filterGaussian(unaries);
	filterBilateral(unaries, image);
	weightAndAggregate();
	applyCompatabilityTransform();
	subtractQDistribution(unaries, aggregatedFilters.get(), QDistributionTmp.get());
	applySoftmax(QDistributionTmp.get(), QDistribution.get());
}

void CRF::setSpatialWeight(float weight){
	spatialWeight = weight;
}

void CRF::setBilateralWeight(float weight){
	bilateralWeight = weight;
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
			applyGaussianKernel(spatialKernel.get(), unaries, gaussianOut.get(), spatialSD, dimensions, j, i, width, height);
		}
	}
}

void CRF::filterBilateral(const float *unaries, const unsigned char *image){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			applyBilateralKernel(bilateralSpatialKernel.get(), bilateralIntensityKernel.get(), unaries,
								 image, bilateralOut.get(), bilateralSpatialSD, bilateralIntensitySD,
								 dimensions, j, i, width, height);
		}
	}
}

void CRF::weightAndAggregate(){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			for(int k=0; k<dimensions; k++){
				int idx = (i*width + j)*dimensions + k;
				weightAndAggregateIndividual(gaussianOut.get(), bilateralOut.get(), aggregatedFilters.get(),
											 spatialWeight, bilateralWeight, idx);
			}
		}
	}
}

void CRF::applyCompatabilityTransform(){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			int idx = i*width + j;
			applyCompatabilityTransformIndividual(pottsModel.get(), aggregatedFilters.get(), idx, dimensions);
		}
	}
}

void CRF::subtractQDistribution(const float *unaries, const float *QDist, float *out){
	for(int i=0; i<width*height*dimensions; i++){
		out[i] = unaries[i] - QDist[i];
	}
}

void CRF::applySoftmax(const float *QDist, float *out){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			int idx = i*width + j;
			applySoftmaxIndividual(QDist, out, idx, dimensions);
		}
	}
}
