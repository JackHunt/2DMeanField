/*
 Copyright (c) 2016, Jack Miles Hunt
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
 * Neither the name of Jack Miles Hunt nor the
      names of contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Jack Miles Hunt BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "meanfield_cpu.h"

using namespace MeanField::CPU;
using namespace MeanField::Filtering;
using namespace MeanField::CPU::Filtering;

CRF::CRF(int width, int height, int dimensions, float spatialSD,
         float bilateralSpatialSD, float bilateralIntensitySD, bool separable) :
    width(width), height(height), dimensions(dimensions),
    spatialWeight(1.0), bilateralWeight(1.0),
    spatialSD(spatialSD), bilateralSpatialSD(bilateralSpatialSD),
    bilateralIntensitySD(bilateralIntensitySD),
    separable(separable),
    QDistribution(new float[width*height*dimensions]()),
    QDistributionTmp(new float[width*height*dimensions]()),
    pottsModel(new float[dimensions*dimensions]()),
    gaussianOut(new float[width*height*dimensions]()),
    bilateralOut(new float[width*height*dimensions]()),
    aggregatedFilters(new float[width*height*dimensions]()),
    filterOutTmp(new float[width*height*dimensions]()),
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


const float *CRF::getQ(){
    return QDistribution.get();
}

void CRF::filterGaussian(const float *unaries){
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(separable){
                GaussianFilterSeparable::applyXDirection(unaries, filterOutTmp.get(), spatialKernel.get(), spatialSD,
                                                         dimensions, j, i, width, height);
                GaussianFilterSeparable::applyYDirection(filterOutTmp.get(), gaussianOut.get(), spatialKernel.get(), spatialSD,
                                                         dimensions, j, i, width, height);
            }else{
                applyGaussianKernel(spatialKernel.get(), unaries, gaussianOut.get(), spatialSD, dimensions, j, i, width, height);
            }
        }
    }
}

void CRF::filterBilateral(const float *unaries, const unsigned char *image){
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(separable){
                BilateralFilterSeparable::applyXDirection(unaries, filterOutTmp.get(), image, bilateralSpatialKernel.get(),
                                                          bilateralIntensityKernel.get(), bilateralSpatialSD, bilateralIntensitySD,
                                                          dimensions, j, i, width, height);
                BilateralFilterSeparable::applyYDirection(filterOutTmp.get(), bilateralOut.get(), image, bilateralSpatialKernel.get(),
                                                          bilateralIntensityKernel.get(), bilateralSpatialSD, bilateralIntensitySD,
                                                          dimensions, j, i, width, height);
            }else{
                applyBilateralKernel(bilateralSpatialKernel.get(), bilateralIntensityKernel.get(), unaries,
                                     image, bilateralOut.get(), bilateralSpatialSD, bilateralIntensitySD,
                                     dimensions, j, i, width, height);
            }
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
