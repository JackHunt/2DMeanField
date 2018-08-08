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

#include "meanfield_cuda.hpp"

using namespace MeanField::CUDA;
using namespace MeanField::Filtering;

CRF::CRF(int width, int height, int dimensions, float spatialSD,
    float bilateralSpatialSD, float bilateralIntensitySD, bool separable) :
    width(width), height(height), dimensions(dimensions),
    spatialWeight(1.0), bilateralWeight(1.0),
    spatialSD(spatialSD), bilateralSpatialSD(bilateralSpatialSD),
    bilateralIntensitySD(bilateralIntensitySD),
    separable(separable),
    QDistribution(width*height*dimensions, 0.0),
    QDistributionTmp(width*height*dimensions, 0.0),
    pottsModel(dimensions*dimensions, 0.0),
    gaussianOut(width*height*dimensions, 0.0),
    bilateralOut(width*height*dimensions, 0.0),
    aggregatedFilters(width*height*dimensions, 0.0),
    filterOutTmp(width*height*dimensions, 0.0),
    spatialKernel(KERNEL_SIZE),
    bilateralSpatialKernel(KERNEL_SIZE),
    bilateralIntensityKernel(KERNEL_SIZE) {

    //Initialise potts model.
    thrust::host_vector<float> pottsTmp(dimensions*dimensions);
    for (int i = 0; i < dimensions; i++) {
        for (int j = 0; j < dimensions; j++) {
            pottsTmp[i*dimensions + j] = (i == j) ? -1.0 : 0.0;
        }
    }
    pottsModel = pottsTmp;

    //Initialise kernels.
    thrust::host_vector<float> spatialKernelTmp(KERNEL_SIZE);
    thrust::host_vector<float> bilateralSpatialKernelTmp(KERNEL_SIZE);
    thrust::host_vector<float> bilateralIntensityKernelTmp(KERNEL_SIZE);
    generateGaussianKernel(&spatialKernelTmp[0], KERNEL_SIZE, spatialSD);
    generateGaussianKernel(&bilateralSpatialKernelTmp[0], KERNEL_SIZE, bilateralSpatialSD);
    generateGaussianKernel(&bilateralIntensityKernelTmp[0], KERNEL_SIZE, bilateralIntensitySD);
    spatialKernel = spatialKernelTmp;
    bilateralSpatialKernel = bilateralSpatialKernelTmp;
    bilateralIntensityKernel = bilateralIntensityKernelTmp;
}

CRF::~CRF() {
    //
}

void CRF::runInference(const unsigned char *image, const float *unaries, int iterations) {
    runInferenceIteration(image, unaries);
    for (int i = 0; i < iterations - 1; i++) {
        runInferenceIteration(image, (&QDistribution[0]).get());
    }
}

void CRF::runInferenceIteration(const unsigned char *image, const float *unaries) {
    filterGaussian(unaries);
    filterBilateral(unaries, image);
    weightAndAggregate();
    applyCompatabilityTransform();
    subtractQDistribution(unaries, (&aggregatedFilters[0]).get(), (&QDistributionTmp[0]).get());
    applySoftmax((&QDistributionTmp[0]).get(), (&QDistribution[0]).get());
}

void CRF::setSpatialWeight(float weight) {
    spatialWeight = weight;
}

void CRF::setBilateralWeight(float weight) {
    bilateralWeight = weight;
}

void CRF::reset() {
    thrust::fill(thrust::device, QDistribution.begin(), QDistribution.end(), 0.0);
    thrust::fill(thrust::device, QDistributionTmp.begin(), QDistributionTmp.end(), 0.0);
    thrust::fill(thrust::device, gaussianOut.begin(), gaussianOut.end(), 0.0);
    thrust::fill(thrust::device, bilateralOut.begin(), bilateralOut.end(), 0.0);
    thrust::fill(thrust::device, filterOutTmp.begin(), filterOutTmp.end(), 0.0);
}


const float *CRF::getQ() {
    return (&QDistribution[0]).get();
}

void CRF::filterGaussian(const float *unaries) {
    dim3 blockDim(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
    dim3 gridDim((int)ceil(width / blockDim.x), (int)ceil(height / blockDim.y));
    if (separable) {
        filterGaussianX_device << <gridDim, blockDim >> > ((&spatialKernel[0]).get(), unaries, (&filterOutTmp[0]).get(),
            spatialSD, dimensions, width, height);
        cudaDeviceSynchronize();
        filterGaussianY_device << <gridDim, blockDim >> > ((&spatialKernel[0]).get(), (&filterOutTmp[0]).get(), (&gaussianOut[0]).get(),
            spatialSD, dimensions, width, height);
    }
    else {
        filterGaussian_device << <gridDim, blockDim >> > ((&spatialKernel[0]).get(), unaries, (&gaussianOut[0]).get(),
            spatialSD, dimensions, width, height);
    }
    cudaDeviceSynchronize();
}

void CRF::filterBilateral(const float *unaries, const unsigned char *image) {
    dim3 blockDim(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
    dim3 gridDim((int)ceil(width / blockDim.x), (int)ceil(height / blockDim.y));
    if (separable) {
        filterBilateralX_device << <gridDim, blockDim >> > ((&bilateralSpatialKernel[0]).get(), (&bilateralIntensityKernel[0]).get(),
            unaries, image, (&filterOutTmp[0]).get(), bilateralSpatialSD, bilateralIntensitySD, dimensions, width, height);
        cudaDeviceSynchronize();
        filterBilateralY_device << <gridDim, blockDim >> > ((&bilateralSpatialKernel[0]).get(), (&bilateralIntensityKernel[0]).get(),
            (&filterOutTmp[0]).get(), image, (&bilateralOut[0]).get(), bilateralSpatialSD, bilateralIntensitySD, dimensions, width, height);
    }
    else {
        filterBilateral_device << <gridDim, blockDim >> > ((&bilateralSpatialKernel[0]).get(), (&bilateralIntensityKernel[0]).get(),
            unaries, image, (&bilateralOut[0]).get(), bilateralSpatialSD, bilateralIntensitySD, dimensions, width, height);
    }
    cudaDeviceSynchronize();
}

void CRF::weightAndAggregate() {
    dim3 blockDim(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
    dim3 gridDim((int)ceil(width / blockDim.x), (int)ceil(height / blockDim.y));
    weightAndAggregate_device << <gridDim, blockDim >> > ((&gaussianOut[0]).get(), (&bilateralOut[0]).get(), (&aggregatedFilters[0]).get(),
        spatialWeight, bilateralWeight, dimensions, width, height);
    cudaDeviceSynchronize();
}

void CRF::applyCompatabilityTransform() {
    dim3 blockDim(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
    dim3 gridDim((int)ceil(width / blockDim.x), (int)ceil(height / blockDim.y));
    applyCompatabilityTransform_device << <gridDim, blockDim >> > ((&pottsModel[0]).get(), (&aggregatedFilters[0]).get(),
        dimensions, width, height);
    cudaDeviceSynchronize();
}

void CRF::subtractQDistribution(const float *unaries, const float *QDist, float *out) {
    int N = width * height*dimensions;
    dim3 blockDim(CUDA_BLOCK_SIZE);
    dim3 gridDim((int)ceil(N / blockDim.x));
    subtractQDistribution_device << <gridDim, blockDim >> > (unaries, QDist, out, N);
    cudaDeviceSynchronize();
}

void CRF::applySoftmax(const float *QDist, float *out) {
    dim3 blockDim(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
    dim3 gridDim((int)ceil(width / blockDim.x), (int)ceil(height / blockDim.y));
    applySoftmax_device << <gridDim, blockDim >> > (QDist, out, dimensions, width, height);
    cudaDeviceSynchronize();
}

__global__
void filterGaussian_device(const float *kernel, const float *input, float *output, float sd, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    applyGaussianKernel(kernel, input, output, sd, dim, x, y, W, H);
}

__global__
void filterGaussianX_device(const float *kernel, const float *input, float *output, float sd, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    applyGaussianKernelX(input, output, kernel, sd, dim, x, y, W, H);
}

__global__
void filterGaussianY_device(const float *kernel, const float *input, float *output, float sd, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    applyGaussianKernelY(input, output, kernel, sd, dim, x, y, W, H);
}

__global__
void filterBilateral_device(const float *spatialKernel, const float *intensityKernel, const float *input, const unsigned char *rgb,
    float *output, float spatialSD, float intensitySD, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    applyBilateralKernel(spatialKernel, intensityKernel, input, rgb, output, spatialSD, intensitySD, dim, x, y, W, H);
}

__global__
void filterBilateralX_device(const float *spatialKernel, const float *intensityKernel, const float *input, const unsigned char *rgb,
    float *output, float spatialSD, float intensitySD, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    applyBilateralKernelX(input, output, rgb, spatialKernel, intensityKernel, spatialSD, intensitySD, dim, x, y, W, H);
}

__global__
void filterBilateralY_device(const float *spatialKernel, const float *intensityKernel, const float *input, const unsigned char *rgb,
    float *output, float spatialSD, float intensitySD, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    applyBilateralKernelY(input, output, rgb, spatialKernel, intensityKernel, spatialSD, intensitySD, dim, x, y, W, H);
}

__global__
void weightAndAggregate_device(const float *spatialOut, const float *bilateralOut, float *out, float spatialWeight, float bilateralWeight,
    int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    int idx;
    for (int i = 0; i < dim; i++) {
        idx = (x + y * W)*dim + i;
        MeanField::weightAndAggregateIndividual(spatialOut, bilateralOut, out, spatialWeight, bilateralWeight, idx);
    }
}

__global__
void applyCompatabilityTransform_device(const float *pottsModel, float *input, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    int idx = x + y * W;
    MeanField::applyCompatabilityTransformIndividual(pottsModel, input, idx, dim);
}

__global__
void subtractQDistribution_device(const float *unaries, const float *QDist, float *out, int N) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < 0 || idx >= N) {
        return;
    }

    out[idx] = unaries[idx] - QDist[idx];
}

__global__
void applySoftmax_device(const float *input, float *output, int dim, int W, int H) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < 0 || x >= W || y < 0 || y >= H) {
        return;
    }

    int idx = x + y * W;

    MeanField::applySoftmaxIndividual(input, output, idx, dim);
}
