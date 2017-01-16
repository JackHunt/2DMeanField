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

#ifndef MEANFIELD_CUDA_HEADER
#define MEANFIELD_CUDA_HEADER


#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

#include "cuda_util.h"
#include "../shared/meanfield.h"

/**
 * \brief CUDA kernel to apply gaussian filter.
 *
 * @param kernel Gaussian kernel to be applied.
 * @param input Input distribution tensor/
 * @param output Output buffer to write to.
 * @param sd Standard deviation of the filter.
 * @param dim Third dimension of the tensor.
 * @param W Second dimension of the tensor.
 * @param H First dimension of the tensor.
 */
__global__
void filterGaussian_device(const float *kernel, const float *input, float *output, float sd, int dim, int W, int H);

/**
 * \brief CUDA kernel to apply bilateral filter.
 *
 * @param spatialKernel Gaussian kernel to be applied for spatial component.
 * @param intensityKernel Gaussian kernel to be applied for intensity component.
 * @param input Input distribution tensor.
 * @param rgb RGB image used for intensity filtering.
 * @param output Output buffer to write to.
 * @param spatialSD Standard deviation of spatial kernel.
 * @param intensitySD Standard deviation of intensity kernel.
 * @param dim Third dimension of the tensor.
 * @param W Second dimension of the tensor.
 * @param H First dimension of the tensor.
 */
__global__
void filterBilateral_device(const float *spatialKernel, const float *intensityKernel, const float *input, const unsigned char *rgb,
                            float *output, float spatialSD, float intensitySD, int dim, int W, int H);

/**
 * \brief CUDA kernel to weight and aggregate the outputs of the spatial and bilateral filters.
 *
 * @param spatialOut Output of the spatial filter.
 * @param bilateralOut Output of the bilateral filter.
 * @param out Output buffer to write to.
 * @param spatialWeight Scalar weight for the spatial filter.
 * @param bilateralWeight Scalar weight for the bilateral filter.
 * @param dim Third dimension of the tensor.
 * @param W Second dimension of the tensor.
 * @param H First dimension of the tensor.
 */
__global__
void weightAndAggregate_device(const float *spatialOut, const float *bilateralOut, float *out, float spatialWeight, float bilateralWeight,
                               int dim, int W, int H);

/**
 * \brief CUDA kernel to apply Potts Model compatability transform.
 * Transform applied in place.
 *
 * @param pottsModel Potts Model to be applied.
 * @param input Input distribution tensor.
 * @param dim Third dimension of the tensor.
 * @param W Second dimension of the tensor.
 * @param H First dimension of the tensor.
 */
__global__
void applyCompatabilityTransform_device(const float *pottsModel, float *input, int dim, int W, int H);

/**
 * \brief CUDA kernel to subtract one tensor from another. Typically Q distribution from unary potentials.
 *
 * @param unaries Input unary potentials.
 * @param QDist Input Q distribution/
 * @param out Output buffer to write to.
 * @param N Total number of elements in the tensor(s).
 */
__global__
void subtractQDistribution_device(const float *unaries, const float *QDist, float *out, int N);

/**
 * \brief CUDA kernel to apply Softmax function to a distribution tensor.
 *
 * @param input Input distribution tensor.
 * @param output Output to write to.
 * @param dim Third dimension of the tensor.
 * @param W Second dimension of the tensor.
 * @param H First dimension of the tensor.
 */
__global__
void applySoftmax_device(const float *input, float *output, int dim, int W, int H);

namespace MeanField{
namespace CUDA{
class CRF : public MeanField::CRF{
private:
    int width, height, dimensions;
    float spatialSD, bilateralSpatialSD, bilateralIntensitySD;
    float spatialWeight, bilateralWeight;
    bool separable;
    thrust::device_vector<float> QDistribution, QDistributionTmp;
    thrust::device_vector<float> pottsModel;
    thrust::device_vector<float> gaussianOut, bilateralOut, aggregatedFilters;
    thrust::device_vector<float> spatialKernel, bilateralSpatialKernel, bilateralIntensityKernel;

protected:
    void filterGaussian(const float *unaries);
    void filterBilateral(const float *unaries, const unsigned char *image);
    void weightAndAggregate();
    void applyCompatabilityTransform();
    void subtractQDistribution(const float *unaries, const float *QDist, float *out);
    void applySoftmax(const float *QDist, float *out);

public:
    /**
                         * \brief Constructs a new CRF with the given configuration.
                         * As this is a GPU implementation, all pointers provided to member functions must point to
                         * the GPU memory space.
                         *
                         * All pointers returned from member functions shall point to GPU memory space.
                         *
                         * @param width Width of the input image.
                         * @param height Height of the input image.
                         * @param dimensions Number of classes.
                         * @param spatial_sd Standard deviation for spatial filtering for message passing.
                         * @param bilateral_spatial_sd Standard deviation for spatial component of bilateral filtering for message passing.
                         * @param bilateral_intensity_sdStandard deviation for intensity component of bilateral filtering for message passing.
                         */
    CRF(int width, int height, int dimensions, float spatial_sd,
        float bilateral_spatial_sd, float bilateral_intensity_sd, bool separable = true);
    ~CRF();

    void runInference(const unsigned char *image, const float *unaries, int iterations);
    void runInferenceIteration(const unsigned char *image, const float *unaries);
    void setSpatialWeight(float weight);
    void setBilateralWeight(float weight);
    void reset();
    const float *getQ();
};
}
}

#endif
