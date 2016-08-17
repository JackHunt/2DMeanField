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

__global__
void filterGaussian_device(const float *kernel, const float *input, float *output, float sd, int dim, int W, int H);

__global__
void filterBilateral_device(const float *spatialKernel, const float *intensityKernel, const float *input, const unsigned char *rgb,
		float *output, float spatialSD, float intensitySD, int dim, int W, int H);

__global__
void weightAndAggregate_device(const float *spatialOut, const float *bilateralOut, float *out, float spatialWeight, float bilateralWeight,
		int dim, int W, int H);

__global__
void applyCompatabilityTransform_device(const float *pottsModel, float *input, int dim, int W, int H);

__global__
void subtractQDistribution_device(const float *unaries, const float *QDist, float *out, int N);

__global__
void applySoftmax_device(const float *input, float *output, int dim, int W, int H);

namespace MeanField{
	namespace CUDA{
		class CRF : public MeanField::CRF{
		private:
			int width, height, dimensions;
			float spatialSD, bilateralSpatialSD, bilateralIntensitySD;
			float spatialWeight, bilateralWeight;
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
			CRF(int width, int height, int dimensions, float spatial_sd,
				float bilateral_spatial_sd, float bilateral_intensity_sd);
			~CRF();
			
			void runInference(const unsigned char *image, const float *unaries, int iterations);
			void runInferenceIteration(const unsigned char *image, const float *unaries);
			void setSpatialWeight(float weight);
			void setBilateralWeight(float weight);
			void reset();
			float *getQ();
		};
	}
}

#endif
