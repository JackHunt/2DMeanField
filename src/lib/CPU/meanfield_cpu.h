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

#ifndef MEANFIELD_CPU_HEADER
#define MEANFIELD_CPU_HEADER

#include "../shared/meanfield.h"
#include <memory>

namespace MeanField{
	namespace CPU{
		class CRF : public MeanField::CRF{
		private:
			int width, height, dimensions;
			float spatialSD, bilateralSpatialSD, bilateralIntensitySD;
			float spatialWeight, bilateralWeight;
			std::unique_ptr<float[]> QDistribution, QDistributionTmp;
			std::unique_ptr<float[]> pottsModel;
			std::unique_ptr<float[]> gaussianOut, bilateralOut, aggregatedFilters;
			std::unique_ptr<float[]> spatialKernel, bilateralSpatialKernel, bilateralIntensityKernel;

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
			 * As this is a CPU implementation, all pointers provided to member functions must point to
			 * the CPU memory space.
			 *
			 * All pointers returned from member functions shall point to CPU memory space.
			 *
			 * @param width Width of the input image.
			 * @param height Height of the input image.
			 * @param dimensions Number of classes.
			 * @param spatial_sd Standard deviation for spatial filtering for message passing.
			 * @param bilateral_spatial_sd Standard deviation for spatial component of bilateral filtering for message passing.
			 * @param bilateral_intensity_sdStandard deviation for intensity component of bilateral filtering for message passing.
			 */
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
