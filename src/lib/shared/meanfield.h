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

#ifndef MEANFIELD_SHARED_HEADER
#define MEANFIELD_SHARED_HEADER

#include "code_sharing.h"
#include "gaussian_filter.h"
#include "bilateral_filter.h"

namespace MeanField{
	__SHARED_CODE__
	inline void weightAndAggregateIndividual(const float *spatialOut, const float *bilateralOut, float *out,
											 float spatialWeight, float bilateralWeight, int idx){
		out[idx] = spatialWeight*spatialOut[idx] + bilateralWeight*bilateralOut[idx];
	}

	__SHARED_CODE__
	inline void applyCompatabilityTransformIndividual(const float *potts, float *out, int idx, int dim){
		float *perDimSum = new float[dim];
		for(int i=0; i<dim; i++){
			perDimSum[i] = 0.0;
		}
		
		for(int i=0; i<dim; i++){
			for(int j=0; j<dim; j++){
				perDimSum[i] += out[idx*dim + j]*potts[i*dim + j];
			}
		}

		for(int i=0; i<dim; i++){
			out[idx*dim + i] = perDimSum[i];
		}
		delete[] perDimSum;
	}

	__SHARED_CODE__
	inline void applySoftmaxIndividual(const float *QDistribution, float *out, int idx, int dimensions){
		float normaliser = 0.0;
		int localIdx;
		for(int i=0; i<dimensions; i++){
			localIdx = idx*dimensions + i;
			out[localIdx] = expf(QDistribution[localIdx]);
			normaliser += out[localIdx];
		}

		for(int i=0; i<dimensions; i++){
			out[idx*dimensions + i] /= normaliser;
		}
	}
	
	class CRF{
	protected:
		static const int KERNEL_SIZE = 256;
		
		virtual void filterGaussian(const float *unaries) = 0;
		virtual void filterBilateral(const float *unaries, const unsigned char *image) = 0;
		virtual void weightAndAggregate() = 0;
		virtual void applyCompatabilityTransform() = 0;
		virtual void subtractQDistribution(const float *unaries, const float *QDist, float *out) = 0;
		virtual void applySoftmax(const float *QDist, float *out) = 0;
		
	public:
		virtual void runInference(const unsigned char *image, const float *unaries, int iterations) = 0;
		virtual void runInferenceIteration(const unsigned char *image, const float *unaries) = 0;
		virtual void setSpatialWeight(float weight) = 0;
		virtual void setBilateralWeight(float weight) = 0;
		virtual void reset() = 0;
		virtual float *getQ() = 0;
		virtual ~CRF(){}
	};
}

#endif
