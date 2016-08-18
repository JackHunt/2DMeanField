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
	/**
	 * \brief Applies weights to filter outputs and aggregates them.
	 *
	 * @param spatialOut Spatial filter output.
	 * @param bilateralOut Bilateral filter output.
	 * @param out Output buffer for weighted and aggregated filters.
	 * @param spatialWeight Scalar weight for spatial filter output.
	 * @param bilateralWeight Scalar weight for bilateral filter output.
	 * @param idx Linear index for tensor component.
	 */
	__SHARED_CODE__
	inline void weightAndAggregateIndividual(const float *spatialOut, const float *bilateralOut, float *out,
											 float spatialWeight, float bilateralWeight, int idx){
		out[idx] = spatialWeight*spatialOut[idx] + bilateralWeight*bilateralOut[idx];
	}

	/**
	 * \brief Applies the Potts Model compatability transform to a tensor along the third dimension.
	 *
	 * @param potts Potts model to be applied.
	 * @param out Output buffer to write transformed tensor.
	 * @param idx Linear index for slice of tensor.
	 * @param dim Third dimension of the tensor.
	 */
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

	/**
	 * \brief Applies the softmax function to the third dimension of a tensor.
	 *
	 * @param QDistribution Input distribution tensor.
	 * @param out Output buffer to write transformed distribution.
	 * @param idx Linear index for tensor slice.
	 * @param dimensions Third dimension of the tensor.
	 */
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
	
	/**
	 * \brief Base CRF interface.
	 */
	class CRF{
	protected:
		static const int KERNEL_SIZE = 256;

		/**
		 * \brief Applies a gaussian filter along each slice of the third dimension of the input
		 * \brief distribution tensor.
		 * Used for message passing.
		 * Output stored in internal buffer.
		 *
		 * @param unaries Input distribution tensor.
		 */
		virtual void filterGaussian(const float *unaries) = 0;

		/**
		 * \brief Applies a bilateral filter along each slice of the third dimension of the input
		 * \brief distribution tensor.
		 * Used for message passing.
		 * Output stored in internal buffer.
		 *
		 * @param unaries Input distribution tensor.
		 * @param image RGB image used for intensity term in filter.
		 */
		virtual void filterBilateral(const float *unaries, const unsigned char *image) = 0;

		/**
		 * \brief Takes the outputs of the gaussian and bilateral filters, weights and aggregates them.
		 * Output stored in internal buffer.
		 */
		virtual void weightAndAggregate() = 0;

		/**
		 * \brief Applies the Potts Model compatability transform to the aggregated filter outputs.
		 * Transform is applied in place.
		 */
		virtual void applyCompatabilityTransform() = 0;

		/**
		 * \brief Subtracts the current Q distribution from the input unaries.
		 *
		 * @param unaries Input unaries.
		 * @param QDist Current Q distribution.
		 * @param out Output buffer to write output.
		 */
		virtual void subtractQDistribution(const float *unaries, const float *QDist, float *out) = 0;

		/**
		 * \brief Applies the Softmax function to the current distribution tensor.
		 * Applies softmax along the third dimension of the tensor for each point in the plane of the
		 * first and second dimension.
		 *
		 * @param QDist Current Q distribution.
		 * @param out Output to write to.
		 */
		virtual void applySoftmax(const float *QDist, float *out) = 0;
		
	public:
		/**
		 * \brief Runs inference for a given number of iterations.
		 *
		 * @param image Input RGB image.
		 * @param unaries Input unary potentials.
		 * @param iterations Number of iterations to run for.
		 */
		virtual void runInference(const unsigned char *image, const float *unaries, int iterations) = 0;

		/**
		 * \brief Run a single Mean Field iteration.
		 *
		 * @param image Input RGB Image.
		 * @param unaries Input unary potentials.
		 */
		virtual void runInferenceIteration(const unsigned char *image, const float *unaries) = 0;

		/**
		 * \brief Sets the spatial weight to be used in the aggregation stage.
		 *
		 * @param weight Scalar weight value.
		 */
		virtual void setSpatialWeight(float weight) = 0;

		/**
		 * \brief Sets the bilateral weight to be used in the aggregation stage.
		 *
		 * @param weight Scalar weight value.
		 */
		virtual void setBilateralWeight(float weight) = 0;

		/**
		 * \brief Reset internal buffers/
		 */
		virtual void reset() = 0;

		/**
		 * \brief Return pointer to the current Q distribution.
		 *
		 * @return Q distribution pointer.
		 */
		virtual float *getQ() = 0;
		virtual ~CRF(){}
	};
}

#endif
