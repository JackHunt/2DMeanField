#ifndef MEANFIELD_SHARED_HEADER
#define MEANFIELD_SHARED_HEADER

#include "gaussian_filter.h"
#include "bilateral_filter.h"

namespace MeanField{
	inline void weightAndAggregateIndividual(const float *spatialOut, const float *bilateralOut, float *out,
											 float spatialWeight, float bilateralWeight, int idx){
		out[idx] = spatialWeight*spatialOut[idx] + bilateralWeight*bilateralOut[idx];
	}

	inline void applyCompatabilityTransformIndividual(const float *potts, float *out, int idx, int dim){
		for(int i=0; i<dim; i++){
			for(int j=0; j<dim; j++){
				out[idx*dim + j] *= potts[i*dim + j];
			}
		}
	}

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
