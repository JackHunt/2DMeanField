#ifndef MEANFIELD_SHARED_HEADER
#define MEANFIELD_SHARED_HEADER

#include "gaussian_filter.h"
#include "bilateral_filter.h"

namespace MeanField{
	class CRF{
	protected:
		const int KERNEL_SIZE = 256;
		
		virtual void filterGaussian(const float *unaries) = 0;
		virtual void filterBilateral(const float *unaries
									 , const unsigned char *image) = 0;
		virtual void weightAndAggregate() = 0;
		virtual void applyCompatabilityTransform() = 0;
		virtual void subtractQDistribution() = 0;
		virtual void applySoftmax() = 0;
		
	public:
		virtual void runInference(const unsigned char *image, const float *unaries, int iterations) = 0;
		virtual void runInferenceIteration(const unsigned char *image, const float *unaries) = 0;
		virtual void reset() = 0;
		virtual float *getQ() = 0;
		virtual ~CRF(){}
	};
}

#endif
