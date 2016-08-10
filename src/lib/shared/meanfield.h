#ifndef MEANFIELD_SHARED_HEADER
#define MEANFIELD_SHARED_HEADER

namespace MeanField{
	class CRF{
	protected:
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
