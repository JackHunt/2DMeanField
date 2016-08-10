#ifndef MEANFIELD_CPU_HEADER
#define MEANFIELD_CPU_HEADER

#include "../shared/meanfield.h"
#include <memory

namespace MeanField{
	namespace CPU{
		class CRF : public MeanField::CRF{
		private:
			int width, height, dimensions;
			std::unique_ptr<float[]> Q_distribution, Q_distribution_tmp;
			std::unique_ptr<float[]> potts_model;
			std::unique_ptr<float[]> gaussian_out, bilateral_out, aggregated_filters;

		protected:
			void filterGaussian(const float *unaries);
			void filterBilateral(const float *unaries, const unsigned char *image);
			void weightAndAggregate();
			void applyCompatabilityTransform();
			void subtractQDistribution();
			void applySoftmax();
			
		public:
			CRF();
			~CRF();
			
			void runInference(const unsigned char *image, const float *unaries, int iterations);
			void runInferenceIteration(const unsigned char *image, const float *unaries);
			void reset();
			float *getQ();
		};
	}
}

#endif
