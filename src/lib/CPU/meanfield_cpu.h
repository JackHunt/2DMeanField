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
