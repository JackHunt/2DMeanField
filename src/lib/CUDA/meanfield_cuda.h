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
