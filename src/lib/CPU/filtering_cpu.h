#ifndef MEANFIELD_FILTERING_CPU_HEADER
#define MEANFIELD_FILTERING_CPU_HEADER

#include "math.h"

namespace MeanField {
	namespace CPU {
		namespace Filtering {
			struct GaussianFilterSeparable {
				/**
				 * @brief applyXDirection Applies separable Gaussian filter in X direction.
				 * @param input Input tensor.
				 * @param output Output tensor buffer
				 * @param kernel Gaussian kernel to be applied.
				 * @param sd Standard Deviation of the given kernel.
				 * @param dim Third dimension of the tensor.
				 * @param x x location in the plane given by the first and second dimensions.
				 * @param y y location in the plane given by the first and second dimensions.
				 * @param W Second dimension of the tensor.
				 * @param H First dimension of the tensor.
				 */
				static void applyXDirection(const float *input, float *output, const float *kernel, float sd,
					int dim, int x, int y, int W, int H);

				/**
				 * @brief applyYDirection Applies separable Gaussian filter in Y direction.
				 * @param input Input tensor.
				 * @param output Output tensor buffer
				 * @param kernel Gaussian kernel to be applied.
				 * @param sd Standard Deviation of the given kernel.
				 * @param dim Third dimension of the tensor.
				 * @param x x location in the plane given by the first and second dimensions.
				 * @param y y location in the plane given by the first and second dimensions.
				 * @param W Second dimension of the tensor.
				 * @param H First dimension of the tensor.
				 */
				static void applyYDirection(const float *input, float *output, const float *kernel, float sd,
					int dim, int x, int y, int W, int H);
			};

			struct BilateralFilterSeparable {
				/**
				 * @brief applyXDirection Applies separable approximation to the Bilateral filter in the X direction.
				 * @param input Input Tensor.
				 * @param output Output tensor buffer.
				 * @param rgb RGB image for intensity factor.
				 * @param spatialKernel Spatial(gaussian) kernel.
				 * @param intensityKernel Bilateral kernel.
				 * @param spatialSD Standard deviation of the given Gaussian kernel.
				 * @param intensitySD Standard deviation of the given Bilateral kernel.
				 * @param dim Third dimension of the tensor.
				 * @param x x location in the plane given by the first and second dimensions.
				 * @param y y location in the plane given by the first and second dimensions.
				 * @param W Second dimension of the tensor.
				 * @param H First dimension of the tensor.
				 */
				static void applyXDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
					const float *intensityKernel, float spatialSD, float intensitySD, int dim,
					int x, int y, int W, int H);

				/**
				 * @brief applyYDirection Applies separable approximation to the Bilateral filter in the Y direction.
				 * @param input Input Tensor.
				 * @param output Output tensor buffer.
				 * @param rgb RGB image for intensity factor.
				 * @param spatialKernel Spatial(gaussian) kernel.
				 * @param intensityKernel Bilateral kernel.
				 * @param spatialSD Standard deviation of the given Gaussian kernel.
				 * @param intensitySD Standard deviation of the given Bilateral kernel.
				 * @param dim Third dimension of the tensor.
				 * @param x x location in the plane given by the first and second dimensions.
				 * @param y y location in the plane given by the first and second dimensions.
				 * @param W Second dimension of the tensor.
				 * @param H First dimension of the tensor.
				 */
				static void applyYDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
					const float *intensityKernel, float spatialSD, float intensitySD, int dim,
					int x, int y, int W, int H);
			};
		}
	}
}

#endif
