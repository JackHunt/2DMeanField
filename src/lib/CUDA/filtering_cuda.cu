#include "filtering_cuda.h"

using namespace MeanField::CUDA::Filtering;

void GaussianFilterSeparable::applyXDirection(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H) {
	dim3 blockDims(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
	dim3 gridDims(static_cast<int>(ceil(static_cast<float>(W) / static_cast<float>(blockDims.x))), static_cast<int>(ceil(static_cast<float>(H) / static_cast<float>(blockDims.y))));
	filterGaussianX_device << <gridDims, blockDims >> > (input, output, kernel, sd, dim, W, H);
}

void GaussianFilterSeparable::applyYDirection(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H) {
	dim3 blockDims(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
	dim3 gridDims(static_cast<int>(ceil(static_cast<float>(W) / static_cast<float>(blockDims.x))), static_cast<int>(ceil(static_cast<float>(H) / static_cast<float>(blockDims.y))));
	filterGaussianY_device << <gridDims, blockDims >> > (input, output, kernel, sd, dim, W, H);
}

void BilateralFilterSeparable::applyXDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H) {
	dim3 blockDims(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
	dim3 gridDims(static_cast<int>(ceil(static_cast<float>(W) / static_cast<float>(blockDims.x))), static_cast<int>(ceil(static_cast<float>(H) / static_cast<float>(blockDims.y))));
	int maxSD = (spatialSD > intensitySD) ? spatialSD : intensitySD;
	filterBilateralX_device << <gridDims, blockDims >> > (input, output, rgb, spatialKernel, intensityKernel,
		spatialSD, intensitySD, dim, W, H);
}

void BilateralFilterSeparable::applyYDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H) {
	dim3 blockDims(CUDA_BLOCK_DIM_SIZE, CUDA_BLOCK_DIM_SIZE, 1);
	dim3 gridDims(static_cast<int>(ceil(static_cast<float>(W) / static_cast<float>(blockDims.x))), static_cast<int>(ceil(static_cast<float>(H) / static_cast<float>(blockDims.y))));
	filterBilateralY_device << <gridDims, blockDims >> > (input, output, rgb, spatialKernel, intensityKernel,
		spatialSD, intensitySD, dim, W, H);
}

__global__
void filterGaussianX_device(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}

	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (int)sd;
	float normaliser = 0.0;
	int idx;
	for (int r = -rad; r <= rad; r++) {
		idx = x + r;
		if (idx < 0 || idx >= W) {
			continue;
		}
		normaliser += kernel[rad - r];
		for (int i = 0; i < dim; i++) {
			channelSums[i] += input[(y*W + idx)*dim + i] * kernel[rad - r];
		}
	}

	if (normaliser > 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}

__global__
void filterGaussianY_device(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}

	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (int)sd;
	float normaliser = 0.0;
	int idx;
	for (int r = -rad; r <= rad; r++) {
		idx = y + r;
		if (idx < 0 || idx >= H) {
			continue;
		}
		normaliser += kernel[rad - r];
		for (int i = 0; i < dim; i++) {
			channelSums[i] += input[(idx*W + x)*dim + i] * kernel[rad - r];
		}
	}

	if (normaliser > 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}

__global__
void filterBilateralX_device(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}

	//Do the filtering(strictly speaking it's not convolution).
	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (spatialSD > intensitySD) ? (int)spatialSD : (int)intensitySD;
	float normaliser = 0.0;
	for (int r = -rad; r <= rad; r++) {
		int idx = x + r;
		if (idx < 0 || idx >= W) {
			continue;
		}

		int pixelIdx = 3 * (y*W + x);
		int neighPixelIdx = 3 * (y*W + idx);
		float spatialFactor = spatialKernel[(int)fabs((float)r)];
		float intensityFactor = intensityKernel[(int)fabs((float)rgb[neighPixelIdx] - (float)rgb[pixelIdx])] * //R
			intensityKernel[(int)fabs((float)rgb[neighPixelIdx + 1] - (float)rgb[pixelIdx + 1])] * //G
			intensityKernel[(int)fabs((float)rgb[neighPixelIdx + 2] - (float)rgb[pixelIdx + 2])];//B


		normaliser += spatialFactor*intensityFactor;
		for (int i = 0; i < dim; i++) {
			channelSums[i] += input[(y*W + idx)*dim + i] * spatialFactor*intensityFactor;
		}
	}

	if (normaliser > 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}

__global__
void filterBilateralY_device(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}

	//Do the filtering(strictly speaking it's not convolution).
	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (spatialSD > intensitySD) ? (int)spatialSD : (int)intensitySD;
	float normaliser = 0.0;
	for (int r = -rad; r <= rad; r++) {
		int idx = y + r;
		if (idx < 0 || idx >= H) {
			continue;
		}

		int pixelIdx = 3 * (y*W + x);
		int neighPixelIdx = 3 * (idx*W + x);
		float spatialFactor = spatialKernel[(int)fabs((float)r)];
		float intensityFactor = intensityKernel[(int)fabs((float)rgb[neighPixelIdx] - (float)rgb[pixelIdx])] * //R
			intensityKernel[(int)fabs((float)rgb[neighPixelIdx + 1] - (float)rgb[pixelIdx + 1])] * //G
			intensityKernel[(int)fabs((float)rgb[neighPixelIdx + 2] - (float)rgb[pixelIdx + 2])];//B


		normaliser += spatialFactor*intensityFactor;
		for (int i = 0; i < dim; i++) {
			channelSums[i] += input[(idx*W + x)*dim + i] * spatialFactor*intensityFactor;
		}
	}

	if (normaliser > 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}