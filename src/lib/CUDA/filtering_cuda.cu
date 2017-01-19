#include "filtering_cuda.h"

using namespace MeanField::CUDA::Filtering;

void GaussianFilterSeparable::applyXDirection(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H) {
	dim3 blockDims(16, 16, 1);
	dim3 gridDims(static_cast<int>(ceil(W / blockDims.x)), static_cast<int>(ceil(H / blockDims.y)));
	size_t sharedBufferSize = ((blockDims.y*(blockDims.x + 4*static_cast<int>(sd)))*dim);//TO-DO: remove hard coding channels.
	filterGaussianX_device << <gridDims, blockDims, sharedBufferSize * sizeof(float) >> > (input, output, kernel, sd, dim, W, H, sharedBufferSize);
}

void GaussianFilterSeparable::applyYDirection(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H) {
	dim3 blockDims(16, 16, 1);
	dim3 gridDims(static_cast<int>(ceil(W / blockDims.x)), static_cast<int>(ceil(H / blockDims.y)));
	size_t sharedBufferSize = ((blockDims.y + 4 * static_cast<int>(sd))*blockDims.x)*dim;//TO-DO: remove hard coding channels.
	filterGaussianY_device << <gridDims, blockDims, sharedBufferSize * sizeof(float) >> > (input, output, kernel, sd, dim, W, H, sharedBufferSize);
}

void BilateralFilterSeparable::applyXDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H) {
	dim3 blockDims(16, 16, 1);
	dim3 gridDims(static_cast<int>(ceil(W / blockDims.x)), static_cast<int>(ceil(H / blockDims.y)));
	int maxSD = (spatialSD > intensitySD) ? spatialSD : intensitySD;
	size_t sharedBufferSize = ((blockDims.y*(blockDims.x + 4 * static_cast<int>(maxSD)))*dim*3);//TO-DO: remove hard coding channels.
	filterBilateralX_device << <gridDims, blockDims, sharedBufferSize * sizeof(float) >> > (input, output, rgb, spatialKernel, intensityKernel, 
		spatialSD, intensitySD, dim, W, H, sharedBufferSize);
}

void BilateralFilterSeparable::applyYDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H) {
	dim3 blockDims(16, 16, 1);
	dim3 gridDims(static_cast<int>(ceil(W / blockDims.x)), static_cast<int>(ceil(H / blockDims.y)));
	int maxSD = (spatialSD > intensitySD) ? spatialSD : intensitySD;
	size_t sharedBufferSize = ((blockDims.y + 4 * static_cast<int>(maxSD))*blockDims.x)*dim*3;//TO-DO: remove hard coding channels.
	filterBilateralY_device << <gridDims, blockDims, sharedBufferSize * sizeof(float) >> > (input, output, rgb, spatialKernel, intensityKernel,
		spatialSD, intensitySD, dim, W, H, sharedBufferSize);
}

__global__
void filterGaussianX_device(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H, size_t sharedSize) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}
	extern __shared__ float distData[];
	__syncthreads();

	//Load data into shared memory.
	int twoSD = 2 * (int)sd;
	int idxLeftGlobal = x - twoSD;
	int idxRightGlobal = x + twoSD;
	int localIdxLeft = threadIdx.y*(blockDim.x + 2 * twoSD) + (threadIdx.x - twoSD);
	int localIdxRight = threadIdx.y*(blockDim.x + 2 * twoSD) + (threadIdx.x + twoSD);
	for (int i = 0; i < dim; i++) {
		int sharedIdx = localIdxLeft*dim + i;
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			distData[sharedIdx] = (idxLeftGlobal > 0) ? input[(y*W + idxLeftGlobal)*dim + i] : 0.0;
		}

		sharedIdx = localIdxRight*dim + i;
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			distData[sharedIdx] = (idxRightGlobal < W) ? input[(y*W + idxRightGlobal)*dim + i] : 0.0;
		}
	}
	__syncthreads();

	//Do the convolution.
	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (int)sd;
	float normaliser = 0.0;
	int sharedIdx = 0;
	for (int r = -rad; r <= rad; r++) {
		normaliser += kernel[rad - r];
		for (int i = 0; i < dim; i++) {
			sharedIdx = (threadIdx.y*(blockDim.x + 2 * twoSD) + threadIdx.x + r)*dim + i;
			if (sharedIdx >= 0 && sharedIdx < sharedSize) {
				channelSums[i] += distData[sharedIdx] * kernel[rad - r];
			}
		}
	}

	if (normaliser > 0.0 || normaliser < 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}

__global__
void filterGaussianY_device(const float *input, float *output, const float *kernel, float sd, int dim, int W, int H, size_t sharedSize) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	} 
	extern __shared__ float distData[];
	__syncthreads();

	//Load data into shared memory.
	int twoSD = 2 * (int)sd;
	int idxUpGlobal = y - twoSD;
	int idxDownGlobal = y + twoSD;
	int localIdxUp = (threadIdx.y - twoSD)*blockDim.x + threadIdx.x;
	int localIdxDown = (threadIdx.y + twoSD)*blockDim.x + threadIdx.x;
	for (int i = 0; i < dim; i++) {
		int sharedIdx = localIdxUp*dim + i;
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			distData[sharedIdx] = (idxUpGlobal > 0) ? input[(idxUpGlobal*W + x)*dim + i] : 0.0;
		}
		
		sharedIdx = localIdxDown*dim + i;
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			distData[sharedIdx] = (idxDownGlobal < H) ? input[(idxDownGlobal*W + x)*dim + i] : 0.0;
		}
	}
	__syncthreads();

	//Do the convolution.
	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (int)sd;
	float normaliser = 0.0;
	int sharedIdx = 0;
	for (int r = -rad; r <= rad; r++) {
		normaliser += kernel[rad - r];
		for (int i = 0; i < dim; i++) {
			sharedIdx = ((threadIdx.y + r)*blockDim.x + threadIdx.x)*dim + i;

			if (sharedIdx >= 0 && sharedIdx < sharedSize) {
				channelSums[i] += distData[sharedIdx] * kernel[rad - r];
			}
		}
	}

	if (normaliser > 0.0 || normaliser < 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}

__global__
void filterBilateralX_device(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H, size_t sharedSize) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}
	extern __shared__ float data[];
	__syncthreads();

	//Load data into shared memory.
	int twoSD = 2 * (spatialSD > intensitySD) ? (int)spatialSD : (int)intensitySD;
	int idxLeftGlobal = x - twoSD;
	int idxRightGlobal = x + twoSD;
	int localIdxLeft = threadIdx.y*(blockDim.x + 2 * twoSD) + (threadIdx.x - twoSD);
	int localIdxRight = threadIdx.y*(blockDim.x + 2 * twoSD) + (threadIdx.x + twoSD);
	int sharedIdx = 0;
	//Load distribution data.
	for (int i = 0; i < dim; i++) {
		sharedIdx = localIdxLeft*dim*3 + i;//TO DO: remove hard coding channels.
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (idxLeftGlobal > 0) ? input[(y*W + idxLeftGlobal)*dim + i] : 0.0;
		}

		sharedIdx = localIdxRight*dim + i;
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (idxRightGlobal < W) ? input[(y*W + idxRightGlobal)*dim + i] : 0.0;
		}
	}

	//Load RGB image data.
	for (int i = 0; i < 3; i++) {
		sharedIdx = localIdxLeft*dim * 3 + dim + i;//TO DO: remove hard coding channels.
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (float)rgb[(y*W + idxLeftGlobal) * 3 + i];
		}

		sharedIdx = localIdxRight*dim * 3 + dim + i;//TO DO: remove hard coding channels.
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (float)rgb[(y*W + idxRightGlobal) * 3 + i];
		}
	}
	__syncthreads();

	//Do the filtering(strictly speaking it's not convolution).
	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (spatialSD > intensitySD) ? (int)spatialSD : (int) intensitySD;
	float normaliser = 0.0;
	for (int r = -rad; r <= rad; r++) {
		int pixelIdx = 3 * dim * (y*W + x) + dim;
		int neighPixelIdx = 3 * dim * (y*W + x + r) + dim;
		float spatialFactor = spatialKernel[(int)fabs((float)r)];
		float intensityFactor = intensityKernel[(int)fabs(data[neighPixelIdx] - data[pixelIdx])] *//R
			intensityKernel[(int)fabs(data[neighPixelIdx + 1] - data[pixelIdx + 1])] *//G
			intensityKernel[(int)fabs(data[neighPixelIdx + 2] - data[pixelIdx + 2])];//B


		normaliser += spatialFactor*intensityFactor;


		for (int i = 0; i < dim; i++) {
			sharedIdx = (threadIdx.y*(blockDim.x + 2 * twoSD) + threadIdx.x + r)*dim*3 + i;//TO DO: remove hard coding channels.
			if (sharedIdx >= 0 && sharedIdx < sharedSize) {
				channelSums[i] += data[sharedIdx] * spatialFactor*intensityFactor;
			}
		}
	}

	if (normaliser > 0.0 || normaliser < 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}

__global__
void filterBilateralY_device(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
	const float *intensityKernel, float spatialSD, float intensitySD, int dim, int W, int H, size_t sharedSize) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) {
		return;
	}
	extern __shared__ float data[];
	__syncthreads();

	//Load data into shared memory.
	int twoSD = 2 * (spatialSD > intensitySD) ? (int)spatialSD : (int)intensitySD;
	int idxUpGlobal = y - twoSD;
	int idxDownGlobal = y + twoSD;
	int localIdxUp = (threadIdx.y - twoSD)*blockDim.x + threadIdx.x;
	int localIdxDown = (threadIdx.y + twoSD)*blockDim.x + threadIdx.x;
	
	//Load distribution data.
	int sharedIdx = 0;
	for (int i = 0; i < dim; i++) {
		sharedIdx = localIdxUp*dim * 3 + i;//TO DO: remove hard coding channels.
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (idxUpGlobal > 0) ? input[(idxUpGlobal*W + x)*dim + i] : 0.0;
		}

		sharedIdx = localIdxDown*dim + i;
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (idxDownGlobal < W) ? input[(idxDownGlobal*W + x)*dim + i] : 0.0;
		}
	}

	//Load RGB image data.
	for (int i = 0; i < 3; i++) {
		sharedIdx = localIdxUp*dim * 3 + dim + i;//TO DO: remove hard coding channels.
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (float)rgb[(localIdxUp*W + x) * 3 + i];
		}

		sharedIdx = localIdxDown*dim * 3 + dim + i;//TO DO: remove hard coding channels.
		if (sharedIdx >= 0 && sharedIdx < sharedSize) {
			data[sharedIdx] = (float)rgb[(localIdxDown*W + x) * 3 + i];
		}
	}
	__syncthreads();

	//Do the filtering(strictly speaking it's not convolution).
	float *channelSums = new float[dim];
	for (int i = 0; i < dim; i++) {
		channelSums[i] = 0.0;
	}

	int rad = (spatialSD > intensitySD) ? (int)spatialSD : (int)intensitySD;
	float normaliser = 0.0;
	for (int r = -rad; r <= rad; r++) {
		int pixelIdx = 3 * dim * (y*W + x) + dim;
		int neighPixelIdx = 3 * dim * ((y+r)*W + x) + dim;
		float spatialFactor = spatialKernel[(int)fabs((float)r)];
		float intensityFactor = intensityKernel[(int)fabs(data[neighPixelIdx] - data[pixelIdx])] *//R
			intensityKernel[(int)fabs(data[neighPixelIdx + 1] - data[pixelIdx + 1])] *//G
			intensityKernel[(int)fabs(data[neighPixelIdx + 2] - data[pixelIdx + 2])];//B


		normaliser += spatialFactor*intensityFactor;


		for (int i = 0; i < dim; i++) {
			sharedIdx = ((threadIdx.y + r)*blockDim.x + threadIdx.x)*dim * 3 + i;//TO DO: remove hard coding channels.
			if (sharedIdx >= 0 && sharedIdx < sharedSize) {
				channelSums[i] += data[sharedIdx] * spatialFactor*intensityFactor;
			}
		}
	}

	if (normaliser > 0.0 || normaliser < 0.0) {
		for (int i = 0; i < dim; i++) {
			output[(y*W + x)*dim + i] = channelSums[i] / normaliser;
		}
	}
	delete[] channelSums;
}