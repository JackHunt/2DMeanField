#ifndef MEANFIELD_SHARED_GAUSSIAN_FILTER_HEADER
#define MEANFIELD_SHARED_GAUSSIAN_FILTER_HEADER

#include "code_sharing.h"
#include <math.h>

namespace MeanField{
	namespace Filtering{
		inline void generateGaussianKernel(float *out, int dim, float sd){
			int rad = dim/2;
			float x;
			for(int i=0; i<dim; i++){
				x = i-rad;
				out[i] = expf(-(i*i) / (2.0*sd*sd));
			}
		}

		__SHARED_CODE__
		inline void applyGaussianKernel(const float *kernel, const float *input, float *output,
										float sd, int dim, int x, int y, int W, int H){
			float normaliser = 0.0;
			float factor;
			float *channelSum = new float[dim];
			for(int i=0; i<dim; i++){
				channelSum[i] = 0.0;
			}

			int sd_int = (int)sd;
			int idx_x, idx_y;
			//Convolve for each channel.
			for(int i=-sd_int; i<sd_int; i++){
				idx_y = y+i;
				if(idx_y < 0 || idx_y >= H){
					continue;
				}
				
				for(int j=-sd_int; j<sd_int; j++){
					idx_x = x+j;
					if(idx_x < 0 || idx_x >= W){
						continue;
					}
					factor = kernel[(int)fabs((float)i)]*kernel[(int)fabs((float)j)];
					normaliser += factor;

					//Update cumulative output for each dimension/channel.
					for(int k=0; k<dim; k++){
						channelSum[k] += input[(idx_y*W + idx_x)*dim + k]*factor;
					}
				}
			}

			//Normalise outputs.
			if(normaliser > 0.0){
				for(int i=0; i<dim; i++){
					output[(y*W + x)*dim + i] = channelSum[i]/normaliser;
				}
			}
			delete[] channelSum;
		}
	}
}

#endif
