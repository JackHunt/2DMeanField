#ifndef MEANFIELD_SHARED_GAUSSIAN_FILTER_HEADER
#define MEANFIELD_SHARED_GAUSSIAN_FILTER_HEADER

#include <cmath>

namespace MeanField{
	namespace Filtering{
		inline void generateGaussianKernel(float *out, int dim, float sd){
			int rad = dim/2;
			float x;
			for(int i=0; i<dim; i++){
				x = i-rad;
				out[i] = expf(-(x*x) / 2*M_PI*sd*sd);
			}
		}

		inline void applyGaussianKernel(const float *kernel, const float *input, float *output,
										float sd, int dim, int x, int y, int W, int H){
			float normaliser = 0.0;
			float factor;
			float channelSum[dim];
			for(int i=0; i<dim; i++){
				channelSum[i] = 0.0;
			}

			int sd_int = (int)sd;
			int idx_x, idx_y;
			//Convolve for each channel.
			for(int i=-sd_int; i<sd_int; i++){
				idx_y = x+i;
				if(idx_y < 0 || idx_y >= H){
					continue;
				}
				
				for(int j=-sd_int; j<sd_int; j++){
					idx_x = x+j;
					if(idx_x < 0 || idx_x >= W){
						continue;
					}
					factor = kernel[i+sd_int]*kernel[j+sd_int];
					normaliser += factor;

					//Update cumulative output for each dimension/channel.
					for(int i=0; i<dim; i++){
						channelSum[i] += input[(idx_y*W + idx_x)*dim + i]*factor;
					}
				}
			}

			//Normalise outputs.
			if(normaliser > 10e-10){
				for(int i=0; i<dim; i++){
					output[(y*W + x)*dim + i] = channelSum[i]/normaliser;
				}
			}
		}
	}
}

#endif
