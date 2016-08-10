#ifndef MEANFIELD_SHARED_BILATERAL_FILTER_HEADER
#define MEANFIELD_SHARED_BILATERAL_FILTER_HEADER

#include <cmath>

namespace MeanField{
	namespace Filtering{
		inline void applyBilateralKernel(const float *spatial_kernel, const float *intensity_kernel,
										 const float *input, const unsigned char *rgb, const float *output,
										 float sd_spatial, float sd_intensity, int dim, int x, int y, int W, int H){
			float normaliser = 0.0;
			float spatialFactor, intensityFactor;
			float channelSum[dim]();

			int sd_int = (sd_spatial > sd_intensity) ? (int)sd_spatial : (int)sd_intensity;
			int idx_x, idx_y, idx_c, idx_n;
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

					idx_c = 3*(y*W + x);//Current pixel idx.
					idx_n = 3*(idx_y*W + idx_x);//Neighbour idx.
					
					spatialFactor = spatial_kernel[i+sd_int]*spatial_kernel[j+sd_int];
					intensityFactor = intensity_kernel[rgb[idx_n] - rgb[idx_c]]*//R
						intensity_kernel[rgb[idx_n + 1] - rgb[idx_c + 1]]*//G
						intensity_kernel[rgb[idx_n + 2] - rgb[idx_c + 2]]//B
					
					normaliser += spatialFactor*intensityFactor;

					//Update cumulative output for each dimension/channel.
					for(int i=0; i<dim; i++){
						channelSum[i] += input[(idx_y*W + idx_x)*dim + i]*spatialFactor*intensityFactor;
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
