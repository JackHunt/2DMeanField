#ifndef MEANFIELD_SHARED_BILATERAL_FILTER_HEADER
#define MEANFIELD_SHARED_BILATERAL_FILTER_HEADER

#include <cmath>

namespace MeanField{
	namespace Filtering{
		inline void applyBilateralKernel(const float *spatial_kernel, const float *intensity_kernel,
										 const float *input, const unsigned char *rgb, float *output,
										 float sd_spatial, float sd_intensity, int dim, int x, int y, int W, int H){
			float normaliser = 0.0;
			float spatialFactor, intensityFactor;
			float channelSum[dim];
			for(int i=0; i<dim; i++){
				channelSum[i] = 0.0;
			}

			int sd_int = (sd_spatial > sd_intensity) ? (int)sd_spatial : (int)sd_intensity;
			int idx_x, idx_y, idx_c, idx_n;
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

					idx_c = 3*(y*W + x);//Current pixel idx.
					idx_n = 3*(idx_y*W + idx_x);//Neighbour idx.
					
					spatialFactor = spatial_kernel[(int)fabs(i)]*spatial_kernel[(int)fabs(j)];
					intensityFactor = intensity_kernel[(int)fabs(rgb[idx_n] - rgb[idx_c])]*//R
						intensity_kernel[(int)fabs(rgb[idx_n + 1] - rgb[idx_c + 1])]*//G
						intensity_kernel[(int)fabs(rgb[idx_n + 2] - rgb[idx_c + 2])];//B
					
					normaliser += spatialFactor*intensityFactor;

					//Update cumulative output for each dimension/channel.
					for(int k=0; k<dim; k++){
						channelSum[k] += input[(idx_y*W + idx_x)*dim + k]*spatialFactor*intensityFactor;
					}
				}
			}

			//Normalise outputs.
			if(normaliser > 0.0){
				for(int i=0; i<dim; i++){
					output[(y*W + x)*dim + i] = channelSum[i]/normaliser;
				}
			}
		}
	}
}

#endif
