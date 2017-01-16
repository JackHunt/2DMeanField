#include "filtering_cpu.h"

using namespace MeanField::CPU::Filtering;

void GaussianFilterSeparable::applyXDirection(const float *input, float *output, const float *kernel, float sd,
                                              int dim, int x, int y, int W, int H){
    float channelSums[dim];
    for(int i=0; i<dim; i++){
        channelSums[i] = 0.0;
    }

    int rad = static_cast<int>(sd);
    float normaliser = 0.0;
    for(int r=-rad; r<=rad; r++){
        int idx = x + r;
        if(idx < 0 || idx >= W){
            continue;
        }

        normaliser += kernel[rad - r];
        for(int i=0; i<dim; i++){
            channelSums[i] += input[(y*W + idx)*dim + i]*kernel[rad - r];
        }
    }

    if(normaliser > 0.0){
        for(int i=0; i<dim; i++){
            output[(y*W + x)*dim + i] = channelSums[i]/normaliser;
        }
    }
}

void GaussianFilterSeparable::applyYDirection(const float *input, float *output, const float *kernel, float sd,
                                              int dim, int x, int y, int W, int H){
    float channelSums[dim];
    for(int i=0; i<dim; i++){
        channelSums[i] = 0.0;
    }

    int rad = static_cast<int>(sd);
    float normaliser = 0.0;
    for(int r=-rad; r<=rad; r++){
        int idx = y + r;
        if(idx < 0 || idx >= H){
            continue;
        }

        normaliser += kernel[rad - r];
        for(int i=0; i<dim; i++){
            channelSums[i] += input[(idx*W + x)*dim + i]*kernel[rad - r];
        }
    }

    if(normaliser > 0.0){
        for(int i=0; i<dim; i++){
            output[(y*W + x)*dim + i] = channelSums[i]/normaliser;
        }
    }
}

void BilateralFilterSeparable::applyXDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
                                               const float *intensityKernel, float spatialSD, float intensitySD, int dim,
                                               int x, int y, int W, int H){
    float channelSums[dim];
    for(int i=0; i<dim; i++){
        channelSums[i] = 0.0;
    }

    int rad = (spatialSD > intensitySD) ? static_cast<int>(spatialSD) : static_cast<int>(intensitySD);
    float normaliser = 0.0;
    for(int r=-rad; r<=rad; r++){
        int idx = x + r;
        if(idx < 0 || idx >= W){
            continue;
        }

        int pixelIdx = 3*(y*W + x);
        int neighPixelIdx = 3*(y*W + idx);

        float spatialFactor = spatialKernel[static_cast<int>(fabs(static_cast<float>(r)))];
        float intensityFactor = intensityKernel[static_cast<int>(fabs(static_cast<float>(rgb[neighPixelIdx]) - static_cast<float>(rgb[pixelIdx])))]*//R
                intensityKernel[static_cast<int>(fabs(static_cast<float>(rgb[neighPixelIdx + 1]) - static_cast<float>(rgb[pixelIdx + 1])))]*//G
                intensityKernel[static_cast<int>(fabs(static_cast<float>(rgb[neighPixelIdx + 2]) - static_cast<float>(rgb[pixelIdx + 2])))];//B

        normaliser += spatialFactor*intensityFactor;
        for(int i=0; i<dim; i++){
            channelSums[i] += input[(y*W + idx)*dim + i]*spatialFactor*intensityFactor;
        }
    }

    if(normaliser > 0.0){
        for(int i=0; i<dim; i++){
            output[(y*W + x)*dim + i] = channelSums[i]/normaliser;
        }
    }
}

void BilateralFilterSeparable::applyYDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
                                               const float *intensityKernel, float spatialSD, float intensitySD, int dim,
                                               int x, int y, int W, int H){
    float channelSums[dim];
    for(int i=0; i<dim; i++){
        channelSums[i] = 0.0;
    }

    int rad = (spatialSD > intensitySD) ? static_cast<int>(spatialSD) : static_cast<int>(intensitySD);
    float normaliser = 0.0;
    for(int r=-rad; r<=rad; r++){
        int idx = y + r;
        if(idx < 0 || idx >= H){
            continue;
        }

        int pixelIdx = 3*(y*W + x);
        int neighPixelIdx = 3*(idx*W + x);

        float spatialFactor = spatialKernel[static_cast<int>(fabs(static_cast<float>(r)))];
        float intensityFactor = intensityKernel[static_cast<int>(fabs(static_cast<float>(rgb[neighPixelIdx]) - static_cast<float>(rgb[pixelIdx])))]*//R
                intensityKernel[static_cast<int>(fabs(static_cast<float>(rgb[neighPixelIdx + 1]) - static_cast<float>(rgb[pixelIdx + 1])))]*//G
                intensityKernel[static_cast<int>(fabs(static_cast<float>(rgb[neighPixelIdx + 2]) - static_cast<float>(rgb[pixelIdx + 2])))];//B

        normaliser += spatialFactor*intensityFactor;
        for(int i=0; i<dim; i++){
            channelSums[i] += input[(idx*W + x)*dim + i]*spatialFactor*intensityFactor;
        }
    }

    if(normaliser > 0.0){
        for(int i=0; i<dim; i++){
            output[(y*W + x)*dim + i] = channelSums[i]/normaliser;
        }
    }
}
