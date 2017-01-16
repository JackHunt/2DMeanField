#ifndef MEANFIELD_FILTERING_CPU_HEADER
#define MEANFIELD_FILTERING_CPU_HEADER

#include "math.h"

namespace MeanField{
namespace CPU{
namespace Filtering{
struct GaussianFilterSeparable{
    static void applyXDirection(const float *input, float *output, const float *kernel, float sd,
                                int dim, int x, int y, int W, int H);
    static void applyYDirection(const float *input, float *output, const float *kernel, float sd,
                                int dim, int x, int y, int W, int H);
};

struct BilateralFilterSeparable{
    static void applyXDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
                                const float *intensityKernel, float spatialSD, float intensitySD, int dim,
                                int x, int y, int W, int H);
    static void applyYDirection(const float *input, float *output, const unsigned char *rgb, const float *spatialKernel,
                                const float *intensityKernel, float spatialSD, float intensitySD, int dim,
                                int x, int y, int W, int H);
};
}
}
}

#endif
