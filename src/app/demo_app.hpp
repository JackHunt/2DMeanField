#ifndef MEANFIELD_DEMO_APP_HEADER
#define MEANFIELD_DEMO_APP_HEADER

#include <iostream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../lib/2DMeanField.hpp"
#ifdef WITH_CUDA
#include "../lib/CUDA/cuda_util.hpp"
#endif

namespace MeanFieldDemo {
    int noColours = 0;
    int colourMap[255] = { 0 };
    const int M = 3;
}

#endif
