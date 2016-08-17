#ifndef CODE_SHARING_H_
#define CODE_SHARING_H_

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define __SHARED_CODE__ __device__
#else
#define __SHARED_CODE__
#endif

#endif
