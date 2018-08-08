set(SHARED_HEADERS
  shared/meanfield.hpp
  shared/bilateral_filter.hpp
  shared/gaussian_filter.hpp
  shared/code_sharing.hpp
)

set(CPU_HEADERS
  CPU/meanfield_cpu.hpp
)

set(CPU_SOURCES
  CPU/meanfield_cpu.cpp
)

set(CUDA_HEADERS
  CUDA/meanfield_cuda.hpp
)

set(CUDA_SOURCES
  CUDA/meanfield_cuda.cu
)

source_group(CPU FILES ${CPU_HEADERS} ${CPU_SOURCES})
source_group(CUDA FILES ${CUDA_HEADERS} ${CUDA_SOURCES})
source_group(shared FILES ${SHARED_HEADERS})

if(WITH_CUDA)
  cuda_add_library(2DMeanField ${SHARED_HEADERS} ${CUDA_HEADERS} ${CUDA_SOURCES} ${CPU_HEADERS} ${CPU_SOURCES})
else()
  add_library(2DMeanField ${SHARED_HEADERS} ${CPU_HEADERS} ${CPU_SOURCES})
endif()
