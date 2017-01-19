set(SHARED_HEADERS
  shared/meanfield.h
  shared/bilateral_filter.h
  shared/gaussian_filter.h
  shared/code_sharing.h
)

set(CPU_HEADERS
  CPU/meanfield_cpu.h
  CPU/filtering_cpu.h
)

set(CPU_SOURCES
  CPU/meanfield_cpu.cpp
  CPU/filtering_cpu.cpp
)

set(CUDA_HEADERS
  CUDA/meanfield_cuda.h
  CUDA/filtering_cuda.h
)

set(CUDA_SOURCES
  CUDA/meanfield_cuda.cu
  CUDA/filtering_cuda.cu
)

source_group(CPU FILES ${CPU_HEADERS} ${CPU_SOURCES})
source_group(CUDA FILES ${CUDA_HEADERS} ${CUDA_SOURCES})
source_group(shared FILES ${SHARED_HEADERS})

if(WITH_CUDA)
  cuda_add_library(2DMeanField ${SHARED_HEADERS} ${CUDA_HEADERS} ${CUDA_SOURCES} ${CPU_HEADERS} ${CPU_SOURCES})
else()
  add_library(2DMeanField ${SHARED_HEADERS} ${CPU_HEADERS} ${CPU_SOURCES})
endif()
