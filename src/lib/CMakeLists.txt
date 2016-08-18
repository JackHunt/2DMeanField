set(SHARED_HEADERS
  shared/meanfield.h
  shared/bilateral_filter.h
  shared/gaussian_filter.h
  shared/code_sharing.h
)

set(CPU_HEADERS
  CPU/meanfield_cpu.h
)

set(CPU_SOURCES
  CPU/meanfield_cpu.cpp
)

set(CUDA_HEADERS
  CUDA/meanfield_cuda.h
)

set(CUDA_SOURCES
  CUDA/meanfield_cuda.cu
)

#add_library(2DMeanField SHARED ${SHARED_HEADERS} ${CPU_HEADERS} ${CPU_SOURCES})
cuda_add_library(2DMeanFieldLib SHARED ${SHARED_HEADERS} ${CUDA_HEADERS} ${CUDA_SOURCES} ${CPU_HEADERS} ${CPU_SOURCES})

#add_library(${PROJECT_NAME} SHARED $<TARGET_OBJECTS:2DMeanFieldLib>)
