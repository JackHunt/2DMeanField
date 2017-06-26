FIND_PATH(MeanField_INCLUDE_DIR 2DMeanField.h)
INCLUDE_DIRECTORIES(${MeanField_INCLUDE_DIR})

IF(MSVC_IDE)
  FIND_LIBRARY(MeanField_LIBRARY_DEBUG 2DMeanField)
  FIND_LIBRARY(MeanField_LIBRARY_RELEASE 2DMeanField)
  SET(MeanField_LIBRARY debug ${MeanField_LIBRARY_DEBUG} optimized ${MeanField_LIBRARY_RELEASE})
ELSE()
  FIND_LIBRARY(MeanField_LIBRARY 2DMeanField)
ENDIF()