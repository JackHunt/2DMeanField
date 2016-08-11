set(APP_HEADERS
  demo_app.h
)

set(APP_SOURCES
  demo_app.cpp
)

add_executable(MeanFieldDemo ${APP_HEADERS} ${APP_SOURCES})
target_link_libraries(MeanFieldDemo ${PROJECT_NAME} ${OpenCV_LIBS})
