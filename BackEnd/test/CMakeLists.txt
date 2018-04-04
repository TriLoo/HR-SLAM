#set(SOURCE_FILES main.cpp ${PROJECT_SOURCE_DIR}/src/PoseGraph.cpp)
set(SOURCE_FILES main.cpp)
add_executable(backend ${SOURCE_FILES})

target_link_libraries(backend ${PROJECT_NAME})
