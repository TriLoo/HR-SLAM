set(SOURCE_FILES main.cpp)
add_executable(mySLAM ${SOURCE_FILES})

# add target link libraries
#target_include_directories(mySLAM ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mySLAM ${PROJECT_NAME})
