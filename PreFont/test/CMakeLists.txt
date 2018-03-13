set(SOURCE_FILES main.cpp)
add_executable(mySLAM ${SOURCE_FILES})

set(VO_FILES VO.cpp)
#message(STATUS ${VO_FILES})
add_executable(VO ${VO_FILES})

# add target link libraries
#target_include_directories(mySLAM ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mySLAM ${PROJECT_NAME})
target_link_libraries(VO ${PROJECT_NAME})
