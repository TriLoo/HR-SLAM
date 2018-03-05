# My Front-end SLAM Project

## Description
* This directories include all the source files needed to complete 
  a simplified front-end SLAM, referring <视觉SLAM十四讲> & ORB-SLAM2
  
## Licence
GPL-v2
  
## Files
* bin : the executable files
* cmake_modules : the find_module.cmake files
* config : necessary configure files
* include : all header files needed by files in *src* folder
* lib : the generated library files
* src : the source files to generate shared libraries
* test : the source files for test, and generate the executable files

## Cmake Syntax
* cmake_minimum_required(VERSION 0.0)
* Project Name setting
* Compiler flag setting
* include cmake modules
* include the third part packages and corresponding header files
* other configures to cmake, such as output path *etc*
* add subdirectories, including *src*, *test*, and *others*

## References
* <视觉SLAM十四讲>
* [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
* Multiple View Geometry in Computer Vision
* Deep Learning
* [Google C++ Style](https://google.github.io/styleguide/cppguide.html)
