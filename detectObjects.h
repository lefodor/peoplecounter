#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void detectObjects(cv::Mat&, int&, int& , std::vector<cv::Vec4i>&, std::vector<std::vector<cv::Point>>& ) ;