#include <vector>
#include "detectObjects.h"

void detectObjects(cv::Mat& imgorig, cv::Mat& imgthres, std::vector<cv::Vec4i>& hierarchy, std::vector<std::vector<cv::Point>>& contoursOut){

	std::vector<std::vector<cv::Point>> contoursImg ;

	// create binary from grayscale
	cv::threshold(imgthres, imgthres,128,255, cv::THRESH_BINARY);

	cv::findContours(
		imgthres, 
		contoursImg, 
		hierarchy, 
		//cv::RETR_CCOMP, 
		cv::RETR_TREE,
		cv::CHAIN_APPROX_SIMPLE );

	contoursOut.resize(contoursImg.size()) ;
	for( size_t k = 0; k < contoursImg.size(); k++ ) {
        cv::approxPolyDP(
			cv::Mat(contoursImg[k]), 
			contoursOut[k], 
			5, 
			true);
	}
}