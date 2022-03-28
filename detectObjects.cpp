#include <vector>
#include "detectObjects.h"

void detectObjects(cv::Mat& imgorig, cv::Mat& imgthres, cv::Mat& imgdetect){
	/*
	std::vector<std::vector<cv::Point>> contoursimg; // detected contours
	std::vector<std::vector<cv::Point>> hull;
	std::vector<std::vector<cv::Point>> countoursres; // 
	std::vector<cv::Vec4i> hierarchy;
	*/

	std::vector<std::vector<cv::Point>> contoursimg ;
	std::vector<cv::Vec4i> hierarchy ;
	std::vector<std::vector<cv::Point>> countoursoutp ;

	// create binary from grayscale
	cv::threshold(imgthres, imgthres,128,255, cv::THRESH_BINARY);

	cv::findContours(
		imgthres, 
		contoursimg, 
		hierarchy, 
		cv::RETR_CCOMP, 
		cv::CHAIN_APPROX_SIMPLE);
/*	hull.resize(contours.size());

	std::vector<cv::Point> approxhull;
	// approximates each contour to polygon
	for (std::vector<cv::Point>cont : contours) {

		// create contours
		cv::convexHull(cv::Mat(cont), approxhull);
		cv::approxPolyDP(approxhull, approxhull, cv::arcLength(approxhull, true) * 0.02, true);

		// keep only circles
		//if (approxhull.size() > 10
		//	&& fabs(cv::contourArea(approxhull)) > 10
		//	) {
			circles.push_back(approxhull);
		//}
		
	}
	*/

	/*
	for (std::vector<cv::Point>cont : contoursimg) {
		// detect polygons
		countoursres = ;

	}
	*/

	countoursoutp.resize(contoursimg.size()) ;
	for( size_t k = 0; k < contoursimg.size(); k++ ) {
        cv::approxPolyDP(
			cv::Mat(contoursimg[k]), 
			countoursoutp[k], 
			10, 
			true);
	}

    // draw contours
	imgdetect = cv::Mat::zeros(imgthres.size(), CV_8UC1);
	cv::drawContours(
		imgdetect, 
		countoursoutp, 
		-1, 
		cv::Scalar(255, 0, 0), 
		2, 
		cv::LINE_8, 
		hierarchy, 
		0);

}