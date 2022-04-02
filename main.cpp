#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>

#include "colorHSVtrackbar.h"
#include "thresholding.h"
#include "detectObjects.h"

//using namespace cv;

// set up global vars ------------------------------------------------------------------
// --- trackbar
const int w = 500;
int levels = 4;
int adaptThresBlock = 5 ;
int adaptThresConst = 12 ;

int iLowH = 22;
int iHighH = 38;

int iLowS = 80;
int iHighS = 150;

int iLowV = 60;
int iHighV = 255;

// --- object detection
std::vector<cv::Vec4i> hierarchy ;
std::vector<std::vector<cv::Point>> contoursOut ;


static void on_trackbar(int, cv::Mat& cnt_img, void*)
{
    int _levels = levels - 3;

	adaptThresBlock = adaptThresBlock<3 ? 3 : adaptThresBlock ;

	if( adaptThresBlock>3 && adaptThresBlock%2!=1 ){
		adaptThresBlock-- ;
	}

	//int _adaptThresConst = adaptThresConst ;

    drawContours( 
		cnt_img, 
		contoursOut, 
		_levels <= 0 ? 3 : -1, 
		cv::Scalar(128,255,255),
        1, cv::LINE_AA, hierarchy, std::abs(_levels) );
}


int main(int argc, char** argv )
{

	// image processing variables
	//cv::Mat imgHSV  ;      // HSV convert
	//cv::Mat imgLines;      // empty image + tracking lines from colored object
	cv::Mat imgGray   ;    // grayscale image
	cv::VideoCapture cap(0);

	if ( cap.isOpened() == false )
	{
		std::cout << "Cannot open the video cam " << std::endl;
		return -1;
	}

    if ( cap.isOpened() == true )
	{
		std::cout << "Video is open " << std::endl;
		std::cout << "Press ESC to end... " << std::endl;
	}

	// setup trackbar - used for manual calibration ----------------------------------------
	// Create trackbars in "Control" window
	cv::namedWindow( "contours", cv::WINDOW_AUTOSIZE );
	cv::createTrackbar("Levels", "contours", &levels, 7); // levels
	cv::createTrackbar("Threshold_Block", "contours", &adaptThresBlock, 11); // adaptive threshold blocksize
	cv::createTrackbar("Threshold_Const", "contours", &adaptThresConst, 12); // adaptive threshold constant

	/*
	cv::createTrackbar("LowH", "contours", &iLowH, 179) ; // Hue (0 - 179)
	cv::createTrackbar("HighH", "contours", &iHighH, 179);

	cv::createTrackbar("LowS", "contours", &iLowS, 255); // Saturation (0 - 255)
	cv::createTrackbar("HighS", "contours", &iHighS, 255);

	cv::createTrackbar("LowV", "contours", &iLowV, 255); // Value (0 - 255)
	cv::createTrackbar("HighV", "contours", &iHighV, 255);
	*/

	// start frame -------------------------------------------------------------------------
	while ( true ) {

		// get video
		cv::Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video 

		//Breaking the while loop at the end of the video
		if ( bSuccess == false )
		{
			std::cout << "Video camera is disconnected" << std::endl;
			break;
		}

		// create HSV image
		//cv::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		// create grayscale image
		cv::cvtColor(imgOriginal, imgGray, cv::COLOR_BGR2GRAY);

		//cv::Mat imgGray_histeq ;
		equalizeHist(imgGray, imgGray);

		// create image with thresholding method v1
		//cv::Mat imgThres = thresholdingv1(imgHSV, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);
		//cv::Mat imgThresv3 = thresholdingv3(imgHSV, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);

		// show video with tracking line
		//cv::imshow("Original", imgOriginal); //show the original image

		// Object detection - Contours
		cv::Mat imgDetect = cv::Mat::zeros(w, w, CV_8UC3);
		on_trackbar(0,imgDetect,0);
		detectObjects(
			imgGray, adaptThresBlock, adaptThresConst, hierarchy, contoursOut);

		// Object detections - HOG
		cv::Mat imgDetect2(imgOriginal) ;
		cv::HOGDescriptor hog;
		hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		std::vector<cv::Rect> detections ;
		hog.detectMultiScale(imgDetect2, detections, 0, cv::Size(8,8), cv::Size(32,32), 1.2, 2 );

		for (auto& detection : detections ){
			cv::rectangle(imgDetect2, detection.tl(), detection.br(), cv::Scalar(255, 0, 0), 2 );
		}
		
		// show thresholded image
		//cv::imshow("Detected Image", imgDetect); //show the thresholded image

		// show detected2 image
		cv::imshow("Detected Image", imgDetect2); //show the thresholded image

		// show grayscale image
		//cv::imshow("Grayscale Image", imgGray); //show the thresholded image

		// exit -------------------------------------------------------------------------------
		int comm = cv::waitKey(10);

		// exit -------------------------------------------------------------------------------
		if ( comm == 27 ) {
			std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
			break;
		}
	}

	cv::destroyAllWindows(); //Destroy all opened windows

    return 0;
}