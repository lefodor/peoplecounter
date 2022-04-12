#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>
#include "opencv2/ml.hpp"
#include "opencv2/videoio.hpp"

#include "colorHSVtrackbar.h"
#include "thresholding.h"
#include "detectObjects.h"

//using namespace cv;

// set up global vars ------------------------------------------------------------------
// --- trackbar
const int w = 500;
int levels = 4;
int hystMin = 25 ;
int hystMax = 75 ;
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

	cv::Mat imgFromStream, imgBlur, imgCanny, imgDil, imgGray, imgAdapt;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)) ;
	cv::VideoCapture cap(0);

	// setup trackbar - used for manual calibration ----------------------------------------
	// Create trackbars in "Control" window

	cv::namedWindow( "contours", cv::WINDOW_AUTOSIZE );
	cv::createTrackbar("Levels", "contours", &levels, 7); // levels
	cv::createTrackbar("HystMin", "contours", &hystMin, 50); // hysteresis min
	cv::createTrackbar("HystMax", "contours", &hystMax, 100); // hysteresis min
	cv::createTrackbar("Threshold_Block", "contours", &adaptThresBlock, 11); // adaptive threshold blocksize
	cv::createTrackbar("Threshold_Const", "contours", &adaptThresConst, 12); // adaptive threshold constant

	// load hog
	cv::HOGDescriptor hog;
	cv::String obj_det_filename = "../detectionoutput.yml" ;
    hog.load( obj_det_filename );

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

	// start frame -------------------------------------------------------------------------
	while ( true ) {

		// get video
		bool bSuccess = cap.read(imgFromStream); // read a new frame from video 

		//Breaking the while loop at the end of the video
		if ( bSuccess == false )
		{
			std::cout << "Video camera is disconnected" << std::endl;
			break;
		}

/*
		// crop from image --------------------------------------------------------------
		cv::Rect roi(100, 100, 300, 300);
		cv::Mat imgCrop(roi) ;

		// draw -------------------------------------------------------------------------
		circle()
		rectangle()
		line()
		putText()
*/

		// create grayscale image
		//cv::cvtColor(imgFromStream, imgGray, cv::COLOR_BGR2GRAY);

		//cv::Mat imgGray_histeq ;
		//equalizeHist(imgGray, imgGray);

		// color detection --------------------------------------------------------------
		// create image with thresholding method v1
		//cv::cvtColor(imgFromStream, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
		//cv::Mat imgThres = thresholdingv1(imgHSV, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);
		//cv::Mat imgThresv3 = thresholdingv3(imgHSV, iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);

		// image pre-process ------------------------------------------------------------
		// for edge detection and contours
		cv::cvtColor(imgFromStream, imgGray, cv::COLOR_BGR2GRAY) ;
		cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 3, 0) ;
		cv::Canny(imgBlur, imgCanny, hystMin, hystMax) ;
		cv::dilate(imgCanny, imgDil, kernel) ;

		// create grayscale image
		//cv::adaptiveThreshold(imgGray, imgAdapt, 128, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, adaptThresBlock, adaptThresConst) ;

		// object detection -------------------------------------------------------------
		// Object detection - Contours
		cv::Mat imgDetect = cv::Mat::zeros(w, w, CV_8UC3);
		on_trackbar(0, imgDetect, 0);
		detectObjects(
			imgDil, hierarchy, contoursOut);

		// show detected2 image
		cv::imshow("Detected Image", imgDetect); //show the thresholded image

/*
		// Object detections - HOG
		std::vector<cv::Rect> detections ;
		hog.detectMultiScale(imgFromStream, detections, 0, cv::Size(8,8), cv::Size(32,32), 1.2, 2 );

		for (auto& detection : detections ){
			cv::rectangle(imgFromStream, detection.tl(), detection.br(), cv::Scalar(255, 0, 0), 2 );
		}
		// show detected2 image
		cv::imshow("Detected Image", imgCanny); //show the thresholded image
*/

		// exit -------------------------------------------------------------------------------
		if ( cv::waitKey(1) == 27 ) {
			std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
			break;
		}
	}

	cv::destroyAllWindows(); //Destroy all opened windows

    return 0;
}