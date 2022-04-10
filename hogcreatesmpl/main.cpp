#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/xobjdetect.hpp>
#include "opencv2/ml.hpp"
#include "opencv2/videoio.hpp"


int main(int argc, char** argv )
{
	cv::Mat imgGray   ;    // grayscale image
	cv::VideoCapture cap(0);
	cv::namedWindow( "video", cv::WINDOW_AUTOSIZE );

	int fcnt = 0 ;
	int cnt_pos = 0;
	int cnt_neg = 0;
	bool pos = false ;
	if( argv[1] =="n" ){
		pos = true ;
	}

	while ( true ) {
	
		// get video
		cv::Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video 

		cv::resize(imgOriginal, imgOriginal, cv::Size(320, 240));

		// create grayscale image
		//cv::cvtColor(imgOriginal, imgGray, cv::COLOR_BGR2GRAY);

		//cv::Mat imgGray_histeq ;
		//equalizeHist(imgGray, imgGray);

		fcnt++ ;
/*
		if(pos){
			if( fcnt % 60 == 0){ 
				int cnt = fcnt/60 ;
				std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/pos/smp_" + std::to_string(cnt) + ".png" ;
				cv::imwrite(filename,imgGray) ;
				std::cout << "pos image saved " << std::to_string(cnt)  << std::endl ;
			}		
		}
		else {
			if( fcnt % 1 == 0){ //neg
				int cnt = fcnt/1 ;
				std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/neg/smp_" + std::to_string(cnt) + ".png" ;
				cv::imwrite(filename,imgGray) ;
				std::cout << "neg image saved " << std::to_string(cnt)  << std::endl ;
				}
		}
*/

		// show detected2 image
		cv::imshow("video", imgOriginal); //show the thresholded image

		// exit -------------------------------------------------------------------------------
		int comm = cv::waitKey(10);

		// keyboard input ---------------------------------------------------------------------
		if ( comm == 27 ) {
			std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
			return 0;
		}
		else if( comm == 113 ){ // q - positive
			cnt_pos++ ;
			std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/pos/smp_" + std::to_string(cnt_pos) + ".png" ;
			cv::imwrite(filename,imgOriginal) ;
			std::cout << "pos image saved " << std::to_string(cnt_pos)  << std::endl ;
		}
		else if( comm == 119 ){ // w - negative
			cnt_neg++ ;
			std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/neg/smp_" + std::to_string(cnt_neg) + ".png" ;
			cv::imwrite(filename,imgOriginal) ;
			std::cout << "neg image saved " << std::to_string(cnt_neg)  << std::endl ;
		}
	}

    cv::destroyAllWindows(); //Destroy all opened windows

	return 0 ;
}