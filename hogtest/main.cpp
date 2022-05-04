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
    /*
    cv::Mat img = cv::imread("../pos_2.png") ;
    cv::imshow("pic",img) ;
    cv::waitKey() ;
    */
    
    cv::Mat imgFromStream;
    //cv::VideoCapture cap(0);

    cv::String obj_det_filename = "../../hogpedestrians/detectionoutput.yml";
    cv::HOGDescriptor hog;
    hog.load( obj_det_filename );

    // use image ---------------------------------------------------------------------------
    //imgFromStream = cv::imread("../../../save/pos/smp_2.png");

    if ( argc != 2 ){
        printf("usage: DisplayImage.out <image path>\n");
        return -1;
    }
    
    imgFromStream = cv::imread( argv[1], 1 );

    if( imgFromStream.empty() ){
        std::clog << "this is empty" << std::endl ;
        return 1 ;
    }

    std::vector< cv::Rect > detections;
    std::vector< double > foundWeights;

    hog.detectMultiScale( imgFromStream, detections, 0, cv::Size(8,8), cv::Size(), 2, 5, false );
    //hog.detectMultiScale( imgFromStream, detections, 1, cv::Size(4,4), cv::Size(), 2, 5, false );

    for ( size_t j = 0; j < detections.size(); j++ ) {
        //cv::Scalar color = cv::Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
        cv::Scalar color = cv::Scalar(0,255,0) ;
        //rectangle( img, detections[j], color, img.cols / 400 + 1 );
        //ResizeBox(detections[j]);
        cv::rectangle( imgFromStream, detections[j].tl(), detections[j].br(), color, 2) ; //img.cols / 400 + 1 );
    }
    
    cv::imshow( "Reaper", imgFromStream );

    if( cv::waitKey() == 27 ){
        return 0;
    }

/*
    // start frame -------------------------------------------------------------------------
	while ( true ) {
        
        bool bSuccess = cap.read(imgFromStream); // read a new frame from video 

		//Breaking the while loop at the end of the video
    
		if ( bSuccess == false )
		{
			std::cout << "Video camera is disconnected" << std::endl;
			break;
		}
        

        std::vector< cv::Rect > detections;
        std::vector< double > foundWeights;
        //hog.detectMultiScale( img, detections, foundWeights );
            // larger scale --> faster
            // lower scale --> slower and more false positive

        hog.detectMultiScale( imgFromStream, detections, 0, cv::Size(4,4), cv::Size(8,8), 2, 2 );
        
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            //cv::Scalar color = cv::Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            cv::Scalar color = cv::Scalar(255,0,0) ;
            //rectangle( img, detections[j], color, img.cols / 400 + 1 );
            //ResizeBox(detections[j]);
            cv::rectangle( imgFromStream, detections[j].tl(), detections[j].br(), color, 2) ; //img.cols / 400 + 1 );
        }
    
        cv::imshow( "Reaper", imgFromStream );

        if( cv::waitKey( 1 ) == 27 )
        {
            return 0;
        }

    }
*/
    return 0;
}