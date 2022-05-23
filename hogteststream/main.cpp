#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/xobjdetect.hpp>
#include "opencv2/ml.hpp"
#include "opencv2/videoio.hpp"

#include "getDetector.hpp"

int main(int argc, char** argv )
{

    cv::String obj_det_filename = "../../hogpedestrians/detectionoutput.yml";

    cv::HOGDescriptor hog;
    hog.load( obj_det_filename );

    // start frame -------------------------------------------------------------------------
    cv::Mat imgFromStream;
    cv::VideoCapture cap(0);
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

        //hog.detectMultiScale( imgFromStream, detections, 0, cv::Size(4,4), cv::Size(8,8), 2, 2 );
        hog.detectMultiScale( imgFromStream, detections, .8, cv::Size(4,4), cv::Size(0,0), 2, 1.3, false );
        
        for ( size_t j = 0; j < detections.size(); j++ ) {
            cv::Scalar color = cv::Scalar(0,255,0) ;
            cv::rectangle( imgFromStream, detections[j].tl(), detections[j].br(), color, 2) ; //img.cols / 400 + 1 );
        }

        std::stringstream counter ;
        counter << detections.size() ;
        cv::putText(imgFromStream, 
            "LEGO cnt: " + counter.str(), 
            cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(0, 255, 0), 2);
    
        cv::imshow( "Reaper", imgFromStream );

        if( cv::waitKey( 1 ) == 27 )
        {
            return 0;
        }

    }

    return 0;
}