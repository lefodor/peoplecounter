#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <typeinfo>
#include <time.h>
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

    // timer -------------------------------------------------------------------------------
    time_t timer;
    struct tm y2k = {0};

    // 2022-05-24 00:00:00.0000
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 122; y2k.tm_mon = 4; y2k.tm_mday = 24;

    time(&timer);  /* get current time; same as: timer = time(NULL)  */

    std::stringstream str_seconds ;
    str_seconds << (int)difftime(timer, mktime(&y2k)) / 60 ;

    // start frame -------------------------------------------------------------------------
    cv::Mat imgFromStream;
    cv::VideoCapture cap(0);

    // create demo video
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::string demoName = "demo" + str_seconds.str() + ".avi" ;
    cv::VideoWriter demostream(demoName, 
        cv::VideoWriter::fourcc('M','J','P','G'), 10, 
        cv::Size(frame_width,frame_height));

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
            "LEGO: " + counter.str(), 
            cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 1., cv::Scalar(0, 255, 0), 2);

        demostream.write(imgFromStream) ;

        cv::imshow( "Reaper", imgFromStream );

        if( cv::waitKey( 1 ) == 27 )
        {
            return 0;
        }

    }

    cap.release();
    demostream.release();
    cv::destroyAllWindows();

    return 0;
}