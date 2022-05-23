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
    
    cv::Mat imgFromStream;

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

    //hog.detectMultiScale( imgFromStream, detections, 0, cv::Size(4,4), cv::Size(8,8), 2, 2 );
    hog.detectMultiScale( imgFromStream, detections, .8, cv::Size(4,4), cv::Size(0,0), 2, 1.3, false );

    for ( size_t j = 0; j < detections.size(); j++ ) {
        cv::Scalar color = cv::Scalar(0,255,0) ;
        cv::rectangle( imgFromStream, detections[j].tl(), detections[j].br(), color, 2) ; //img.cols / 400 + 1 );
    }
    
    cv::imshow( "Reaper", imgFromStream );

    if( cv::waitKey() == 27 ){
        return 0;
    }

    return 0;
}