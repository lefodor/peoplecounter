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

///home/woodrat/opencv/opencv/modules/objdetect/src/hog.cpp

void ResizeBox(cv::Rect& box){
    box.x += cvRound(box.width*0.1) ;
    box.width = cvRound(box.width*0.8) ;
    box.y += cvRound(box.height*0.06) ;
    box.height = cvRound(box.height*0.8) ;
}

std::vector< float > get_svm_detector( const cv::Ptr< cv::ml::SVM >& svm )
{
    // get the support vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );
    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    std::vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/

void load_images( const cv::String & dirname, std::vector< cv::Mat > & img_lst, bool showImages = false )
{
    std::vector< cv::String > files;
    cv::glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        cv::Mat img = cv::imread( files[i] ); // load the image
        if ( img.empty() )
        {
            std::cout << files[i] << " is invalid!" << std::endl; // invalid image, skip it.
            continue;
        }
        if ( showImages )
        {
            cv::imshow( "image", img );
            cv::waitKey( 1 );
        }
        img_lst.push_back( img );
    }
};

// image greater than detector then take sample from image
// otherwise take full image
void sample_neg( const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size )
{
    cv::Rect box;
    box.width = size.width; //64 320
    box.height = size.height; //128 240
    srand( (unsigned int)time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - box.width );//320-64=[0,256]
            box.y = rand() % ( full_neg_lst[i].rows - box.height );//240-128=[0,112]
            cv::Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
            //std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/sample_neg/smpneg_" + std::to_string(i) + ".png" ;
			//cv::imwrite(filename,roi) ;
        }
        // £££ save image with same size as detector
        /*else{
            neg_lst.push_back( full_neg_lst[i].clone() );
        }
        */
}

//computeHOGs( pos_image_size, pos_lst, gradient_lst, false );
//computeHOGs( pos_image_size, neg_lst, gradient_lst, false );
void computeHOGs( const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip )
{
    cv::HOGDescriptor hog;
    hog.winSize = wsize; //64x128
    cv::Mat gray;
    std::vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            // crop from middle of image
            cv::Rect r = cv::Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);

            //cv::cvtColor( img_lst[i](r), gray, cv::COLOR_BGR2GRAY );
            gray = img_lst[i](r);

            // no cropping
            /*
            cv::resize( img_lst[i], gray, wsize ); 
            cv::cvtColor( gray, gray, cv::COLOR_BGR2GRAY );
            */

            // £££ output images
            /*
            if( img_lst.size() > 100 ){
                std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/computeHOG/hogneg_" + std::to_string(i) + ".png" ;
			    cv::imwrite(filename,gray) ;
            }
            else{
                std::string filename = "/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/computeHOG/hogpos_" + std::to_string(i) + ".png" ;
			    cv::imwrite(filename,gray) ;
            }
            */
            
            hog.compute( gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
            gradient_lst.push_back( cv::Mat( descriptors ).clone() );

            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
                gradient_lst.push_back( cv::Mat( descriptors ).clone() );
            }
        }
    }
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/

//cv::Mat train_data;
//convert_to_ml( gradient_lst, train_data );
void convert_to_ml( const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = cv::Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void test_trained_detector( cv::String obj_det_filename, cv::String test_dir, cv::String videofilename )
{
    std::cout << "Testing trained detector..." << std::endl;
    cv::HOGDescriptor hog;
    hog.load( obj_det_filename );
    std::vector< cv::String > files;
    cv::glob( test_dir, files );
    int delay = 0;
    cv::VideoCapture cap;
    if ( videofilename != "" )
    {
        if ( videofilename.size() == 1 && isdigit( videofilename[0] ) )
            cap.open( videofilename[0] - '0' );
        else
            cap.open( videofilename );
    }
    obj_det_filename = "testing " + obj_det_filename;
    cv::namedWindow( obj_det_filename, cv::WINDOW_NORMAL );
    for( size_t i=0;; i++ )
    {
        cv::Mat img;
        if ( cap.isOpened() )
        {
            cap >> img;
            delay = 1;
        }
        else if( i < files.size() )
        {
            img = cv::imread( files[i] );
        }
        if ( img.empty() )
        {
            return;
        }

        // save original
        //cv::Mat imgOrig = img ;

        std::vector< cv::Rect > detections;
        std::vector< double > foundWeights;
        //hog.detectMultiScale( img, detections, foundWeights );
            // larger scale --> faster
            // lower scale --> slower and more false positive
        hog.detectMultiScale( img, detections, 0, cv::Size(4,4), cv::Size(8,8), 1.1, 2 );
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            //cv::Scalar color = cv::Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            cv::Scalar color = cv::Scalar(255,0,0) ;
            //rectangle( img, detections[j], color, img.cols / 400 + 1 );
            //ResizeBox(detections[j]);
            cv::rectangle( img, detections[j].tl(), detections[j].br(), color, 2) ; //img.cols / 400 + 1 );
        }
        cv::imshow( obj_det_filename, img );
        if( cv::waitKey( delay ) == 27 )
        {
            return;
        }
    }
}

int main(int argc, char** argv )
{
	//cv::Mat imgOriginal = cv::imread("/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/sample2.jpeg",1) ;
    //cv::Mat imgOriginal = cv::imread("/home/woodrat/projects/szeuni/computervision/samples/hogpedestrians/sample2.jpeg",1) ;
	//cv::resize(imgOriginal, imgOriginal, cv::Size(640, 480));
	//cv::resize(imgOriginal, imgOriginal, cv::Size(0,0), .5, .5);

	//CreatSmpls() ; // for creating samples

	cv::String pos_dir = "/home/woodrat/projects/szeuni/computervision/peoplecounter/hogpedestrians/pos/";
    cv::String neg_dir = "/home/woodrat/projects/szeuni/computervision/peoplecounter/hogpedestrians/neg/";
    cv::String test_dir = "/home/woodrat/projects/szeuni/computervision/peoplecounter/hogpedestrians/test/";
    cv::String videofilename = "sample6.jpg";
    cv::String imgfilename = "sample4.jpg";
	cv::String obj_det_filename = "../detectionoutput.yml";
    int detector_width =64;//= 64 320 ; // parser.get< int >( "dw" );
    int detector_height =128;//= 128 240 ; // parser.get< int >( "dh" );
	
    std::vector< cv::Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
    std::vector< int > labels;

	std::clog << "Positive images are being loaded..." ;
    load_images( pos_dir, pos_lst, false );
    if ( pos_lst.size() > 0 )
    {
        std::clog << "...[done] " << pos_lst.size() << " files." << std::endl;
    }
    else
    {
        std::clog << "no image in " << pos_dir << std::endl;
        return 1;
    }

    cv::Size pos_image_size ; //= pos_lst[0].size();
    //std::cout << pos_image_size << std::endl ;

    if ( detector_width && detector_height )
    {
        pos_image_size = cv::Size( detector_width, detector_height );
    }
    else
    {
        for ( size_t i = 0; i < pos_lst.size(); ++i )
        {
            if( pos_lst[i].size() != pos_image_size )
            {
                std::cout << "All positive images should be same size!" << std::endl;
                exit( 1 );
            }
        }
        pos_image_size = pos_image_size / 8 * 8;
    }

    std::cout << "pos_image_size: " << pos_image_size << std::endl ;

    std::clog << "Negative images are being loaded...";
    load_images( neg_dir, full_neg_lst, false );
    std::clog << "...[done] " << full_neg_lst.size() << " files." << std::endl;

    std::clog << "Negative images are being processed...";
    sample_neg( full_neg_lst, neg_lst, pos_image_size ); // [64x128]
    std::clog << "...[done] " << neg_lst.size() << " files." << std::endl;

    std::clog << "Histogram of Gradients are being calculated for positive images...";
    computeHOGs( pos_image_size, pos_lst, gradient_lst, false );
    size_t positive_count = gradient_lst.size();
    labels.assign( positive_count, +1 );
    std::clog << "...[done] ( positive images count : " << positive_count << " )" << std::endl;

    std::clog << "Histogram of Gradients are being calculated for negative images...";
    computeHOGs( pos_image_size, neg_lst, gradient_lst, false );
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, -1 );
    CV_Assert( positive_count < labels.size() ); // returns error if expression is false
    std::clog << "...[done] ( negative images count : " << negative_count << " )" << std::endl;
    
	cv::Mat train_data;
    convert_to_ml( gradient_lst, train_data );
    std::clog << "Training SVM...";
    cv::Ptr< cv::ml::SVM > svm = cv::ml::SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( cv::ml::SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( cv::ml::SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train( train_data, cv::ml::ROW_SAMPLE, labels );
    std::clog << "...[done]" << std::endl;

    cv::HOGDescriptor hog;
    hog.winSize = pos_image_size;
    hog.setSVMDetector( get_svm_detector( svm ) );
    hog.save( obj_det_filename );

    //test_trained_detector( obj_det_filename, test_dir, videofilename );

    
    cv::Mat imgFromStream;
    cv::VideoCapture cap(0);
    cv::namedWindow( "Reaper", cv::WINDOW_NORMAL );
    //hog.load( obj_det_filename );

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

    return 0;
}