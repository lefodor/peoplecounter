# peoplecounter
opencv project to count people in crowded street

course project for Gépi látás (GKLB_INTM038) at SZE UNI 2021-22/02
deadline: 2022-06-12

# How to build
3 programs in repository.
* hogpedestrians
   loads positive and negative image files from specified folders and fits SVM model on HOG features
   output is detectionoutput.yml used by the other programs
* hogtestimg
   test fitted model on single image specified in 1st argument, example command line call:  
   $ ./hogtestimg "../raw_1.jpg"
* hogteststream
   test fitted model with videos stream from webcam

1) Update CMakeLists.txt file with correct opencv path in variable OpenCV_DIR
   set(OpenCV_DIR "~/opencv/build")
2) Create build folder
   $ mkdir build; cd build;
3) Run cmake
   $ cmake ..
4) Build
   $ cmake --build .
5) Execute
   $ ./hogpedestrians # modell fitting  
   $ ./hogptestimg <path to your image file> # test on single image file  
   $ ./hogpteststream  # test with videostream input  

# ToDo:
 * ~~add counter for detections~~
 * treatment for overlapping detections
 * ~~create demo video~~
 * detector dynamic parametrization

 # Resources
 * http://vision.stanford.edu/teaching/cs231b_spring1213/papers/CVPR05_DalalTriggs.pdf
 * https://learnopencv.com/histogram-of-oriented-gradients/
 * https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
 * https://docs.opencv.org/4.x/examples.html
 * https://docs.opencv.org/4.x/d0/df8/samples_2cpp_2train_HOG_8cpp-example.html
 * https://docs.opencv.org/4.x/d8/d61/samples_2tapi_2hog_8cpp-example.html
 * https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
 * https://docs.opencv.org/4.x/d7/d9e/tutorial_video_write.html
 * https://www.youtube.com/watch?v=cvGEWBO0Vho
 * https://www.kaggle.com/datasets/jeffreyhfuentes/lego-minifigures?select=test