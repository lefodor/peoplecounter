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
   $ ./hogtestimg "../testimg.jpeg"
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
