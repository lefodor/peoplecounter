# peoplecounter
opencv project to count people in crowded street

course project for Gépi látás (GKLB_INTM038) at SZE UNI 2021-22/02
deadline: 2022-06-12

# How to build
1) Update CMakeLists.txt file with correct opencv path in variable OpenCV_DIR
   set(OpenCV_DIR "~/opencv/build")
2) Create build folder
   $ mkdir build; cd build;
3) Run cmake
   $ cmake ..
4) Build
   $ cmake --build .
5) Execute
   $ ./peoplecounter <path to your image file>