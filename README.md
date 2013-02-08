image2map
=========

A program for generating "maps" from image files

libraries:
OpenCV ()



---------compiling (LINUX):----------
mkdir build
cd build
cmake ..
make
-------------------------------------
--------------running:---------------
(inside /build)
./MapCreator <image_file_name>

You also find a script (run.sh) for running the program on every image in the directory "images". Just put images in there and run the script from the project root directory.
-------------------------------------

NOTES:
A map is a text file where every line has the 2d coordinates for a landmark

Image files are supposed to be "mainly white" with "mainly dark" areas, which represent landmarks.

Examples are available in the "images" directory.

Output maps will be placed in a directory called "created_maps"