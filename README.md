# Actual logic in steps
- loop thourgh data/images for take all of the images
- via SIFT take descriptors
- run KMeans clustering algorithm for SIFT descriptors
- once more read images, take SIFT descriptors and predict cluster
- finally per each photo find most similar image

# Packages
bag_of_visual_words is package contains of next files:
- <b>KMeans (cpp and hpp)</b> -- recreates K Means clustering algorithm via C++
- <b>preprocess.cpp</b> -- recreates algorithm tf-idf via C++</b>
- <b>comparison.cpp</b> -- recreates cosine similarity algorithm via C++
- <b>main.cpp</b> -- implements bag of visual words algorithm

# How to run:
Go to terminal and run the following commands
```
mkdir build/
cd build/
cmake ../
make
./main
```

# ToDo:
1. Implement logic for k similar images, except just one
2. Implement parallel calculation via CUDA
