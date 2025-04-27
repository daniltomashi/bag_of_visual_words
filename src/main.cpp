#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <format>
#include <unordered_map>
#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>

#include "KMeans.hpp"
#include "comparison.cpp"
#include "preprocess.cpp"

#include <algorithm>




int main() {
    std::string path = "../data/images";

    cv::Mat all_descriptors;

    // read all of the images from "../data/" directoryc
    // then calculate their SIFT features (descriptors) and put into vector
    for (const auto& entry : std::filesystem::directory_iterator(path)) {

        const cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        

        // create Sift and extract features (100 features)
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        // put Sift features into keypoints vector
        sift->detect(image, keypoints);
        sift->compute(image, keypoints, descriptors);
        
        // each element of descritor add to global descriptor value to use for K-Means
        for (int i=0; i < descriptors.rows; i++) {
            all_descriptors.push_back(descriptors.row(i));
        }
    
    }

    std::cout << "Overall data: " << all_descriptors.size() << std::endl;


    int n_clusters = 200;
    int max_iter = 10;
    
    KMeans k_means(n_clusters, max_iter);

    std::cout << "Fitting started" << std::endl;
    
    k_means.fit(all_descriptors);

    std::cout << "Fitting finished" << std::endl << std::endl;

    //////////////////////////// Test on one of the images
    // cv::Mat test_image = cv::imread("../data/images/clamperl.png", cv::IMREAD_GRAYSCALE);

    // cv::Ptr<cv::SIFT> sift_test = cv::SIFT::create();
    // std::vector<cv::KeyPoint> test_keypoints;
    // cv::Mat test_descriptors;

    // sift_test->detect(test_image, test_keypoints);
    // sift_test->compute(test_image, test_keypoints, test_descriptors);

    // std::vector<int> outputs = k_means.predict(test_descriptors);

    // for(int i=0; i < n_clusters; i++) {
    //     std::cout << "Cluster: " << i << " has " << outputs[i] << " points" << std::endl;
    // }
    //////////////////////////// Test on one of the images

    std::cout << "Extract SIFT features and predict clusters STARTED!" << std::endl;

    std::vector<std::vector<float>> all_outputs;
    std::vector<std::string> all_paths;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        const cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

        // save filename to separate vector
        all_paths.emplace_back(entry.path().filename().string());

        // create Sift and extract features (100 features)
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        // put Sift features into keypoints vector
        sift->detect(image, keypoints);
        sift->compute(image, keypoints, descriptors);
        
        // predict per each feature corresponding cluster
        std::vector<float> outputs = k_means.predict(descriptors);
        all_outputs.emplace_back(outputs);
    }

    std::cout << "Extract SIFT features and predict clusters FINISHED!" << std::endl << std::endl;

    std::cout << "-----------------------" << std::endl;

    std::cout << "Tf-Idf started" << std::endl;

    // use tf-idf algorithm to change all_outputs
    std::vector<std::vector<float>> data_after_tf_idf = tf_idf(all_outputs, n_clusters);

    std::cout << "Tf-Idf started" << std::endl << std::endl;


    // find most similar image for all images
    std::vector<std::string> results;
    for (int i=0; i < all_outputs.size(); i++) {
        float max_score = -1;
        int most_similar_image_index;
        for (int j=0; j < all_outputs.size(); j++) {
            if (i == j)
                continue;
            float similarity = cosine_similarity(all_outputs[i], all_outputs[j]);
            if (similarity >= max_score) {
                max_score = similarity;
                most_similar_image_index = j;
            }
        }
        std::string ss = std::format("Image {} very similar to {} with similarity: {}", all_paths[i], all_paths[most_similar_image_index], max_score);
        results.emplace_back(ss);
        std::cout << ss << std::endl;
    }


    std::cout << "Success" << std::endl;

    return 0;
}