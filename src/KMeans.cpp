#include "KMeans.hpp"
#include <vector>
#include <unordered_map>
#include <limits>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>



// input data (for example): (1280, 10). 1280 data points with 10 features each


KMeans::KMeans(int n_clusters, int max_iter) {
    n_clusters_ = n_clusters;
    max_iter_ = max_iter;
};
void KMeans::fit(cv::Mat data) {
    // create random matirx for centoids with float type within 0-1 range
    centroids_ = cv::Mat(n_clusters_, data.cols, CV_32F);
    double min_val;
    double max_val;
    cv::minMaxLoc(data, &min_val, &max_val);
    cv::randu(centroids_, min_val, max_val);


    for (int iter=0; iter < max_iter_; iter++) {
        // calculate euclidean distance between each point and each centroids
        cv::Mat dist = find_distances(data);

        // create dictionary, where keys = clusters and values = vectors corresponding to this clusters
        std::unordered_map<int, cv::Mat> dict_to_calc;
        dict_to_calc = assign_to_cluster(dist, data);

        // actual reassign new centroids
        for (const auto& [centroid_name, values] : dict_to_calc) {
            // std::cout << "Values shape: " << values.size() << std::endl;
            double mean = cv::mean(values)[0];
            centroids_.row(centroid_name) = mean;
        }
    }
};
cv::Mat KMeans::find_distances(cv::Mat data) const {
    cv::Mat dist;
    cv::batchDistance(data, centroids_, dist, CV_32F, cv::noArray(), cv::NORM_L2);

    return dist;
};
std::unordered_map<int, cv::Mat> KMeans::assign_to_cluster(cv::Mat dist, cv::Mat data) {
        // assign to each element specific label(cluster)
        std::unordered_map<int, cv::Mat> dict_to_calc;
        
        for (int i=0; i < dist.rows; i++) {
            float min_elem = std::numeric_limits<float>::max();
            int indx_of_min = 0;
            for (int j=0; j < dist.cols; j++) {
                float elem = dist.at<float>(i, j);
                if (min_elem >= elem) {
                    min_elem = elem;
                    indx_of_min = j;
                }
            }
            dict_to_calc[indx_of_min].push_back(data.row(i));
        }

        return dict_to_calc;
};
int KMeans::cluster_output(cv::Mat dist, cv::Mat data) const {
        // // assign to each element specific label(cluster)
        int cluster;

        float min_elem = std::numeric_limits<float>::max();
        int indx_of_min = 0;

        for (int j=0; j < dist.cols; j++) {
            float elem = dist.at<float>(0, j);
            if (min_elem >= elem) {
                min_elem = elem;
                indx_of_min = j;
            }
        }
        cluster = indx_of_min;

        return cluster;
}
std::vector<float> KMeans::predict(cv::Mat data) const {
    std::vector<float> features(n_clusters_, 0);

    for (int i=0; i < data.rows; i++) {
        cv::Mat row = data.row(i);
        cv::Mat dist = find_distances(row);
        int cluster = cluster_output(dist, row);

        features[cluster] += 1;
    }

    return features;
};
void KMeans::print_centroid(int row) {
    std::cout << centroids_.row(row) << std::endl;
}