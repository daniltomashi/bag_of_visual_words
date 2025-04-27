#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <unordered_map>
#include <vector>


class KMeans {
    public:
        KMeans(int n_clusters, int max_iter);
        cv::Mat find_distances(cv::Mat data) const;
        std::unordered_map<int, cv::Mat> assign_to_cluster(cv::Mat dist, cv::Mat data);
        void fit(cv::Mat data);
        int cluster_output(cv::Mat dist, cv::Mat data) const;
        std::vector<float> predict(cv::Mat data) const;

        void print_centroid(int row);
    private:
        int n_clusters_;
        int max_iter_;
        cv::Mat centroids_;
};