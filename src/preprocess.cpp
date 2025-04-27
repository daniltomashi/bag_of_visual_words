#include <vector>
#include <cmath>



// all_outputs contains (n, x), 
//                              n=number of photos, 
//                              x=per each x1, x2... we have number of occurances
std::vector<std::vector<float>> tf_idf(const std::vector<std::vector<float>>& all_outputs, int n_clusters) {
    std::vector<std::vector<float>> output(all_outputs.size(), std::vector<float>(n_clusters, 1.0f));

    std::vector<int> sentences_contains_word(n_clusters, 0);
    std::vector<float> document_vocabulary(all_outputs.size(), 0);
    float n_images = all_outputs.size();

    // find per each feature how many images contains it
    for (int sentence_ind=0; sentence_ind < n_images; sentence_ind++) {
        for (int feature_ind=0; feature_ind < n_clusters; feature_ind++) {
            float count_of_feature = all_outputs[sentence_ind][feature_ind];

            if (count_of_feature > 0) {
                sentences_contains_word[feature_ind] += 1;
                document_vocabulary[sentence_ind] += count_of_feature;
            }
        }
    }

    // per each sentence pair with feature find tf-idf value
    for (int sentence_ind=0; sentence_ind < n_images; sentence_ind++) {
        for (int feature_ind=0; feature_ind < n_clusters; feature_ind++) {
            float count_of_feature = all_outputs[sentence_ind][feature_ind];
            
            float tf = count_of_feature / document_vocabulary[sentence_ind];
            float idf = std::log((n_images + 1.0f) / (1.0f + sentences_contains_word[feature_ind])) + 1.0f;
            float tf_idf = tf * idf;

            output[sentence_ind][feature_ind] = tf_idf;
        }
    }

    return output;
}