#include <vector>
#include <cmath>



float cosine_similarity(std::vector<float> a, std::vector<float> b) {
    float dot_product = 0;
    float a_norm = 0;
    float b_norm = 0;

    for (int i=0; i < a.size(); i++) {
        dot_product += a[i] * b[i];
        a_norm += std::pow(a[i], 2);
        b_norm += std::pow(b[i], 2);
    }

    a_norm = std::sqrt(a_norm);
    b_norm = std::sqrt(b_norm);

    return dot_product / (a_norm * b_norm);
}