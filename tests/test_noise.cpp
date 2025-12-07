#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Noise generation", "[noise]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
    
    SECTION("Salt and pepper noise with fixed seed") {
        unsigned int seed = 42;
        cv::Mat noisy1 = addSaltPepperNoise(test_img, 0.1, seed);
        cv::Mat noisy2 = addSaltPepperNoise(test_img, 0.1, seed);
        
        // With same seed, results should be identical
        cv::Mat diff;
        cv::absdiff(noisy1, noisy2, diff);
        REQUIRE(cv::countNonZero(diff) == 0);
    }
    
    SECTION("Gaussian noise with fixed seed") {
        unsigned int seed = 123;
        cv::Mat noisy1 = addGaussianNoise(test_img, 0.0, 10.0, seed);
        cv::Mat noisy2 = addGaussianNoise(test_img, 0.0, 10.0, seed);
        
        // With same seed, results should be identical
        cv::Mat diff;
        cv::absdiff(noisy1, noisy2, diff);
        REQUIRE(cv::countNonZero(diff) == 0);
    }
    
    SECTION("Speckle noise with fixed seed") {
        unsigned int seed = 456;
        cv::Mat noisy1 = addSpeckleNoise(test_img, 0.05, seed);
        cv::Mat noisy2 = addSpeckleNoise(test_img, 0.05, seed);
        
        // With same seed, results should be identical
        cv::Mat diff;
        cv::absdiff(noisy1, noisy2, diff);
        REQUIRE(cv::countNonZero(diff) == 0);
    }
}
