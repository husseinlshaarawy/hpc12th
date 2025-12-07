#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Bonus features - Advanced thresholding", "[bonus]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
    
    SECTION("Otsu threshold") {
        cv::Mat binary = otsuThreshold(test_img);
        REQUIRE(binary.rows == test_img.rows);
        REQUIRE(binary.cols == test_img.cols);
        REQUIRE(binary.channels() == 1);
    }
    
    SECTION("Adaptive Gaussian threshold") {
        cv::Mat binary = adaptiveGaussianThreshold(test_img, 5, 2.0);
        REQUIRE(binary.rows == test_img.rows);
        REQUIRE(binary.cols == test_img.cols);
    }
    
    SECTION("Sauvola threshold") {
        cv::Mat binary = sauvolaThreshold(test_img, 5, 0.5, 128.0);
        REQUIRE(binary.rows == test_img.rows);
        REQUIRE(binary.cols == test_img.cols);
    }
    
    SECTION("Niblack threshold") {
        cv::Mat binary = niblackThreshold(test_img, 5, -0.2);
        REQUIRE(binary.rows == test_img.rows);
        REQUIRE(binary.cols == test_img.cols);
    }
}
