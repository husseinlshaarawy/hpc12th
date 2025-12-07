#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Smoothing filters", "[smoothing]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
    
    SECTION("Box blur") {
        cv::Mat blurred = boxBlur(test_img, 3);
        REQUIRE(blurred.rows == test_img.rows);
        REQUIRE(blurred.cols == test_img.cols);
        REQUIRE(blurred.channels() == test_img.channels());
        
        // Blurred image should be smoother (less variation)
        cv::Scalar mean_orig, stddev_orig;
        cv::Scalar mean_blur, stddev_blur;
        cv::meanStdDev(test_img, mean_orig, stddev_orig);
        cv::meanStdDev(blurred, mean_blur, stddev_blur);
        
        REQUIRE(stddev_blur[0] < stddev_orig[0]);
    }
    
    SECTION("Gaussian blur") {
        cv::Mat blurred = gaussianBlur(test_img, 5, 1.0);
        REQUIRE(blurred.rows == test_img.rows);
        REQUIRE(blurred.cols == test_img.cols);
        REQUIRE(blurred.channels() == test_img.channels());
    }
    
    SECTION("Median filter") {
        cv::Mat filtered = medianFilter(test_img, 3);
        REQUIRE(filtered.rows == test_img.rows);
        REQUIRE(filtered.cols == test_img.cols);
        REQUIRE(filtered.channels() == test_img.channels());
    }
    
    SECTION("Bilateral filter") {
        cv::Mat filtered = bilateralFilter(test_img, 5, 50.0, 50.0);
        REQUIRE(filtered.rows == test_img.rows);
        REQUIRE(filtered.cols == test_img.cols);
        REQUIRE(filtered.channels() == test_img.channels());
    }
}
