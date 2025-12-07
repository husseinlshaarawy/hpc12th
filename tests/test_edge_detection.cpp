#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Edge detection and sharpening", "[edge_detection]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
    
    SECTION("Sobel edge detection") {
        cv::Mat edges = sobelEdgeDetection(test_img, 3);
        REQUIRE(edges.rows == test_img.rows);
        REQUIRE(edges.cols == test_img.cols);
    }
    
    SECTION("Canny edge detection") {
        cv::Mat edges = cannyEdgeDetection(test_img, 50, 150);
        REQUIRE(edges.rows == test_img.rows);
        REQUIRE(edges.cols == test_img.cols);
        REQUIRE(edges.channels() == 1);
    }
    
    SECTION("Sharpen filter") {
        cv::Mat sharpened = sharpenFilter(test_img);
        REQUIRE(sharpened.rows == test_img.rows);
        REQUIRE(sharpened.cols == test_img.cols);
    }
    
    SECTION("Prewitt edge detection") {
        cv::Mat edges = prewittEdgeDetection(test_img);
        REQUIRE(edges.rows == test_img.rows);
        REQUIRE(edges.cols == test_img.cols);
    }
    
    SECTION("Laplacian edge detection") {
        cv::Mat edges = laplacianEdgeDetection(test_img, 3);
        REQUIRE(edges.rows == test_img.rows);
        REQUIRE(edges.cols == test_img.cols);
    }
}
