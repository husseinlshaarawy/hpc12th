#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Geometric transformations", "[geometric]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16.png");
    
    SECTION("Rotation") {
        cv::Mat rotated = rotate(test_img, 45.0);
        REQUIRE(rotated.rows == test_img.rows);
        REQUIRE(rotated.cols == test_img.cols);
    }
    
    SECTION("Scaling") {
        cv::Mat scaled = scale(test_img, 2.0, 2.0);
        REQUIRE(scaled.rows == test_img.rows * 2);
        REQUIRE(scaled.cols == test_img.cols * 2);
        
        cv::Mat scaled_down = scale(test_img, 0.5, 0.5);
        REQUIRE(scaled_down.rows == test_img.rows / 2);
        REQUIRE(scaled_down.cols == test_img.cols / 2);
    }
    
    SECTION("Translation") {
        cv::Mat translated = translate(test_img, 5, 3);
        REQUIRE(translated.rows == test_img.rows);
        REQUIRE(translated.cols == test_img.cols);
    }
    
    SECTION("Horizontal flip") {
        cv::Mat flipped = flipHorizontal(test_img);
        REQUIRE(flipped.rows == test_img.rows);
        REQUIRE(flipped.cols == test_img.cols);
    }
    
    SECTION("Vertical flip") {
        cv::Mat flipped = flipVertical(test_img);
        REQUIRE(flipped.rows == test_img.rows);
        REQUIRE(flipped.cols == test_img.cols);
    }
    
    SECTION("Perspective transform") {
        std::vector<cv::Point2f> src_points = {
            cv::Point2f(0, 0),
            cv::Point2f(15, 0),
            cv::Point2f(15, 15),
            cv::Point2f(0, 15)
        };
        std::vector<cv::Point2f> dst_points = {
            cv::Point2f(2, 2),
            cv::Point2f(13, 3),
            cv::Point2f(13, 13),
            cv::Point2f(3, 12)
        };
        
        cv::Mat transformed = perspectiveTransform(test_img, src_points, dst_points);
        REQUIRE(transformed.rows == test_img.rows);
        REQUIRE(transformed.cols == test_img.cols);
    }
}
