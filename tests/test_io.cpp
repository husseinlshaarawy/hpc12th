#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"
#include <opencv2/opencv.hpp>

using namespace iproc;

TEST_CASE("Image I/O operations", "[io]") {
    SECTION("Load grayscale image") {
        cv::Mat img = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
        REQUIRE(img.rows == 16);
        REQUIRE(img.cols == 16);
        REQUIRE(img.channels() == 1);
    }
    
    SECTION("Load color image") {
        cv::Mat img = loadImage("tests/data/test_16x16_color.png", cv::IMREAD_COLOR);
        REQUIRE(img.rows == 16);
        REQUIRE(img.cols == 16);
        REQUIRE(img.channels() == 3);
    }
    
    SECTION("Save and reload image") {
        cv::Mat original = loadImage("tests/data/test_16x16.png");
        REQUIRE(saveImage("/tmp/test_save.png", original));
        
        cv::Mat reloaded = loadImage("/tmp/test_save.png");
        REQUIRE(reloaded.rows == original.rows);
        REQUIRE(reloaded.cols == original.cols);
    }
}
