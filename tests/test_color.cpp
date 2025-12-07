#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Color and channel operations", "[color]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16_color.png");
    
    SECTION("Split channels") {
        std::vector<cv::Mat> channels = splitChannels(test_img);
        REQUIRE(channels.size() == 3);
        REQUIRE(channels[0].rows == test_img.rows);
        REQUIRE(channels[0].channels() == 1);
    }
    
    SECTION("Merge channels") {
        std::vector<cv::Mat> channels = splitChannels(test_img);
        cv::Mat merged = mergeChannels(channels);
        REQUIRE(merged.rows == test_img.rows);
        REQUIRE(merged.cols == test_img.cols);
        REQUIRE(merged.channels() == 3);
    }
    
    SECTION("HSV adjustments") {
        cv::Mat adjusted = adjustHSV(test_img, 30, 1.2, 1.1);
        REQUIRE(adjusted.rows == test_img.rows);
        REQUIRE(adjusted.cols == test_img.cols);
        REQUIRE(adjusted.channels() == 3);
    }
    
    SECTION("Lab tone mapping") {
        cv::Mat mapped = labToneMapping(test_img, 1.2);
        REQUIRE(mapped.rows == test_img.rows);
        REQUIRE(mapped.cols == test_img.cols);
        REQUIRE(mapped.channels() == 3);
    }
}
