#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Point operations", "[point_ops]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16_color.png");
    
    SECTION("Convert to grayscale") {
        cv::Mat gray = convertToGrayscale(test_img);
        REQUIRE(gray.channels() == 1);
        REQUIRE(gray.rows == test_img.rows);
        REQUIRE(gray.cols == test_img.cols);
    }
    
    SECTION("Adjust brightness") {
        cv::Mat bright = adjustBrightness(test_img, 50);
        REQUIRE(bright.rows == test_img.rows);
        REQUIRE(bright.cols == test_img.cols);
        
        // Check that brightness increased
        double sum_orig = cv::sum(test_img)[0];
        double sum_bright = cv::sum(bright)[0];
        REQUIRE(sum_bright > sum_orig);
    }
    
    SECTION("Adjust contrast") {
        cv::Mat contrast = adjustContrast(test_img, 1.5);
        REQUIRE(contrast.rows == test_img.rows);
        REQUIRE(contrast.cols == test_img.cols);
    }
    
    SECTION("Binary threshold") {
        cv::Mat gray = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
        cv::Mat binary = threshold(gray, 128);
        REQUIRE(binary.channels() == 1);
        
        // Check that all values are either 0 or 255
        bool all_binary = true;
        for (int i = 0; i < binary.rows; ++i) {
            for (int j = 0; j < binary.cols; ++j) {
                uchar val = binary.at<uchar>(i, j);
                if (val != 0 && val != 255) {
                    all_binary = false;
                }
            }
        }
        REQUIRE(all_binary);
    }
    
    SECTION("Invert image") {
        cv::Mat inverted = invert(test_img);
        REQUIRE(inverted.rows == test_img.rows);
        REQUIRE(inverted.cols == test_img.cols);
        
        // Check specific pixel inversion
        for (int i = 0; i < test_img.rows; i += 4) {
            for (int j = 0; j < test_img.cols; j += 4) {
                cv::Vec3b orig = test_img.at<cv::Vec3b>(i, j);
                cv::Vec3b inv = inverted.at<cv::Vec3b>(i, j);
                REQUIRE(inv[0] == 255 - orig[0]);
                REQUIRE(inv[1] == 255 - orig[1]);
                REQUIRE(inv[2] == 255 - orig[2]);
            }
        }
    }
    
    SECTION("Gamma correction") {
        cv::Mat gamma = gammaCorrection(test_img, 0.5);
        REQUIRE(gamma.rows == test_img.rows);
        REQUIRE(gamma.cols == test_img.cols);
    }
}
