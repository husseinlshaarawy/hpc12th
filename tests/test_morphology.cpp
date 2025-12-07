#include <catch2/catch_test_macros.hpp>
#include "image_filters.h"

using namespace iproc;

TEST_CASE("Morphological operations", "[morphology]") {
    cv::Mat test_img = loadImage("tests/data/test_16x16.png", cv::IMREAD_GRAYSCALE);
    cv::Mat binary = threshold(test_img, 128);
    
    SECTION("Erosion") {
        cv::Mat eroded = erosion(binary, 3, 1);
        REQUIRE(eroded.rows == binary.rows);
        REQUIRE(eroded.cols == binary.cols);
    }
    
    SECTION("Dilation") {
        cv::Mat dilated = dilation(binary, 3, 1);
        REQUIRE(dilated.rows == binary.rows);
        REQUIRE(dilated.cols == binary.cols);
    }
    
    SECTION("Opening") {
        cv::Mat opened = opening(binary, 3);
        REQUIRE(opened.rows == binary.rows);
        REQUIRE(opened.cols == binary.cols);
    }
    
    SECTION("Closing") {
        cv::Mat closed = closing(binary, 3);
        REQUIRE(closed.rows == binary.rows);
        REQUIRE(closed.cols == binary.cols);
    }
    
    SECTION("Morphological reconstruction") {
        cv::Mat marker = erosion(binary, 5, 1);
        cv::Mat reconstructed = morphologicalReconstruction(marker, binary, 4);
        REQUIRE(reconstructed.rows == binary.rows);
        REQUIRE(reconstructed.cols == binary.cols);
    }
}
