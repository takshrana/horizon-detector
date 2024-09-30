#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include "optical_flow.h"

using namespace std;
using namespace cv;

// Function declarations
int opticalFlow(const Mat &prevFrame, const Mat &curFrame);

void processVideoFeed() {
    string ip_camera_url = "http://192.168.10.123:8080/video";

    VideoCapture cap(ip_camera_url); // Open the default camera (0 for default camera)

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    cv::Mat prevFrame, curFrame;
    
    // Capture the first frame
    cap >> prevFrame;

    if (prevFrame.empty()) {
        std::cerr << "Error: Could not read frame." << std::endl;
        return;
    }

    while (true) {
        // Capture the current frame
        cap >> curFrame;

        if (curFrame.empty()) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break; // Break the loop if there's an error
        }

        // Call the opticalFlow function with the previous and current frames
        int result = opticalFlow(prevFrame, curFrame);

        // Show the current frame (optional)
        cv::imshow("Current Frame", curFrame);

        // Check for exit condition (press 'q' to exit)
        if (cv::waitKey(30) >= 0) break;

        // Update previous frame
        prevFrame = curFrame.clone();
    }

    // Release the camera and destroy windows
    cap.release();
    cv::destroyAllWindows();
}

int main() {
    processVideoFeed();
    return 0;
}
