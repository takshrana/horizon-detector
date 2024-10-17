#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

const int rows = 360, cols = 640;
const float top_ad = 200, bottom_ad = -30, left_ad = 20, right_ad = -20;
const float LSD_strength_threshold = 10.0;
bool DrawCrossPoint2f = true;
const float center_x = cols/2,  center_y = rows/2;
const float EPSILON = 1e-6;  // Precision threshold for comparing floating-point values

bool DrawCrossPoint = true;

Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(0);

// // Function to calculate line length
float LSDLineLength(Vec4f line) {
    //Manhattan distance
    return abs(line[2] - line[0]) + abs(line[3] - line[1]);
}

float LSDLineTheta(Vec4f line){
    double dx = line[2] - line[0];
    double dy = line[3] - line[1];

    // Handle vertical line case
    if (dx == 0) {
        return (dy > 0) ? 90.0 : 270.0;  // 90 degrees up or 270 degrees down
    }

    // Calculate angle in radians
    double theta_radians = atan2(dy, dx);

    // Convert to degrees
    double theta_degrees = theta_radians * (180.0 / M_PI);

    // Adjust theta to be in the range of 0 to 180
    if (theta_degrees < 0) {
        theta_degrees += 180;
    }

    return theta_degrees;
}

// LSD filtering function
vector<Vec6f> LSDFilter(const vector<Vec4f>& inputLines, const vector<float> &width) {
    vector<Vec6f> lines;
    for(int i = 0; i < inputLines.size(); i++) {
        float strength = LSDLineLength(inputLines[i])/width[i];
        float theta = LSDLineTheta(inputLines[i]);
        // cout<<strength<<endl;
        if (strength > LSD_strength_threshold && ((theta>20 && theta<70) || (theta>110 && theta<170))) {
            Vec6f point = Vec6f(inputLines[i][0] + left_ad, inputLines[i][1] + top_ad,
                                inputLines[i][2] + left_ad, inputLines[i][3] + top_ad,
                                strength);    
            lines.push_back(Vec6f(point));
        }
    }
    return lines;
}

// Function to detect lines using LSD
vector<Vec6f> LSD(Mat& showImage, Mat& src) {
    Mat gray;
    if (src.channels() > 2) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    vector<Vec4f> lines;
    vector<float> width;

    float crop_width = cols - (left_ad - right_ad), crop_height = rows - (top_ad - bottom_ad);
    // (Rect(center_x,center_y, left_ad - right_ad, top_ad - bottom_ad ))
    lsd->detect(gray(Rect(left_ad, top_ad, crop_width, crop_height)), lines, width);
    
    // cout<<lines.size()<<endl;
    vector<Vec6f> filter_lines = LSDFilter(lines, width);
    
    for(int i = 0 ; i<filter_lines.size(); i++){
        line(showImage, Point(filter_lines[i][0], filter_lines[i][1]),
                        Point(filter_lines[i][2], filter_lines[i][3]), Scalar(0, 0, 255), 3);
    }

    return filter_lines;
}

Vec2f GetCrossPoint(Vec6f LineA, Vec6f LineB) {
    // Calculate the denominators for the slopes
    float denominatorA = LineA[2] - LineA[0];  // LineA x2 - x1
    float denominatorB = LineB[2] - LineB[0];  // LineB x2 - x1

    // Check if either line is vertical
    if (fabs(denominatorA) < EPSILON && fabs(denominatorB) < EPSILON) {
        // Both lines are vertical (no intersection)
        // cout << "Both lines are vertical." << endl;
        return Vec2f(-1, -1);  // Indicating no intersection
    }

    if (fabs(denominatorA) < EPSILON) {
        // LineA is vertical, intersection is along x = LineA[0]
        float x = LineA[0];
        float y = (LineB[3] - LineB[1]) / denominatorB * (x - LineB[0]) + LineB[1];
        return Vec2f(x, y);
    }

    if (fabs(denominatorB) < EPSILON) {
        // LineB is vertical, intersection is along x = LineB[0]
        float x = LineB[0];
        float y = (LineA[3] - LineA[1]) / denominatorA * (x - LineA[0]) + LineA[1];
        return Vec2f(x, y);
    }

    // Calculate the slopes of the lines
    float ka = (LineA[3] - LineA[1]) / denominatorA;  // Slope of LineA
    float kb = (LineB[3] - LineB[1]) / denominatorB;  // Slope of LineB

    // Check if lines are parallel
    if (fabs(ka - kb) < EPSILON) {
        // Lines are parallel (no intersection)
        // cout << "Lines are parallel." << endl;
        return Vec2f(-1, -1);  // Indicating no intersection
    }

    // Calculate the x and y coordinates of the intersection point
    float x = (ka * LineA[0] - LineA[1] - kb * LineB[0] + LineB[1]) / (ka - kb);
    float y = (ka * kb * (LineA[0] - LineB[0]) + ka * LineB[1] - kb * LineA[1]) / (ka - kb);

    return Vec2f(x, y);  // Return as Vec2f (float coordinates)
}


Point2f CalcLineCrossPoint(Mat& showImage, vector<Vec6f>& lines){
    Mat strength_img = Mat::zeros(showImage.size(), CV_32F);

    Vec3f max_strength_point(0, 0, 0);

    for (int i = 0; i < lines.size(); i++) {
        for (int j = 0; j < lines.size(); j++) {
            if (i != j) {
                Vec2f crossPoint = GetCrossPoint(lines[i], lines[j]);  // Get the intersection point

                // Check if the crossPoint is within the image bounds
                if (crossPoint[0] > 0 && crossPoint[0] < cols && crossPoint[1] > 0 && crossPoint[1] < rows) {
                    int x = static_cast<int>(crossPoint[0]);
                    int y = static_cast<int>(crossPoint[1]);

                    // Accumulate strength at that point
                    strength_img.at<float>(y, x) += lines[i][4] + lines[j][4]; // Assuming lines[i][4] contains the strength

                    // Update max_strength_point if the current strength is greater
                    if (strength_img.at<float>(y, x) > max_strength_point[2]) {
                        max_strength_point[0] = crossPoint[0];
                        max_strength_point[1] = crossPoint[1];
                        max_strength_point[2] = strength_img.at<float>(y, x);
                    }
                }
            }
        }
    }
    if(DrawCrossPoint){
        imshow("crosspoint_img", strength_img);  // Display the strength image

    }


    // Draw the maximum strength point on the image
    
    circle(showImage, Point(max_strength_point[0], max_strength_point[1]), 10, Scalar(255, 0, 0), -1);

    return Point2f(max_strength_point[0], max_strength_point[1]);
}

Point2f calculateAverage(const deque<Point2f>& points) {
    float sum_x = 0, sum_y = 0;
    for (const auto& point : points) {
        sum_x += point.x;
        sum_y += point.y;
    }
    return Point2f(sum_x / points.size(), sum_y / points.size());
}

// Function to compute the median of vanishing points in the buffer
Point2f calculateMedian(vector<Point2f>& points) {
    vector<float> x_vals, y_vals;
    for (const auto& point : points) {
        x_vals.push_back(point.x);
        y_vals.push_back(point.y);
    }

    // Sort the x and y values separately
    sort(x_vals.begin(), x_vals.end());
    sort(y_vals.begin(), y_vals.end());

    // Get the median for x and y
    float median_x = x_vals[x_vals.size() / 2];
    float median_y = y_vals[y_vals.size() / 2];

    return Point2f(median_x, median_y);
}

// Function to calculate Euclidean distance between two points
float calculateDistance(const Point2f& p1, const Point2f& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Function to apply median filter with threshold-based filtering
// Point2f medianFilterWithThreshold(Point2f current_vp) {
//     // Add the new vanishing point to the buffer
//     if(current_vp.x>50 && current_vp.x<cols-50 && current_vp.y>=50 && current_vp.y<=rows-50){
//         return current_vp;


//     // If the buffer size exceeds the limit, remove the oldest point
//     if (vp_buffer.size() > buffer_size) {
//         vp_buffer.pop_front();
//     }

//     // Calculate the median of the points in the buffer
//     vector<Point2f> vp_list(vp_buffer.begin(), vp_buffer.end());
//     return calculateMedian(vp_list);
//     }
//     else{
//         return average_vp;
//     }
// }

class KalmanFilterVP {
public:
    Point2f v_est;    // Estimated vanishing point
    Mat p_est;        // Covariance matrix
    float q;          // Process noise (variance of prediction)
    float r;          // Measurement noise (variance of observation)

    KalmanFilterVP(const Point2f& initial_vp, float process_noise = 0.1f, float measurement_noise = 10.0f) {
        // Initial state estimate is the initial vanishing point
        v_est = initial_vp;
        // Covariance matrix initialized to identity (uncertainty in x and y)
        p_est = Mat::eye(2, 2, CV_32F);
        q = process_noise;
        r = measurement_noise;
    }

    void reset(float process_noise, float measurement_noise){
        q = process_noise;
        r = measurement_noise;
    }

    // Update Kalman filter with the current observed vanishing point
    Point2f update(const Point2f& v_obs) {
        // Prediction step (assume a constant model)
        Mat F = Mat::eye(2, 2, CV_32F);  // Transition matrix (identity, since no velocity is used)
        Mat Q = q * Mat::eye(2, 2, CV_32F);  // Process noise covariance
        Mat R = r * Mat::eye(2, 2, CV_32F);  // Measurement noise covariance

        // Prediction of the next position
        Mat v_pred = Mat(v_est);  // Prediction based on the previous state
        Mat p_pred = F * p_est * F.t() + Q;  // Update covariance

        // Kalman gain calculation
        Mat H = Mat::eye(2, 2, CV_32F);  // Measurement matrix (identity)
        Mat S = H * p_pred * H.t() + R;  // Innovation covariance
        Mat K = p_pred * H.t() * S.inv();  // Kalman gain

        // Measurement residual (innovation)
        Mat v_obs_mat = Mat(v_obs);  // Current observed vanishing point
        Mat y = v_obs_mat - H * v_pred;

        // Update estimate
        Mat v_new = v_pred + K * y;
        p_est = (Mat::eye(2, 2, CV_32F) - K * H) * p_pred;

        // Store the updated estimate
        v_est = Point2f(v_new.at<float>(0), v_new.at<float>(1));
        return v_est;
    }
};

// vector<Vec6f> FilterLines(const vector<Vec6f>& filter_lines){
    
// }

// Draw answer on the image
void DrawAns(Mat& img, Point2f crossPoint2f, Point2f theta) {
    // Example drawing
    circle(img, Point2f(crossPoint2f.x, crossPoint2f.y), 10, Scalar(255, 102, 255), -1);
}

// Main program
int main() {
    string input_file = "/home/nayan/Downloads/input.mp4";
    VideoCapture cap(2);  // Replace with video path if necessary
    // vector<Point2f> qV, qVTemp;
    // vector<float> qThetaLeft, qThetaRight;

    // int frame_count = 0;

    Point2f initial_vp(center_x, center_y);

    // // Kalman filter instance for vanishing point smoothing
    KalmanFilterVP kalman_filter(initial_vp); // Initial value
    // bool initialized = false; 

    // deque<Point2f> vp_buffer;
    // int buffer_size = 10;
    // float threshold_distance = 200.0f;

    Point2f previous_vanishing_point = initial_vp;  // Initialize with the center of the frame
    int unchanged_frames = 0;

    Point2f vanishing_point;

    // int state_dim = 4;
    // int measurement_dim = 2;

    // EKF ekf(state_dim, measurement_dim);

    // vector<double> x0 = {0, 0, 0, 0}; // Initial state
    // vector<vector<double>> P0 = {
    //     {1, 0, 0, 0},
    //     {0, 1, 0, 0},
    //     {0, 0, 1, 0},
    //     {0, 0, 0, 1}
    // };
    // ekf.setInitialEstimate(x0, P0);

    // // Set process and measurement noise
    // vector<vector<double>> Q = {
    //     {0.1, 0, 0, 0},
    //     {0, 0.1, 0, 0},
    //     {0, 0, 0.1, 0},
    //     {0, 0, 0, 0.1}
    // };
    // vector<vector<double>> R = {
    //     {0.5, 0},
    //     {0, 0.5}
    // };
    // ekf.setProcessNoise(Q);
    // ekf.setMeasurementNoise(R);

    // // Define state transition matrix (A)
    // vector<vector<double>> A = {
    //     {1, 0, 1, 0},
    //     {0, 1, 0, 1},
    //     {0, 0, 1, 0},
    //     {0, 0, 0, 1}
    // };

    // vector<double> smoothed_vp;

    while (true) {
        Mat img;
        bool ret = cap.read(img);
        if (!ret) {
            cout << "Video Empty" << endl;
            break;
        }

        resize(img, img, Size(cols, rows));
        Mat showImage = img.clone();

        vector<Vec6f> lsdLines = LSD(showImage, img);
        if (!lsdLines.empty() && lsdLines.size() >= 3) {
            // vector<Vec6f> filter_lines = FilterLines(filter_lines);
            // Process cross points and filtering
            vanishing_point = CalcLineCrossPoint(showImage, lsdLines);  // Example point

            // ekf.predict(A);
            // ekf.update(vector<double>({vanishing_point.x, vanishing_point.y}), {{1, 0}, {0, 1}});
            
            // smoothed_vp = ekf.getState();

            // if (!initialized && frame_count > 5 ) {
            // kalman_filter.reset(0.05F, 50.0F);
            // cout<<"initialized"<<endl;
            // initialized = true;
            // threshold_distance = 80.0f;
            // }

        // vanishing_point = medianFilterWithThreshold(vanishing_point);

        // circle(showImage, vanishing_point, 5, Scalar(0, 255, 255), -1);

        // // cout<<previous_vanishing_point<<"- Prev "<<vanishing_point<<" - curr "<<endl;
        


        

        // Update previous vanishing point

        if(vanishing_point.x>=0 && vanishing_point.x<=cols && vanishing_point.y>=0 && vanishing_point.y<=rows){
            vanishing_point = kalman_filter.update(vanishing_point);
            circle(showImage, vanishing_point, 5, Scalar(0, 0, 255), -1);
        }
        // cout<<"Unchanged - "<<unchanged_frames<<" frame count "<<frame_count<<endl;
        // if (initialized && calculateDistance(vanishing_point, previous_vanishing_point) <= 1.5f) {
        //     unchanged_frames++;
        // } else {0
        //     unchanged_frames = 0;  // Reset counter if it changes
        // }

        // // If unchanged for 200 frames, reset the vanishing point
        // if (unchanged_frames >= 5 && initialized) {
        //     // cout<<"reset"<<endl;
        //     vanishing_point = Point2f(center_x, center_y);  // Reset to center
        //     unchanged_frames = 0;  // Reset the counter
        //     kalman_filter.reset(0.1F, 0.10F);
        //     vp_buffer.clear();
        //     initialized=false;
        //     threshold_distance=200.0f;
        //     frame_count = 0;
        // }

        // previous_vanishing_point = vanishing_point;
            // Point theta(30, 150);  // Example angles
            // DrawAns(showImage, crossPoint, theta);
        // line(showImage, Point2d(0 , vanishing_point.y), Point2d(cols, vanishing_point.y), Scalar(0, 255, 9), 2);
        
        }

        // frame_count++;

        line(showImage,  Point2d(center_x-80, center_y-50), Point2d(center_x-80, center_y+50), Scalar(255,0 ,0), 2);
        line(showImage,  Point2d(center_x-80, center_y-50), Point2d(center_x+80, center_y-50), Scalar(255,0 ,0), 2);
        line(showImage,  Point2d(center_x-80, center_y+50), Point2d(center_x+80, center_y+50), Scalar(255,0 ,0), 2);
        line(showImage,  Point2d(center_x+80, center_y-50), Point2d(center_x+80, center_y+50), Scalar(255,0 ,0), 2);


        imshow("Result", showImage);
        if (waitKey(1) == 'q') break;
    }

    return 0;
}
