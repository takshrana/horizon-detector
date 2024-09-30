#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <cmath>

using namespace cv;
using namespace std;

int distanceCalculate(int x1, int y1, int x2, int y2)
{
	int x = x1 - x2; //calculating number to square in next step
	int y = y1 - y2;
	int dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);                  

	return dist;
}

Point2f find_intersection(const Point2f& p1, const Point2f& p2, const Point2f& p3, const Point2f& p4) {
    float A1 = p2.y - p1.y;
    float B1 = p1.x - p2.x;
    float C1 = A1 * p1.x + B1 * p1.y;
    
    float A2 = p4.y - p3.y;
    float B2 = p3.x - p4.x;
    float C2 = A2 * p3.x + B2 * p3.y;
    
    float det = A1 * B2 - A2 * B1;
    if (det == 0) {
        throw runtime_error("Lines do not intersect");
    }
    
    return Point2f(int((B2 * C1 - B1 * C2) / det), int((A1 * C2 - A2 * C1) / det));
}

float vector_norm(const Point2f& v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

Point2f point_diff(const pair<Point2f, Point2f> p) {
    return Point2f(p.second.x - p.first.x, p.second.y - p.first.y);
}

float angle_between_vectors(const pair<Point2f, Point2f>& v1, const pair<Point2f, Point2f>& v2) {
    Point2f diff_v1 = point_diff(v1);
    Point2f diff_v2 = point_diff(v2);
    
    float dot_product = diff_v1.x * diff_v2.x + diff_v1.y * diff_v2.y;
    float cos_theta = dot_product / (vector_norm(diff_v1) * vector_norm(diff_v2));
    cos_theta = min(max(cos_theta, -1.0f), 1.0f);
    return acos(cos_theta) * 180.0f / CV_PI;
}

Point2f angle_ransac(const vector<pair<Point2f, Point2f>>& motion_vectors, int num_iterations = 45, float max_inlier_angle = 45.0f) {
    Point2f best_vp;
    float best_vp_score = -numeric_limits<float>::infinity();
    
    for (int i = 0; i < num_iterations; ++i) {
        if (motion_vectors.size() < 2) {
            return Point2f();
        }

        int idx1 = rand() % motion_vectors.size();
        int idx2 = rand() % motion_vectors.size();

        while(idx2 == idx1) {
            idx2 = rand() % motion_vectors.size();
        }

        Point2f vp;
        try {
            vp = find_intersection(motion_vectors[idx1].first, motion_vectors[idx1].second, 
                                   motion_vectors[idx2].first, motion_vectors[idx2].second);
        } catch (const runtime_error&) {
            continue;
        }

        float vp_score = 0.0f;
        for (const auto& v : motion_vectors) {
            Point2f base = v.second;
            pair<Point2f, Point2f> u = {vp, base};

            // cout<<u.first<<" "<<u.second<<endl;            
            float theta = angle_between_vectors(v, u);
            float score = (theta < max_inlier_angle) ? exp(-abs(theta)) : 0;
            vp_score += score;
        }

        if (vp_score > best_vp_score) {
            best_vp_score = vp_score;
            best_vp = vp;
        }
    }

    return best_vp;
}

vector<pair<Point2f, Point2f>> filter_motion_vectors(vector<pair<Point2f, Point2f>> motion_vectors){
    pair<int, int> center = {1280/2, 720/2};
    vector<pair<Point2f, Point2f>> filtered_motion_vectors;
    for (auto vector: motion_vectors){
        int dist_old = distanceCalculate(vector.first.x, vector.first.y, center.first, center.second);
        int dist_new = distanceCalculate(vector.second.x, vector.second.y, center.first, center.second);
    
        double slope =  (vector.second.y - vector.first.y)/(vector.second.x - vector.first.x);
        double theta = abs(atan(slope) * 180/3.1415);

        if (dist_old < dist_new && (theta > 10 && theta < 170)){
            filtered_motion_vectors.push_back(vector);
        }

        
    }
    
    return filtered_motion_vectors;
}

int main() {
    string input_video_path = "/home/nayan/Taksh/Homography/moving_camera/videos/Camera/Kent Camera/pol-6503-lat-23.057601666666663lon-72.54946dt-05-09-24ti-16-45-44.mp4";
    VideoCapture cap(input_video_path);
    
    if (!cap.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    Mat first_frame;
    
    cap.read(first_frame);
    resize(first_frame, first_frame, Size(1280, 720));
    Mat gray_first;
    cvtColor(first_frame, gray_first, COLOR_BGR2GRAY);

    vector<Point2f> p0, curr_p0;
    goodFeaturesToTrack(gray_first, p0, 500, 0.01, 10);
    curr_p0 = p0;
    
    Mat gray_prev = gray_first.clone();
    Mat frame;
    int frame_count = 1;

    cout<<gray_first.type()<<" Row "<<gray_first.rows<<"Col : "  << gray_first.cols <<endl;
    cout<<first_frame.type()<<" Row "<<first_frame.rows<<"Col : "  << first_frame.cols <<endl;

    double start = cv::getTickCount();
    double elapsedTime=0;

    while (true) {
        double processTimeStart = cv::getTickCount();

        bool ret = cap.read(frame);
        if (!ret) break;

        resize(frame, frame, Size(1280, 720));
        Mat gray_frame;
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

        vector<Point2f> p1;
        vector<uchar> status;
        vector<float> err;

        calcOpticalFlowPyrLK(gray_prev, gray_frame, curr_p0, p1, status, err);

                
        vector<pair<Point2f, Point2f>> motion_vectors;
        vector<Point2f> tracked_p0 ;
        vector<Point2f> tracked_curr_p0 ;
        vector<Point2f> tracked_p1 ;
        for (size_t i = 0; i < p1.size(); ++i) {
            if (status[i]) {
                // cout<<status[i]<<' ';
                // int distance = sqrt(pow(p1[i].x - curr_p0[i].x, 2) + pow(p1[i].y -  curr_p0[i].y, 2));
                int distance = distanceCalculate(curr_p0[i].x, curr_p0[i].y, p1[i].x, p1[i].y);
                if(distance > 2){
                    tracked_p0.push_back(p0[i]);
                    tracked_p1.push_back(p1[i]);
                    tracked_curr_p0.push_back(curr_p0[i]);
                    motion_vectors.push_back({p0[i], p1[i]});
                }
            }
        }
        p0 = tracked_p0;
        p1 = tracked_p1;
        curr_p0 = tracked_curr_p0;

        // Point2f vanishing_point = angle_ransac(motion_vectors);
        // if (vanishing_point != Point2f()) {
        //     circle(frame, vanishing_point, 7, Scalar(0, 0, 255), -1);
        // }
        // cout<<p1.size();
        
        bool detection = true;
        // cout << motion_vectors.size() << endl;
        if (motion_vectors.size() < 400){
            detection = false;
            vector<Point2f> new_p0;
            goodFeaturesToTrack(gray_frame, new_p0, 500, 0.01, 10);
            p0.insert(p0.end(), new_p0.begin(), new_p0.end());
            p1.insert(p1.end(), new_p0.begin(), new_p0.end());
        }

        gray_prev = gray_frame.clone();
        curr_p0 = p1;

        if (detection == false){
            // cout<<"skipping detection \n";
            continue;
        }

        // cout<<"Before Filtering "<<motion_vectors.size()<<endl;
        motion_vectors = filter_motion_vectors(motion_vectors);
        // cout<<"After Filtering "<<motion_vectors.size()<<endl;

        Point2f vanishing_point = angle_ransac(motion_vectors);


        // for(auto point: motion_vectors){
        //     // circle(frame, point, 4,  Scalar(0, 255, 0), -1);
        //     line(frame, point.first,  point.second, Scalar(0, 255, 0), 2);
        // }

        line(frame, Point2d(0 , vanishing_point.y), Point2d(1280, vanishing_point.y), Scalar(0, 255, 9), 2);
        circle(frame, vanishing_point, 5, Scalar(0, 0, 255), -1);
        // cout<<vanishing_point;

        double currentTime = (cv::getTickCount() - start) / cv::getTickFrequency();
        elapsedTime = currentTime;

        // Calculate the actual FPS during playback
        double actual_fps = frame_count / elapsedTime;
        // string fps = actual_fps - '0';
        putText(frame, to_string(int(actual_fps)), Point(10,30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        
        imshow("Tracking", frame);
        frame_count++;

        if(vanishing_point.y < 200){
            cout<<"Tilt Your Phone Up"<<endl;
        }
        else if(vanishing_point.y > 520){
            cout<<"Tilt Your Phone Down"<<endl;
        }
        else{
            cout<<"Homography Active"<<endl;
        }
        
        // double processTimeEnd = cv::getTickCount();
        // double  processTime = (processTimeEnd - processTimeStart) / cv::getTickFrequency();
        // if(processTime < 200){
        //     waitKey(200 - processTime);
        // }


        if (waitKey(100) == 'q') break;

    }

    cap.release();
    // destroyAllWindows();
    return 0;
}
