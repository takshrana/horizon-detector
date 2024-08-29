import numpy as np
import cv2
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename

adjustment_y = 600

def RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio):
    if len(lines) < 2:
        # Not enough lines to perform RANSAC, return a default vanishing point or None
        return (0, 0)
    
    inlier_count_ratio = 0.0
    vanishing_point = (0, 0)
    for iteration in range(ransac_iterations):
        line1, line2 = random.sample(lines, 2)
        intersection_point = find_intersection_point(line1, line2)
        if intersection_point is not None:
            inlier_count = sum(1 for line in lines if find_dist_to_line(intersection_point, line) < ransac_threshold)
            if inlier_count / len(lines) > inlier_count_ratio:
                inlier_count_ratio = inlier_count / len(lines)
                vanishing_point = intersection_point
            if inlier_count > len(lines) * ransac_ratio:
                break
    return vanishing_point

def find_dist_to_line(point, line):
    x0, y0 = point
    rho, theta = line[0]
    m = (-np.cos(theta)) / np.sin(theta)
    c = rho / np.sin(theta)
    x = (x0 + m * y0 - m * c) / (1 + m**2)
    y = (m * x0 + m**2 * y0 - m**2 * c) / (1 + m**2) + c
    dist = np.sqrt((x - x0)**2 + (y - y0)**2)
    return dist

def find_intersection_point(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    det_A = np.linalg.det(A)
    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        return int(np.round(x0)), int(np.round(y0))
    else:
        return None

def draw_point(image, point):
    image_copy = image.copy()
    cv2.circle(image_copy, (point[0], point[1] + adjustment_y), 5, (0, 255, 0), thickness=3)
    return image_copy

def merge_similar_lines(lines, distance_threshold, angle_threshold):
    merged_lines = []
    for line in lines:
        rho, theta = line[0]
        # print(rho, theta)
        similar_found = False
        for merged_line in merged_lines:
            merged_rho, merged_theta = merged_line[0]
            # print("merged" , abs(rho - merged_rho), abs(theta - merged_theta))
            if (abs(rho - merged_rho) < distance_threshold) and (abs(theta - merged_theta) < angle_threshold):
                # Average the similar lines
                merged_line[0][0] = (rho + merged_rho) / 2
                merged_line[0][1] = (theta + merged_theta) / 2
                similar_found = True
                break
        if not similar_found:
            merged_lines.append(line)
    return merged_lines

def ekf_predict(state, P, Q):
    F = np.array([[1, 0, 1, 0],  # State transition matrix
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    state = F @ state
    P = F @ P @ F.T + Q
    return state, P

def ekf_update(state, P, z, R):
    H = np.array([[1, 0, 0, 0],  # Measurement matrix
                  [0, 1, 0, 0]])
    y = z.reshape(2, 1) - H @ state
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    state = state + K @ y
    P = (np.eye(4) - K @ H) @ P
    return state, P

# EKF Predict function for horizontal line
def ekf_predict_horizontal(state_h, P_h, Q_h):
    F_h = np.array([[1, 1],  # State transition matrix
                    [0, 1]])
    state_h = F_h @ state_h
    P_h = F_h @ P_h @ F_h.T + Q_h
    return state_h, P_h

# EKF Update function for horizontal line
def ekf_update_horizontal(state_h, P_h, z_h, R_h):
    H_h = np.array([[1, 0]])  # Measurement matrix
    y_h = z_h.reshape(1, 1) - H_h @ state_h
    S_h = H_h @ P_h @ H_h.T + R_h
    K_h = P_h @ H_h.T @ np.linalg.inv(S_h)
    state_h = state_h + K_h @ y_h
    P_h = (np.eye(2) - K_h @ H_h) @ P_h
    return state_h, P_h

def compute_average_vanishing_point(vanishing_points):
    x_coords = [pt[0] for pt in vanishing_points]
    y_coords = [pt[1] for pt in vanishing_points]
    avg_x = np.mean(x_coords)
    avg_y = np.mean(y_coords)
    return int(avg_x), int(avg_y)

def is_within_threshold(vanishing_point, average_vp, threshold=50):
    return np.sqrt((vanishing_point[0] - average_vp[0])**2 + (vanishing_point[1] - average_vp[1])**2) <= threshold

def main(input_video_path, output_video_path=None):
    
    # EKF Initialization
    state = np.zeros((4, 1))  # [x, y, vx, vy]
    P = np.eye(4) * 1000  # Large initial uncertainty
    Q = np.eye(4) * 0.01  # Process noise
    R = np.eye(2) * 10  # Measurement noise

    state_h = np.zeros((2, 1))  # [y, vy]
    P_h = np.eye(2) * 1000  # Large initial uncertainty
    Q_h = np.eye(2) * 0.01  # Process noise
    R_h = np.eye(1) * 10  # Measurement noise

    vanishing_points = []
    
    video_capture = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gau_kernel = cv2.getGaussianKernel(100, 2)
        gau_kern2d = np.outer(gau_kernel, gau_kernel)
        gau_kern2d = gau_kern2d / gau_kern2d.sum()
        blur_image = cv2.filter2D(gray_frame, -1, gau_kern2d)

        edge_image = cv2.Canny(blur_image, 90, 110, apertureSize=3, L2gradient=True)
        edge_image_copy = edge_image[adjustment_y:1080].copy()

        lines = cv2.HoughLines(edge_image_copy, 1, np.pi / 180, 120)
        lines = [line for line in lines if (np.radians(30) < line[0][1] < np.radians(80)) or (np.radians(100) < line[0][1] < np.radians(150))]
        frame_copy = frame.copy()

        lines = merge_similar_lines(lines, 100, np.radians(5))

        # for line in lines:
        #     rho, theta = line[0]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1920 * (-b)) 
        #     y1 = int(y0 + 1920 * (a)) + adjustment_y
        #     x2 = int(x0 - 1920 * (-b))
        #     y2 = int(y0 - 1920 * (a)) + adjustment_y
        #     cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        vanishing_point = RANSAC(lines, 350, 20, 0.93)

        if vanishing_points:
            if len(vanishing_points) > 50:
                vanishing_points.pop(0)
            average_vp = compute_average_vanishing_point(vanishing_points)
            if is_within_threshold(vanishing_point, average_vp, 100):
                vanishing_points.append(vanishing_point)
            else:
                vanishing_point = average_vp  # Use the average vanishing point
        else:
            vanishing_points.append(vanishing_point)

        # EKF Prediction
        state, P = ekf_predict(state, P, Q)

        # EKF Update with the detected vanishing point
        state, P = ekf_update(state, P, np.array(vanishing_point), R)

        # Use EKF-smoothed vanishing point for drawing
        smoothed_vanishing_point = (int(state[0, 0]), int(state[1, 0]))

        # EKF Prediction for horizontal line
        state_h, P_h = ekf_predict_horizontal(state_h, P_h, Q_h)

        # EKF Update for horizontal line with the detected vanishing point's y-coordinate
        state_h, P_h = ekf_update_horizontal(state_h, P_h, np.array([smoothed_vanishing_point[1]]), R_h)

        smoothed_y = int(state_h[0, 0])

        # Draw the horizontal line
        cv2.line(frame_copy, (0, smoothed_y + adjustment_y), (1920, smoothed_y + adjustment_y), (255, 0, 0), 2)

        frame_copy = draw_point(frame_copy, smoothed_vanishing_point)

        cv2.imshow('Points Video', frame_copy)

        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        if output_video_path:
            video_writer.write(frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the Input File Path
    input_video_path = 'inputpath.mp4'
    # Specify the Output Video Path
    output_video_path = 'outputpath.mp4'
    
    main(input_video_path, output_video_path)