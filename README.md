# horizon-detector

1. Image Loading

The load_image function is used to load an image file selected by the user through a file dialog. The image is converted from BGR (OpenCV default) to RGB format for proper display in Matplotlib.

2. Line Detection

Blurring: The detect_lines function applies a Gaussian blur to the image using a 2D Gaussian kernel. Blurring helps reduce noise, which can improve edge detection.

Edge Detection: Canny edge detection is performed on the blurred image to identify the edges.

Hough Transform: The detected edges are then processed using the Hough Line Transform to find lines in the image. Only lines within specific angle ranges (0.4 < θ < 1.47 and 1.67 < θ < 2.74 radians) are considered, filtering out near-horizontal lines.

3. RANSAC for Vanishing Point Detection

The RANSAC function iteratively selects random pairs of lines and calculates their intersection points. For each intersection point, the number of "inlier" lines (lines close to this point) is counted.

The intersection point with the highest inlier count is chosen as the vanishing point. The algorithm stops early if the number of inliers exceeds a predefined ratio, improving efficiency.

4. Drawing Detected Lines

The draw_lines function draws the detected lines on the original image, providing a visual representation of the detected structures.

5. Marking the Vanishing Point

The draw_point function marks the detected vanishing point on the image with a green circle.

6. Drawing the Horizon Line

The draw_horizontal function draws a horizontal green line across the image at the y-coordinate of the vanishing point, representing the horizon line.

![test1](https://github.com/takshrana/horizon-detector/blob/main/test1.png)

![test2](https://github.com/takshrana/horizon-detector/blob/main/test2.png)
