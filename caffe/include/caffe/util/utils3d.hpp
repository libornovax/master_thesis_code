//
// Libor Novak
// 04/18/2017
//
// Functions and structs for 3D point reconstruction and camera projection
//

#ifndef UTILS3D_H
#define UTILS3D_H

#include <opencv2/core/core.hpp>


namespace geometry {

    /**
     * @brief Project points X from the 3D world into the camera ginven by the P_3x4 matrix
     * @param X_3xn Matrix of coordinates of the points, each column is a point
     * @param P_3x4 Image projection matrix
     * @return 2x1 matrix of coordinates of the projected points in the image
     */
    cv::Mat projectXtox (const cv::Mat &X_3xn, const cv::Mat P_3x4)
    {
        // Create a 4xn matrix from the points
        cv::Mat X_4xn = cv::Mat::ones(4, X_3xn.cols, CV_64FC1);
        X_3xn.copyTo(X_4xn(cv::Rect(0, 0, X_3xn.cols, 3)));

        cv::Mat x_3xn = P_3x4 * X_4xn;
        cv::Mat tmp = cv::repeat(x_3xn.row(2), 2, 1);
        cv::Mat x_2xn = x_3xn(cv::Rect(0, 0, x_3xn.cols, 2)) / tmp;

        return x_2xn;
    }


    /**
     * @brief Reconstructs a 3D point from image coordinates (u,v), which lies in the plane given by p_1x4
     * @param u X coordinate in the image
     * @param v Y coordinate in the image
     * @param KR_3x3_inv Inverse KR matrix from the P=KR[I|-C] equation
     * @param p_1x4 Coefficients in the ax+by+cz+d=0 plane equation
     * @return 3x1 coordinates of the point in the 3D world
     */
    cv::Mat reconstructXInPlane (double u, double v, const cv::Mat &KR_3x3_inv, const cv::Mat &C_3x1, const cv::Mat &p_1x4)
    {
        // Homogenous coordinates of the point in the image
        cv::Mat x_3x1 = cv::Mat::ones(3, 1, CV_64FC1);
        x_3x1.at<double>(0,0) = u;
        x_3x1.at<double>(1,0) = v;
        // Compute the direction of the ray from the camera center
        cv::Mat X_d_3x1 = KR_3x3_inv * x_3x1;

        // Intersect the plane p_1x4 with the ray - find lambda from the X = C + lambda*X_d equation
        cv::Mat tmp1 = p_1x4(cv::Rect(0, 0, 3, 1))*C_3x1;
        cv::Mat tmp2 = p_1x4(cv::Rect(0, 0, 3, 1))*X_d_3x1;
        double lambda = - (tmp1.at<double>(0,0) + p_1x4.at<double>(0,3)) / tmp2.at<double>(0,0);

        return C_3x1 + lambda*X_d_3x1;
    }

}


#endif // UTILS3D_H

