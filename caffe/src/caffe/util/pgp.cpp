#include <caffe/caffe.hpp>

#include "caffe/util/pgp.hpp"


PGP::PGP (const cv::Mat &P_3x4, const cv::Mat &gp_1x4)
    : P_3x4(P_3x4.clone()),
      gp_1x4(gp_1x4.clone())
{
    // Create the inverse KR matrix
    this->P_3x4(cv::Rect(0, 0, 3, 3)).copyTo(this->KR_3x3_inv);
    this->KR_3x3_inv = this->KR_3x3_inv.inv();

    // And the image pose C
    this->C_3x1 = - this->KR_3x3_inv * this->P_3x4(cv::Rect(3, 0, 1, 3));
}


std::map<std::string, PGP> PGP::readPGPFile (const std::string &path_pgp)
{
    std::map<std::string, PGP> pgp_map;

    std::ifstream infile(path_pgp.c_str());
    CHECK(infile) << "Unable to open PGP file '" << path_pgp << "'!";

    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream ss(line);

        // A line of a PGP file looks like this:
        // filename p00 p01 p02 p03 p10 p11 p12 p13 p20 p21 p22 p23 a b c d

        std::string filename;
        ss >> filename;

        // Read the P matrix
        double tmp1[12];
        ss >> tmp1[0] >> tmp1[1] >> tmp1[2] >> tmp1[3] >> tmp1[4] >> tmp1[5] >> tmp1[6] >> tmp1[7] >> tmp1[8]
           >> tmp1[9] >> tmp1[10] >> tmp1[11];
        // Read the ground plane equation coefficients
        double tmp2[4];
        ss >> tmp2[0] >> tmp2[1] >> tmp2[2] >> tmp2[3];

        cv::Mat P_3x4(3, 4, CV_64FC1, tmp1);
        cv::Mat gp_1x4(1, 4, CV_64FC1, tmp2);

        pgp_map.insert(std::map<std::string, PGP>::value_type(filename, PGP(P_3x4, gp_1x4)));
    }

    infile.close();

    return pgp_map;
}


cv::Mat PGP::reconstructXGround (double u, double v) const
{
    return geometry::reconstructXInPlane(u, v, this->KR_3x3_inv, this->C_3x1, this->gp_1x4);
}


cv::Mat PGP::projectXtox (const cv::Mat &X_3xn) const
{
    return geometry::projectXtox(X_3xn, this->P_3x4);
}


cv::Mat PGP::reconstructAndFixBB3D (BB3D &bb3d) const
{
    // Reconstruct the corners, which lie in the ground plane
    cv::Mat FBL_3x1 = this->reconstructXGround(bb3d.fblx, bb3d.fbly);
    cv::Mat FBR_3x1 = this->reconstructXGround(bb3d.fbrx, bb3d.fbry);
    cv::Mat RBL_3x1 = this->reconstructXGround(bb3d.rblx, bb3d.rbly);
    cv::Mat RBR_3x1 = FBR_3x1 + (RBL_3x1-FBL_3x1);

    // Top of the 3D bounding box - reconstruct FTL and then just move all the other points
    // We do this by intersecting the ray to FTL with the front side plane of the bounding box - i.e. extract
    // the front side plane equation and then use it to reconstruct the FTL
    cv::Mat n_F_3x1 = FBL_3x1 - RBL_3x1;  // Normal vector of the front side
    double d_F = - n_F_3x1.dot(FBL_3x1);

    // Front plane
    cv::Mat fp_1x4(1, 4, CV_64FC1);
    fp_1x4.at<double>(0,0) = n_F_3x1.at<double>(0,0);
    fp_1x4.at<double>(0,1) = n_F_3x1.at<double>(1,0);
    fp_1x4.at<double>(0,2) = n_F_3x1.at<double>(2,0);
    fp_1x4.at<double>(0,3) = d_F;

    cv::Mat FTL_3x1 = geometry::reconstructXInPlane(bb3d.fblx, bb3d.ftly, this->KR_3x3_inv, this->C_3x1, fp_1x4);
    // Extract the vector pointing from the bottom to the top plane
    cv::Mat BT_3x1 = FTL_3x1 - FBL_3x1;

    {
        // The reconstruction we obtained so far is a parallelogram (not a rectangle) in the ground plane.
        // Now we are going to fix it by moving the corners along the diagonals - we need to make
        // the diagonals equal

        // Center of mass or the parallelogram
        cv::Mat CM_3x1 = (FBL_3x1 + RBR_3x1) / 2.0;
        // Half diagonals
        cv::Mat d1_3x1 = FBL_3x1 - CM_3x1; double d1_l = cv::norm(d1_3x1);
        cv::Mat d2_3x1 = FBR_3x1 - CM_3x1; double d2_l = cv::norm(d2_3x1);

        double delta = std::abs(d1_l - d2_l) / 2.0;

        cv::Mat d1_new_3x1; cv::Mat d2_new_3x1;
        if (d1_l > d2_l)
        {
            // First diagonal has to be shortened, second prolonged
            d1_new_3x1 = d1_3x1 * (1 - delta / d1_l);
            d2_new_3x1 = d2_3x1 * (1 + delta / d2_l);
        }
        else
        {
            // Second diagonal has to be shortened, first prolonged
            d1_new_3x1 = d1_3x1 * (1 + delta / d1_l);
            d2_new_3x1 = d2_3x1 * (1 - delta / d2_l);
        }

        // Update the corners of the ground plane rectangle
        FBL_3x1 = CM_3x1 + d1_new_3x1;
        FBR_3x1 = CM_3x1 + d2_new_3x1;
        RBL_3x1 = CM_3x1 - d2_new_3x1;
        RBR_3x1 = CM_3x1 - d1_new_3x1;
    }

    // Top rectangle
    FTL_3x1         = FBL_3x1 + BT_3x1;
    cv::Mat FTR_3x1 = FBR_3x1 + BT_3x1;
    cv::Mat RTL_3x1 = RBL_3x1 + BT_3x1;
    cv::Mat RTR_3x1 = RBR_3x1 + BT_3x1;

    // Combine everything to the output
    cv::Mat X_3x8(3, 8, CV_64FC1);
    FBL_3x1.copyTo(X_3x8(cv::Rect(0, 0, 1, 3)));
    FBR_3x1.copyTo(X_3x8(cv::Rect(1, 0, 1, 3)));
    RBR_3x1.copyTo(X_3x8(cv::Rect(2, 0, 1, 3)));
    RBL_3x1.copyTo(X_3x8(cv::Rect(3, 0, 1, 3)));
    FTL_3x1.copyTo(X_3x8(cv::Rect(4, 0, 1, 3)));
    FTR_3x1.copyTo(X_3x8(cv::Rect(5, 0, 1, 3)));
    RTR_3x1.copyTo(X_3x8(cv::Rect(6, 0, 1, 3)));
    RTL_3x1.copyTo(X_3x8(cv::Rect(7, 0, 1, 3)));

    // Update the coordinates in bb3d
    cv::Mat x_2x8 = this->projectXtox(X_3x8);
    bb3d.fblx = x_2x8.at<double>(0,0);
    bb3d.fbly = x_2x8.at<double>(1,0);
    bb3d.fbrx = x_2x8.at<double>(0,1);
    bb3d.fbry = x_2x8.at<double>(1,1);
    bb3d.rblx = x_2x8.at<double>(0,3);
    bb3d.rbly = x_2x8.at<double>(1,3);
    bb3d.ftly = x_2x8.at<double>(1,4);

    return X_3x8;
}
