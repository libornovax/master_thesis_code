//
// Libor Novak
// 04/18/2017
//
// Functions for processing the PGP file format
//

#ifndef PGP_H
#define PGP_H

#include <opencv2/core/core.hpp>
#include <map>
#include "utils3d.hpp"
#include "utils_bb.hpp"


/**
 * @brief The PGP class
 * Stores a 3x4 image projection matrix P and coefficients of the ground plane equation ax+by+cz+d=0. It
 * is to be read as one line from a PGP file.
 */
class PGP
{
public:

    PGP (const cv::Mat &P_3x4, const cv::Mat &gp_1x4);


    /**
     * @brief Reads a PGP file into a map indexed by image filenames
     * @param path_pgp Path to the PGP file to be read
     * @return
     */
    static std::map<std::string, PGP> readPGPFile (const std::string &path_pgp);


    /**
     * @brief Reconstructs a 3D point, which lies on the ground plane out of (u,v) image coordinates
     * @param u X image coordinate
     * @param v Y image coordinate
     * @return 3x1 matrix with the reconstructed 3D point
     */
    cv::Mat reconstructXGround (double u, double v) const;

    /**
     * @brief Projects 3D points into the camera defined by this PGP
     * @param X_3xn n 3D points in a matrix
     * @return 2xn matrix of projected points
     */
    cv::Mat projectXtox (const cv::Mat &X_3xn) const;

    /**
     * @brief Reconstructs the coordinates of the corners of the 3D bounding box defined by the given detection
     * @param bb3d Detected 3D bounding box (7 parameters)
     * @return 3x8 matrix of point coordinates in this order: FBL FBR RBR RBL FTL FTR RTR RTL
     */
    cv::Mat reconstructAndFixBB3D (BB3D &bb3d) const;


    // -----------------------------------------  PUBLIC MEMBERS  ----------------------------------------- //
    // Image projection matrix P = KR[I|-C]
    cv::Mat P_3x4;
    // Coefficients of the equation of the ground plane ax+by+cz+d=0
    cv::Mat gp_1x4;
    // Inverse of the KR matrix from the P matrix
    cv::Mat KR_3x3_inv;
    // Camera pose from the P matrix
    cv::Mat C_3x1;

};


#endif // PGP_H

