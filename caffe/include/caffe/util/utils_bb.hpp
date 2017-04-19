//
// Libor Novak
// 04/18/2017
//
// Functions and structs handling bounding boxes
//

#ifndef UTILS_BB_H
#define UTILS_BB_H

#include <opencv2/core/core.hpp>


/**
 * @brief The BB2D struct
 * Struct representing a detected 2D bounding box
 */
struct BB2D
{
    BB2D () {}
    BB2D (const std::string &path_image, int label, double conf, double xmin, double ymin, double xmax,
          double ymax)
        : path_image(path_image),
          label(label),
          conf(conf),
          xmin(xmin),
          ymin(ymin),
          xmax(xmax),
          ymax(ymax)
    {
        // Check ordder of min and max
        if (this->xmin > this->xmax)
        {
            double temp = this->xmin;
            this->xmin = this->xmax;
            this->xmax = temp;
        }
        if (this->ymin > this->ymax)
        {
            double temp = this->ymin;
            this->ymin = this->ymax;
            this->ymax = temp;
        }
    }

    std::string path_image;
    int label;
    double conf;
    double xmin;
    double ymin;
    double xmax;
    double ymax;
};


/**
 * @brief The BB3D struct
 * Struct representing a detected 3D bounding box (its 6+1 coordinates needed for reconstruction in 3D)
 */
struct BB3D
{
    BB3D () {}
    BB3D (const std::string &path_image, int label, double conf, double fblx, double fbly, double fbrx,
          double fbry, double rblx, double rbly, double ftly)
        : path_image(path_image),
          label(label),
          conf(conf),
          fblx(fblx),
          fbly(fbly),
          fbrx(fbrx),
          fbry(fbry),
          rblx(rblx),
          rbly(rbly),
          ftly(ftly),
          xmin(0),
          ymin(0),
          xmax(0),
          ymax(0)
    {
    }

    std::string path_image;
    int label;
    double conf;
    double fblx;
    double fbly;
    double fbrx;
    double fbry;
    double rblx;
    double rbly;
    double ftly;

    double xmin;
    double ymin;
    double xmax;
    double ymax;
};


/**
 * @brief Compute intersection over union of 2 bounding boxes
 * Intersection over union in the image
 * @return Intersection over union
 */
template<typename BB>
double iou2d (const BB &bb1, const BB &bb2)
{
    double iw = std::max(0.0, std::min(bb1.xmax, bb2.xmax) - std::max(bb1.xmin, bb2.xmin));
    double ih = std::max(0.0, std::min(bb1.ymax, bb2.ymax) - std::max(bb1.ymin, bb2.ymin));

    double area1 = (bb1.xmax-bb1.xmin) * (bb1.ymax-bb1.ymin);
    double area2 = (bb2.xmax-bb2.xmin) * (bb2.ymax-bb2.ymin);
    double iarea = iw * ih;

    return iarea / (area1+area2 - iarea);
}


#endif // UTILS_BB_H

