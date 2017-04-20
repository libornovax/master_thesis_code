//
// Libor Novak
// 03/21/2017
//
// Tests object detection of 3D bounding boxes of a single scale detector on an image pyramid
//
// Optionaly it can apply non-maxima suppression (merging) on bounding boxes.
//

#include <caffe/caffe.hpp>
#include "caffe/util/benchmark.hpp"
#include "caffe/util/pgp.hpp"

// This code only works with OpenCV!
#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;


// Maximum 2D intersection over union of two boxes that will be both kept after NMS
#define IOU_2D_THRESHOLD 0.5


namespace {

    /**
     * @brief Checks if the Z coordinate of all points in X_3x8 is in front of C_3x1
     * @param X_3x8
     * @param C_3x1
     * @return True if z of all points in larger than z of C
     */
    bool checkZ (const cv::Mat &X_3x8, const cv::Mat &C_3x1)
    {
        for (int p = 0; p < 8; ++p)
        {
            if (X_3x8.at<double>(2, p) < C_3x1.at<double>(2,0))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * @brief Check if the bottom trapezoid has some reasonable size (area)
     * @param X_3x8 Coordinates of 3D bounding box corners ordered FBL FBR RBR RBL FTL FTR RTR RTL
     * @return True if it has good size
     */
    bool checkBBSize (const cv::Mat &X_3x8)
    {
        // We will compute the area of the bottom trapezoid of the bounding box - we need to find the
        // "height" h of the trapezoid in order to use the A = (a+b)*h / 2 equation for computing its area

        // Get the plane defined by the RBL-FBL normal vector and FBL and intersect it with the RBR-FBR line
        // The coordinates are ordered FBL FBR RBR RBL FTL FTR RTR RTL
        cv::Mat n_3x1 = X_3x8(cv::Rect(0, 0, 1, 3)) - X_3x8(cv::Rect(3, 0, 1, 3));  // RBL - FBL
        double d = -n_3x1.dot(X_3x8(cv::Rect(3, 0, 1, 3)));  // d from ax+by+cz+d=0

        // Intersect the plane with RBR-FBR line (it has the same direction as RBL-FBL)
        double lambda = -(n_3x1.dot(X_3x8(cv::Rect(1, 0, 1, 3))) + d) / n_3x1.dot(n_3x1);
        cv::Mat ip_3x1 = X_3x8(cv::Rect(1, 0, 1, 3)) + lambda*n_3x1;  // Intersection point

        // Now compute the distance of the intersected point and FBL to get the h
        cv::Mat temp = X_3x8(cv::Rect(3, 0, 1, 3)) - ip_3x1;
        double h = std::sqrt(temp.dot(temp));
        // Distance of FBL and RBL - another side of the trapezoid
        double a = std::sqrt(n_3x1.dot(n_3x1));

        // Check the area - if it is too small or too big throw it away
        double area = (a*a*h) / 2.0;
        if (area < 1.5 || area > 7500) return false;
        return true;
    }

}


/**
 * @brief Wraps the input layer into a vector of cv::Mat so we could assign data to it more easily
 * @param input_layer Pointer to the net input layer blob
 * @param input_channels Vector of cv::Mat, which will be assigned
 */
void wrapInputLayer (caffe::Blob<float>* input_layer, std::vector<cv::Mat> &out_input_channels)
{
    out_input_channels.clear();

    int height = input_layer->shape(2);
    int width  = input_layer->shape(3);

    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->shape(1); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        out_input_channels.push_back(channel);
        input_data += width * height;
    }
}


std::vector<BB3D> extract3DBoundingBoxes (caffe::Blob<float> *output, const std::string &path_image,
                                          double scale, const PGP *pgp_p, bool size_filter)
{
    std::vector<BB3D> bounding_boxes;

    float *data_output = output->mutable_cpu_data();

    // 3D bounding box
    cv::Mat acc_prob(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 0));
    cv::Mat acc_fblx(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 1));
    cv::Mat acc_fbly(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 2));
    cv::Mat acc_fbrx(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 3));
    cv::Mat acc_fbry(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 4));
    cv::Mat acc_rblx(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 5));
    cv::Mat acc_rbly(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 6));
    cv::Mat acc_ftly(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 7));

    // Extract detected 3D boxes - only extract local maxima from 3x3 neighborhood
    for (int i = 0; i < acc_prob.rows; ++i)
    {
        for (int j = 0; j < acc_prob.cols; ++j)
        {
            float conf = acc_prob.at<float>(i, j);
            if (conf >= 0.1)
            {
                // Check if it is a local maximum
                if (i > 0)
                {
                    if (j > 0 && acc_prob.at<float>(i-1, j-1) > conf) continue;
                    if (acc_prob.at<float>(i-1, j) > conf) continue;
                    if (j < acc_prob.cols-1 && acc_prob.at<float>(i-1, j+1) > conf) continue;
                }
                if (j > 0 && acc_prob.at<float>(i, j-1) > conf) continue;
                if (j < acc_prob.cols-1 && acc_prob.at<float>(i, j+1) > conf) continue;
                if (i < acc_prob.rows-1)
                {
                    if (j > 0 && acc_prob.at<float>(i+1, j-1) > conf) continue;
                    if (acc_prob.at<float>(i+1, j) > conf) continue;
                    if (j < acc_prob.cols-1 && acc_prob.at<float>(i+1, j+1) > conf) continue;
                }

                // Ok, this is a local maximum
                double fblx = acc_fblx.at<float>(i, j) / scale;
                double fbly = acc_fbly.at<float>(i, j) / scale;
                double fbrx = acc_fbrx.at<float>(i, j) / scale;
                double fbry = acc_fbry.at<float>(i, j) / scale;
                double rblx = acc_rblx.at<float>(i, j) / scale;
                double rbly = acc_rbly.at<float>(i, j) / scale;
                double ftly = acc_ftly.at<float>(i, j) / scale;


                if (pgp_p == NULL)
                {
                    // We do not have image projection matrices and ground planes - just output the detection
                    bounding_boxes.emplace_back(path_image, 1, conf, fblx, fbly, fbrx, fbry, rblx, rbly, ftly);
                }
                else
                {
                    // There are image projection matrices and ground planes - reconstruct the 3D bounding box
                    BB3D bb3d(path_image, 1, conf, fblx, fbly, fbrx, fbry, rblx, rbly, ftly);
                    cv::Mat X_3x8 = pgp_p->reconstructBB3D(bb3d);

                    // Now since the Z axis points forward, we can check if the reconstructed points are in
                    // front of the camera, if not then we can throw away this bounding box
                    if (!checkZ(X_3x8, pgp_p->C_3x1)) continue;

                    // Check if the bounding box is large enough - its bottom trapezoid should be of
                    // reasonable size
                    if (size_filter && !checkBBSize(X_3x8)) continue;

                    // Extract the 2D bounding box - project back the 3D one and find extremes
                    cv::Mat x_2x8 = pgp_p->projectXtox(X_3x8);
                    double xmin = DBL_MAX; double ymin = DBL_MAX; double xmax = DBL_MIN; double ymax = DBL_MIN;
                    for (int p = 0; p < 8; ++p)
                    {
                        const double u = x_2x8.at<double>(0, p);
                        const double v = x_2x8.at<double>(1, p);
                        if (u < xmin) xmin = u;
                        if (u > xmax) xmax = u;
                        if (v < ymin) ymin = v;
                        if (v > ymax) ymax = v;
                    }
                    // Set the 2D bounding box
                    bb3d.xmin = xmin;
                    bb3d.ymin = ymin;
                    bb3d.xmax = xmax;
                    bb3d.ymax = ymax;

                    bounding_boxes.push_back(bb3d);
                }
            }
        }
    }

    return bounding_boxes;
}


std::vector<BB3D> detectObjects (const std::string &path_image, const std::vector<double> &scales,
                                 const std::shared_ptr<caffe::Net<float>> &net,
                                 const std::map<std::string, PGP> &pgps, bool size_filter)
{
#ifdef MEASURE_TIME
    caffe::CPUTimer timer;
#endif
    std::vector<BB3D> bounding_boxes;

    caffe::Blob<float>* input_layer  = net->input_blobs()[0];

    std::vector<cv::Mat> input_channels;

#ifdef MEASURE_TIME
    timer.Start();
#endif
    // Load the image
    cv::Mat image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
    // Convert to zero mean and unit variance
    cv::Mat imagef; image.convertTo(imagef, CV_32FC3);
    imagef -= cv::Scalar(128.0f, 128.0f, 128.0f);
    imagef *= 1.0f/128.0f;
#ifdef MEASURE_TIME
    timer.Stop(); std::cout << "Time to to read image: " << timer.MilliSeconds() << " ms" << std::endl;
#endif

    // Get the image projection matrix and ground plane if we have them
    const PGP* pgp_p = NULL;
    if (pgps.size() > 0)
    {
        // There are image projection matrices and ground planes - find the one for this image
        auto pgpi = pgps.find(path_image);
        if (pgpi != pgps.end())
        {
            pgp_p = &((*pgpi).second);
        }
        else
        {
            std::cerr << "WARNING: PGP entry not found for image '" << path_image << "'" << std::endl;
        }
    }

#ifdef MEASURE_TIME
    timer.Start();
#endif
    // Build the image pyramid and run detection on each scale of the pyramid
    for (double s: scales)
    {
        cv::Mat imagef_scaled;
        cv::resize(imagef, imagef_scaled, cv::Size(), s, s);

        // Reshape the network
        input_layer->Reshape(1, input_layer->shape(1), imagef_scaled.rows, imagef_scaled.cols);
        net->Reshape();

        // Prepare the cv::Mats for input
        wrapInputLayer(input_layer, input_channels);
        // Copy the image to the input layer of the network
        cv::split(imagef_scaled, input_channels);

        net->Forward();

        for (caffe::Blob<float>* output: net->output_blobs())
        {
            std::vector<BB3D> new_bbs = extract3DBoundingBoxes(output, path_image, s, pgp_p, size_filter);
            bounding_boxes.insert(bounding_boxes.end(), new_bbs.begin(), new_bbs.end());
        }
    }

#ifdef MEASURE_TIME
    timer.Stop(); std::cout << "Time net + bb extraction: " << timer.MilliSeconds() << " ms" << std::endl;
#endif

    return bounding_boxes;
}


std::vector<BB3D> nonMaximaSuppression (std::vector<BB3D> &bbs)
{
    // For now we do NMS in the image! Not in 3D!
    //
    // This is standard non-maxima suppression, where we throw away boxes, which have a higher 2D intersection
    // over union then given threshold with some other box with a higher confidence

    // Sort by confidence in the descending order
    std::sort(bbs.begin(), bbs.end(), [](const BB3D &a, const BB3D &b) { return a.conf > b.conf; });

    std::vector<BB3D> bbs_out;
    std::vector<bool> active(bbs.size(), true);

    for (int i = 0; i < bbs.size(); ++i)
    {
        if (active[i])
        {
            active[i] = false;
            bbs_out.push_back(bbs[i]);

            // Check intersection with all remaining bounding boxes
            for (int j = i; j < bbs.size(); ++j)
            {
                if (active[j] && iou2d(bbs_out.back(), bbs[j]) > IOU_2D_THRESHOLD)
                {
                    active[j] = false;
                }
            }
        }
    }

    return bbs_out;
}


void writeBoundingBoxes (const std::vector<BB3D> &bbs, std::ofstream &fout)
{
    for (const BB3D &bb: bbs)
    {
        // The line of a BB3TXT file is:
        // filename label confidence xmin ymin xmax ymax fblx fbly fbrx fbry rblx rbly ftly
        fout << bb.path_image << " 1 " << bb.conf << " " << bb.xmin << " " << bb.ymin << " " << bb.xmax
             << " " << bb.ymax << " " << bb.fblx << " " << bb.fbly << " " << bb.fbrx << " " << bb.fbry
             << " " << bb.rblx << " " << bb.rbly << " " << bb.ftly << std::endl;
    }
}



void runPyramidDetection (const std::string &path_prototxt, const std::string &path_caffemodel,
                          const std::string &path_image_list, const std::string &path_out,
                          const std::string &path_pgp, bool size_filter)
{
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

#ifdef MEASURE_TIME
    caffe::CPUTimer timer;
#endif

    // Scaling factor is 1.5
//    const std::vector<double> scales = { 1.0, 0.66, 0.44, 0.29, 0.19 };
    const std::vector<double> scales = { 1.0 };

    // Create network and load trained weights from caffemodel file
    auto net = std::make_shared<caffe::Net<float>>(path_prototxt, caffe::TEST);
    net->CopyTrainedLayersFrom(path_caffemodel);

    caffe::Blob<float>* input_layer  = net->input_blobs()[0];
    caffe::Blob<float>* output_layer = net->output_blobs()[0];

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(input_layer->shape(1), 3) << "Input layer must have 3 channels.";
    CHECK_EQ(output_layer->shape(1), 8) << "Unsupported network, only 8 channels!";

    // Prepare the input channels
    std::vector<cv::Mat> input_channels;
    wrapInputLayer(input_layer, input_channels);

    std::ifstream infile(path_image_list.c_str());
    CHECK(infile) << "Unable to open image list TXT file '" << path_image_list << "'!";
    std::string line;

    // Load the P matrices and ground planes
    std::map<std::string, PGP> pgps;
    if (path_pgp != "") pgps = PGP::readPGPFile(path_pgp);

    std::ofstream fout; fout.open(path_out);
    CHECK(fout) << "Output file '" << path_out << "' could not have been created!";
    std::ofstream fout_nms;
    if (pgps.size() > 0) fout_nms.open(path_out.substr(0, path_out.size()-7) + "_nms.bb3txt");

    // -- RUN THE DETECTOR ON EACH IMAGE -- //
    while (std::getline(infile, line))
    {
        LOG(INFO) << line;
        CHECK(boost::filesystem::exists(line)) << "Image '" << line << "' not found!";

        // Detect bbs on the image
        std::vector<BB3D> bbs = detectObjects(line, scales, net, pgps, size_filter);

        // Save the bounding boxes before NMS to a BBTXT file
        writeBoundingBoxes(bbs, fout);

        // Only do NMS if we can reconstruct the 3D boxes
        if (pgps.size() > 0)
        {
#ifdef MEASURE_TIME
            timer.Start();
#endif
            // Non-maxima suppression
            bbs = nonMaximaSuppression(bbs);
#ifdef MEASURE_TIME
            timer.Stop(); std::cout << "Time to perform NMS: " << timer.MilliSeconds() << " ms" << std::endl;
#endif

            // Save the bounding boxes after NMS to a BBTXT file
            writeBoundingBoxes(bbs, fout_nms);
        }
    }

    fout.close();
    if (pgps.size() > 0) fout_nms.close();
}


// -----------------------------------------------  MAIN  ------------------------------------------------ //

struct ProgramArguments
{
    std::string path_prototxt;
    std::string path_caffemodel;
    std::string path_image_list;
    std::string path_out;
    std::string path_pgp;
    bool size_filter;
};


/**
 * @brief Parses arguments of the program
 */
void parseArguments (int argc, char** argv, ProgramArguments &pa)
{
    try {
        po::options_description desc("Arguments");
        desc.add_options()
            ("help", "Print help")
            ("prototxt", po::value<std::string>(&pa.path_prototxt)->required(),
             "Model file of the network (*.prototxt)")
            ("caffemodel", po::value<std::string>(&pa.path_caffemodel)->required(),
             "Weight file of the network (*.caffemodel)")
            ("image_list", po::value<std::string>(&pa.path_image_list)->required(),
             "Path to a TXT file with paths to the images to be tested")
            ("path_out", po::value<std::string>(&pa.path_out)->required(),
             "Path to the output BB3TXT file")
            ("pgp", po::value<std::string>(&pa.path_pgp)->default_value(""),
             "Path to a PGP file with calibration matrices and ground planes")
            ("size_filter", po::bool_switch(&pa.size_filter)->default_value(false),
             "Turns on filtering of all bounding boxes, which are too small")
        ;

        po::positional_options_description positional;
        positional.add("prototxt", 1);
        positional.add("caffemodel", 1);
        positional.add("image_list", 1);
        positional.add("path_out", 1);
        positional.add("pgp", 1);


        // Parse the input arguments
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: ./macc3d_pyramid_test path/f.prototxt path/f.caffemodel path/image_list.txt path/out.bb3txt (path/calib.pgp)\n";
            std::cout << desc;
            exit(EXIT_SUCCESS);
        }

        po::notify(vm);

        if (!boost::filesystem::exists(pa.path_prototxt))
        {
            std::cerr << "ERROR: File '" << pa.path_prototxt << "' does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!boost::filesystem::exists(pa.path_caffemodel))
        {
            std::cerr << "ERROR: File '" << pa.path_caffemodel << "' does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!boost::filesystem::exists(pa.path_image_list))
        {
            std::cerr << "ERROR: File '" << pa.path_image_list << "' does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (boost::filesystem::exists(pa.path_out))
        {
            std::cerr << "ERROR: File '" << pa.path_out << "' already exists!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (pa.path_pgp != "" && !boost::filesystem::exists(pa.path_pgp))
        {
            std::cerr << "ERROR: File '" << pa.path_pgp << "' does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << "\n";
        exit(EXIT_FAILURE);
    }
}


int main (int argc, char** argv)
{
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = ::google::INFO;
    ::google::InitGoogleLogging(argv[0]);

    ProgramArguments pa;
    parseArguments(argc, argv, pa);

std::cout << pa.size_filter << std::endl;
    runPyramidDetection(pa.path_prototxt, pa.path_caffemodel, pa.path_image_list, pa.path_out, pa.path_pgp, pa.size_filter);


    return EXIT_SUCCESS;
}


#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
