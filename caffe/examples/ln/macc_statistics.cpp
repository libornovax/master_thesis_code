//
// Libor Novak
// 04/12/2017
//
// Various kinds of statistics on the accumulators - to find out if it will gives us some interesting extra
// information.
//

#include <caffe/caffe.hpp>
#include "caffe/util/bbtxt.hpp"

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


namespace {

    class Hist2D
    {
    public:

        Hist2D (double xmin, double xmax, double ymin, double ymax, int num_bins)
            : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax),
              xspread(xmax-xmin), yspread(ymax-ymin),
              hist(num_bins, num_bins, CV_64FC1, cv::Scalar(0)),
              counts(num_bins, num_bins, CV_64FC1, cv::Scalar(0))
        {
        }

        void addEntry (double x, double y, double weight=1.0)
        {
            int col = std::round((x - this->xmin) / this->xspread * hist.cols);
            int row = std::round((y - this->ymin) / this->yspread * hist.rows);

            if (col >= 0 && row >= 0 && col < hist.cols && row < hist.rows)
            {
                hist.at<double>(row, col) += weight;
                counts.at<double>(row, col) += 1.0;
            }
            else
            {
                std::cout << "Out of bounds: " << x << ", " << y << " => " << col << ", " << row << std::endl;
            }
        }

        cv::Mat normalized ()
        {
            cv::Mat hist_norm; this->hist.copyTo(hist_norm);
            double m; cv::minMaxLoc(hist_norm, 0, &m);

            hist_norm *= 1.0 / m;

            return hist_norm;
        }

        cv::Mat countNormalized ()
        {
            cv::Mat hist_norm;
            cv::divide(this->hist, this->counts, hist_norm);
            return hist_norm;
        }

        cv::Mat countNormalizedNormalized ()
        {
            cv::Mat hist_norm;
            cv::divide(this->hist, this->counts, hist_norm);

            double m; cv::minMaxLoc(hist_norm, 0, &m);
            hist_norm *= 1.0 / m;

            return hist_norm;
        }

        friend std::ostream& operator<< (std::ostream& os, const Hist2D &h);


        // --------------------------------------- PUBLIC MEMBERS ---------------------------------------- //
        double xmin, xmax, ymin, ymax;
        double xspread, yspread;
        cv::Mat hist;
        cv::Mat counts;
        int total;
    };


    std::ostream& operator<< (std::ostream& os, const Hist2D &h)
    {
        os << h.hist;
        return os;
    }


    std::vector<Hist2D> hist_wh_neg;
    std::vector<Hist2D> hist_tl_neg;
    std::vector<Hist2D> hist_br_neg;
    std::vector<Hist2D> hist_wh_pos;
    std::vector<Hist2D> hist_tl_pos;
    std::vector<Hist2D> hist_br_pos;

    std::vector<Hist2D> hist_wh_car_g_bb;
    std::vector<Hist2D> hist_wh_notcar_g_bb;

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


void histogramOfCoords (caffe::Blob<float> *output, int a, const std::vector<BB2D> &gt_bbs)
{
    // Build probabilistic target (ground truth) accumulator - we need it to determine positive and negative
    // pixels
    static std::vector<std::pair<double, double>> size_bounds;
    static std::vector<double> scales;
    if (size_bounds.size() == 0)
    {
        // WARNING! These are size bounds for "macc_0.3_r2_x2_to_x16"!!
        size_bounds.emplace_back(22.25, 55.5);
        size_bounds.emplace_back(44.5, 111.0);
        size_bounds.emplace_back(89.0, 222.0);
        size_bounds.emplace_back(178.0, 444.0);
        scales.push_back(2.0);
        scales.push_back(4.0);
        scales.push_back(8.0);
        scales.push_back(16.0);
    }
    cv::Mat acc_gt_prob(output->shape(2), output->shape(3), CV_32FC1, cv::Scalar(0.0f));
    for (const BB2D &gt_bb: gt_bbs)
    {
        double size = std::max(gt_bb.width(), gt_bb.height());
        if (size > size_bounds[a].first && size < size_bounds[a].second)
        {
            // This ground truth should be detected by this accumulator
            cv::Point2d co = gt_bb.center();
            cv::circle(acc_gt_prob, cv::Point(co.x/scales[a], co.y/scales[a]), 3, cv::Scalar(1.0f), -1);
        }
    }

//    cv::imshow("gt acc " + std::to_string(a), acc_gt_prob);


    float *data_output = output->mutable_cpu_data();

    // 2D bounding box
    cv::Mat acc_prob(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 0));
    cv::Mat acc_xmin(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 1));
    cv::Mat acc_ymin(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 2));
    cv::Mat acc_xmax(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 3));
    cv::Mat acc_ymax(output->shape(2), output->shape(3), CV_32FC1, data_output+output->offset(0, 4));

//    cv::imshow("detected acc " + std::to_string(a), acc_prob);

    // Extract detected boxes - only extract local maxima from 3x3 neighborhood
    for (int i = 0; i < acc_prob.rows; ++i)
    {
        for (int j = 0; j < acc_prob.cols; ++j)
        {
            float label = acc_gt_prob.at<float>(i, j);

            double w = acc_xmax.at<float>(i, j) - acc_xmin.at<float>(i, j);
            double h = acc_ymax.at<float>(i, j) - acc_ymin.at<float>(i, j);

            if (label > 0.0f)
            {
                // This is a positive pixel
                hist_wh_pos[a].addEntry(w, h);
                hist_tl_pos[a].addEntry(acc_xmin.at<float>(i, j), acc_ymin.at<float>(i, j));
                hist_br_pos[a].addEntry(acc_xmax.at<float>(i, j), acc_ymax.at<float>(i, j));
                hist_wh_car_g_bb[a].addEntry(w, h, 1.0);
                hist_wh_notcar_g_bb[a].addEntry(w, h, 0.0);
            }
            else
            {
                // This is a background pixel
                hist_wh_neg[a].addEntry(w, h);
                hist_tl_neg[a].addEntry(acc_xmin.at<float>(i, j), acc_ymin.at<float>(i, j));
                hist_br_neg[a].addEntry(acc_xmax.at<float>(i, j), acc_ymax.at<float>(i, j));
                hist_wh_car_g_bb[a].addEntry(w, h, 0.0);
                hist_wh_notcar_g_bb[a].addEntry(w, h, 1.0);
            }
        }
    }
}


void computeStatistics (const std::string &path_image, const std::shared_ptr<caffe::Net<float>> &net,
                        const std::map<std::string, std::vector<BB2D>> &gt_bbs_list)
{
    caffe::Blob<float>* input_layer  = net->input_blobs()[0];

    std::vector<cv::Mat> input_channels;

    // Read the image
    cv::Mat image = cv::imread(path_image, CV_LOAD_IMAGE_COLOR);
    // Convert to zero mean and unit variance
    cv::Mat imagef; image.convertTo(imagef, CV_32FC3);
    imagef -= cv::Scalar(128.0f, 128.0f, 128.0f);
    imagef *= 1.0f/128.0f;

    // Ground truth bounding boxes
    std::vector<BB2D> gt_bbs;
    auto gt_bbsi = gt_bbs_list.find(path_image);
    if (gt_bbsi == gt_bbs_list.end())
    {
        LOG(WARNING) << "No ground truth for image '" << path_image << "'";
    }
    else
    {
        gt_bbs = (*gt_bbsi).second;
    }


    // Reshape the network
    input_layer->Reshape(1, input_layer->shape(1), imagef.rows, imagef.cols);
    net->Reshape();

    // Prepare the cv::Mats for input
    wrapInputLayer(input_layer, input_channels);
    // Copy the image to the input layer of the network
    cv::split(imagef, input_channels);

    net->Forward();

    // For each accumulator
    for (int a = 0; a < net->output_blobs().size(); ++a)
    {
        histogramOfCoords(net->output_blobs()[a], a, gt_bbs);
    }

//    cv::imshow("image", image);
//    cv::waitKey(0);
}


void runStatisticsComputation (const std::string &path_prototxt, const std::string &path_caffemodel,
                               const std::string &path_image_list, const std::string &path_gt_bbtxt,
                               const std::string &path_out)
{
#ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

    // Create network and load trained weights from caffemodel file
    auto net = std::make_shared<caffe::Net<float>>(path_prototxt, caffe::TEST);
    net->CopyTrainedLayersFrom(path_caffemodel);

    caffe::Blob<float>* input_layer  = net->input_blobs()[0];
    caffe::Blob<float>* output_layer = net->output_blobs()[0];

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(input_layer->shape(1), 3) << "Input layer must have 3 channels.";
    CHECK_EQ(output_layer->shape(1), 5) << "Unsupported network, only 5 channels!";

    std::ifstream infile(path_image_list.c_str());
    CHECK(infile) << "Unable to open image list TXT file '" << path_image_list << "'!";
    std::string line;

    // Load ground truth
    std::map<std::string, std::vector<BB2D>> gt_bbs_list = readBBTXTFile(path_gt_bbtxt);


    for (int i = 0; i < net->output_blobs().size(); ++i)
    {
        hist_wh_neg.emplace_back(0, 2, 0, 2, 200);
        hist_tl_neg.emplace_back(-1, 1, -1, 1, 200);
        hist_br_neg.emplace_back(0, 2, 0, 2, 200);
        hist_wh_pos.emplace_back(0, 2, 0, 2, 200);
        hist_tl_pos.emplace_back(-1, 1, -1, 1, 200);
        hist_br_pos.emplace_back(0, 2, 0, 2, 200);
        hist_wh_car_g_bb.emplace_back(0, 2, 0, 2, 200);
        hist_wh_notcar_g_bb.emplace_back(0, 2, 0, 2, 200);
    }


    // -- RUN THE DETECTOR ON EACH IMAGE -- //
    while (std::getline(infile, line))
    {
        LOG(INFO) << line;
        CHECK(boost::filesystem::exists(line)) << "Image '" << line << "' not found!";

        // Detect bbs on the image
        computeStatistics(line, net, gt_bbs_list);
    }


    for (int i = 0; i < net->output_blobs().size(); ++i)
    {
//        {
//            std::vector<cv::Mat> chs_tl;
//            chs_tl.push_back(cv::Mat::zeros(hist_tl_pos[i].hist.size(), CV_64FC1));
//            chs_tl.push_back(hist_tl_pos[i].normalized());
//            chs_tl.push_back(hist_tl_neg[i].normalized());
//            cv::line(chs_tl[0], cv::Point(chs_tl[0].cols/2,0), cv::Point(chs_tl[0].cols/2, chs_tl[0].rows), cv::Scalar(0.5), 2);
//            cv::line(chs_tl[0], cv::Point(0,chs_tl[0].rows/2), cv::Point(chs_tl[0].cols, chs_tl[0].rows/2), cv::Scalar(0.5), 2);
//            cv::Mat comb_tl; cv::merge(chs_tl, comb_tl);
//            cv::imshow("Normalized tl histogram " + std::to_string(i), comb_tl);
//            cv::imwrite(path_out + "/hist_tl_" + net->blob_names()[net->output_blob_indices()[i]] + ".png", comb_tl*255);
//        }

//        {
//            std::vector<cv::Mat> chs_br;
//            chs_br.push_back(cv::Mat::zeros(hist_br_pos[i].hist.size(), CV_64FC1));
//            chs_br.push_back(hist_br_pos[i].normalized());
//            chs_br.push_back(hist_br_neg[i].normalized());
//            cv::line(chs_br[0], cv::Point(chs_br[0].cols/2,0), cv::Point(chs_br[0].cols/2, chs_br[0].rows), cv::Scalar(0.5));
//            cv::line(chs_br[0], cv::Point(0,chs_br[0].rows/2), cv::Point(chs_br[0].cols, chs_br[0].rows/2), cv::Scalar(0.5));
//            cv::Mat comb_br; cv::merge(chs_br, comb_br);
//            cv::imshow("Normalized br histogram " + std::to_string(i), comb_br);
//            cv::imwrite(path_out + "/hist_br_" + net->blob_names()[net->output_blob_indices()[i]] + ".png", comb_br*255);
//        }

        {
            // P(BB|CAR_GT) WxH
            std::vector<cv::Mat> chs;
            chs.push_back(cv::Mat::zeros(hist_wh_pos[i].hist.size(), CV_64FC1));
            chs.push_back(hist_wh_pos[i].normalized());
            chs.push_back(hist_wh_neg[i].normalized());
            cv::line(chs[0], cv::Point(chs[0].cols/2,0), cv::Point(chs[0].cols/2, chs[0].rows), cv::Scalar(0.5));
            cv::line(chs[0], cv::Point(0,chs[0].rows/2), cv::Point(chs[0].cols, chs[0].rows/2), cv::Scalar(0.5));
            cv::line(chs[0], cv::Point(0,0), cv::Point(chs[0].cols, chs[0].rows), cv::Scalar(0.5));
            cv::Mat comb_wh; cv::merge(chs, comb_wh);
//            cv::imshow("P(BB|CAR_GT) WxH " + std::to_string(i), comb_wh);
            cv::imwrite(path_out + "/hist_wh_bb_g_car_notcar_" + net->blob_names()[net->output_blob_indices()[i]] + ".png", comb_wh*255);
        }

        {
            // P(CAR_GT|BB) WxH
            std::vector<cv::Mat> chs;
            chs.push_back(cv::Mat::zeros(hist_wh_car_g_bb[i].hist.size(), CV_64FC1));
            chs.push_back(hist_wh_car_g_bb[i].countNormalized());
            chs.push_back(cv::Mat::zeros(hist_wh_car_g_bb[i].hist.size(), CV_64FC1));
            cv::line(chs[0], cv::Point(chs[0].cols/2,0), cv::Point(chs[0].cols/2, chs[0].rows), cv::Scalar(0.5));
            cv::line(chs[0], cv::Point(0,chs[0].rows/2), cv::Point(chs[0].cols, chs[0].rows/2), cv::Scalar(0.5));
            cv::line(chs[0], cv::Point(0,0), cv::Point(chs[0].cols, chs[0].rows), cv::Scalar(0.5));
            cv::Mat comb_cf; cv::merge(chs, comb_cf);
//            cv::imshow("P(CAR_GT|BB) WxH " + std::to_string(i), comb_cf);
            cv::imwrite(path_out + "/hist_wh_car_g_bb_" + net->blob_names()[net->output_blob_indices()[i]] + ".png", comb_cf*255);
        }

        {
            // P(NOT_CAR_GT|BB) WxH
            std::vector<cv::Mat> chs;
            chs.push_back(cv::Mat::zeros(hist_wh_car_g_bb[i].hist.size(), CV_64FC1));
            chs.push_back(cv::Mat::zeros(hist_wh_car_g_bb[i].hist.size(), CV_64FC1));
            chs.push_back(hist_wh_notcar_g_bb[i].countNormalized());
            cv::line(chs[0], cv::Point(chs[0].cols/2,0), cv::Point(chs[0].cols/2, chs[0].rows), cv::Scalar(0.5));
            cv::line(chs[0], cv::Point(0,chs[0].rows/2), cv::Point(chs[0].cols, chs[0].rows/2), cv::Scalar(0.5));
            cv::line(chs[0], cv::Point(0,0), cv::Point(chs[0].cols, chs[0].rows), cv::Scalar(0.5));
            cv::Mat comb_cf; cv::merge(chs, comb_cf);
//            cv::imshow("P(NOT_CAR_GT|BB) WxH " + std::to_string(i), comb_cf);
            cv::imwrite(path_out + "/hist_wh_notcar_g_bb_" + net->blob_names()[net->output_blob_indices()[i]] + ".png", comb_cf*255);
        }

    }

//    cv::waitKey();
}



// -----------------------------------------------  MAIN  ------------------------------------------------ //

struct ProgramArguments
{
    std::string path_prototxt;
    std::string path_caffemodel;
    std::string path_image_list;
    std::string path_gt_bbtxt;
    std::string path_out;
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
            ("gt_bbtxt", po::value<std::string>(&pa.path_gt_bbtxt)->required(),
             "Path to a BBTXT file with ground truth annotation for the images in image list")
            ("path_out", po::value<std::string>(&pa.path_out)->required(),
             "Path to the output folder")
        ;

        po::positional_options_description positional;
        positional.add("prototxt", 1);
        positional.add("caffemodel", 1);
        positional.add("image_list", 1);
        positional.add("gt_bbtxt", 1);
        positional.add("path_out", 1);


        // Parse the input arguments
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: ./macc_statistics path/f.prototxt path/f.caffemodel path/image_list.txt path/out/folder\n";
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
        if (!boost::filesystem::exists(pa.path_gt_bbtxt))
        {
            std::cerr << "ERROR: File '" << pa.path_gt_bbtxt << "' does not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (not boost::filesystem::exists(pa.path_out))
        {
            std::cerr << "ERROR: Output folder '" << pa.path_out << "' does not exist!" << std::endl;
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


    runStatisticsComputation(pa.path_prototxt, pa.path_caffemodel, pa.path_image_list, pa.path_gt_bbtxt, pa.path_out);


    return EXIT_SUCCESS;
}


#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
