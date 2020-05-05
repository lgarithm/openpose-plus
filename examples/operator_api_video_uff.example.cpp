#include "utils.hpp"
#include <experimental/filesystem>
#include <gflags/gflags.h>
#include <openpose_plus/openpose_plus.hpp>

// Model flags
DEFINE_string(model_file, "../data/models/hao28-600000-256x384.uff",
              "Path to uff model.");
DEFINE_string(input_name, "image", "The input node name of your uff model file.");
DEFINE_string(output_name_list, "outputs/conf,outputs/paf", "The output node names(maybe more than one) of your uff model file.");

DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");

DEFINE_int32(max_batch_size, 8, "Max batch size for inference engine to execute.");

DEFINE_string(input_video, "../data/media/video.avi", "Video to be processed.");
DEFINE_string(output_video, "output_video.avi", "The name of output video.");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    namespace fs = std::experimental::filesystem;

    // * Input video.
    auto capture = cv::VideoCapture(FLAGS_input_video);

    // * Output video.
    auto writer = cv::VideoWriter(
            FLAGS_output_video,
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            capture.get(cv::CAP_PROP_FPS),
            cv::Size(FLAGS_input_width, FLAGS_input_height));

    // * Create TensorRT engine.
    namespace pp = poseplus;
    pp::dnn::tensorrt engine(
            pp::dnn::uff{FLAGS_model_file, FLAGS_input_name, split(FLAGS_output_name_list, ',')},
            { FLAGS_input_width, FLAGS_input_height },
            FLAGS_max_batch_size);

    // * post-processing: Using paf. // TODO: Add proposal networks processing.
    pp::parser::paf parser({FLAGS_input_width, FLAGS_input_height});

    using clk_t = std::chrono::high_resolution_clock;

    size_t frame_count = 0;
    auto beg = clk_t::now();
    {
        while(capture.isOpened())
        {
            std::vector<cv::Mat> batch;
            for(int i = 0; i < FLAGS_max_batch_size; ++i) {
                cv::Mat mat;
                capture >> mat;
                if(mat.empty())
                    break;
                batch.push_back(mat);
            }

            if(batch.empty())
                break;

            // * TensorRT Inference.
            auto feature_map_packets = engine.inference(batch);

            // * Paf.
            std::vector<std::vector<pp::human_t>> pose_vectors;
            pose_vectors.reserve(feature_map_packets.size());
            for (auto&& packet : feature_map_packets) {
                pose_vectors.push_back(parser.process(packet[0], packet[1]));
            }

            for (size_t i = 0; i < batch.size(); ++i) {
                cv::resize(batch[i], batch[i], { FLAGS_input_width, FLAGS_input_height });
                for (auto&& pose : pose_vectors[i])
                    pp::draw_human(batch[i], pose);
                writer << batch[i];
                ++frame_count;
            }
        }
    }
    std::cout << frame_count << " images got processed. FPS = "
              << 1000. * frame_count / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count()
              << '\n';
}