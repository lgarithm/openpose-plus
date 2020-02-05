// https://github.com/tensorlayer/openpose-plus/blob/master/openpose_plus/models/models_hao28_experimental.py

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <stdtracer>
#include <ttl/contrib/opencv>
#include <ttl/cuda_tensor>
#include <ttl/nn/kernels/cuda>
#include <ttl/nn/layers>
#include <ttl/range>

DEFINE_TRACE_CONTEXTS;

#include <example_openpose_plus_hao28.hpp>
#include <opencv2/opencv.hpp>
#include <openpose-plus.h>
#include <vis.h>

namespace ttl::nn::ops
{
template <typename T, typename Op, typename... Ts>
T apply(const Op &op, const Ts &... args)
{
    auto y = T(op(args.shape()...));
    op(ttl::ref(y), ttl::view(args)...);
    return y;
}
}  // namespace ttl::nn::ops

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);

    const std::string home(std::getenv("HOME"));
    const auto prefix = home + "/var/models/openpose";

    // const auto data_dir = home + "/var/data/openpose";
    const std::string data_dir("./data/openpose");
    const auto filename =
        data_dir + "/examples/media/COCO_val2014_000000000192.jpg";

    openpose_plus_hao28 openpose(prefix);

    auto paf_runner =
        create_paf_processor(32, 48, openpose.h, openpose.w, 19, 19, 13);

    auto x = ttl::tensor<float, 3>(openpose.h, openpose.w, 3);

    // TODO: input images
    auto input = ttl::tensor<uint8_t, 3>(x.shape());
    {
        ttl::cv::imread_resize(filename.c_str(), ttl::ref(input));
        std::transform(input.data(), input.data_end(), x.data(),
                       [](uint8_t p) { return p / 255.0; });
    }

    int repeats = 5;
    for (auto i : ttl::range(repeats)) {
        printf("inference %d\n", i);
        ttl::tensor_view<float, 4> x_batch(x.data(), 1, openpose.h, openpose.w,
                                           3);
        auto [l_conf, l_paf] = openpose(x_batch);
        // TODO: run paf process
        auto conf = ttl::nn::ops::apply<ttl::tensor<float, 4>>(
            ttl::nn::ops::to_channels_first(), *l_conf);
        auto paf = ttl::nn::ops::apply<ttl::tensor<float, 4>>(
            ttl::nn::ops::to_channels_first(), *l_paf);

        auto human = (*paf_runner)(conf.data(), paf.data(), false);
        {
            auto resized_image = ttl::cv::as_cv_mat(input);
            for (auto h : human) {
                h.print();
                draw_human(resized_image, h);
            }
            cv::imwrite("a.png", resized_image);
        }
    }
    return 0;
}
