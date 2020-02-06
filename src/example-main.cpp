// https://github.com/tensorlayer/openpose-plus/blob/master/openpose_plus/models/models_hao28_experimental.py

// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <stdtracer>
#include <ttl/contrib/opencv>
#include <ttl/cuda_tensor>
#include <ttl/nn/experimental/functional>
#include <ttl/nn/kernels/cuda>
#include <ttl/nn/layers>
#include <ttl/range>

DEFINE_TRACE_CONTEXTS;

#include <example_openpose_plus_hao28.hpp>
#include <opencv2/opencv.hpp>
#include <openpose-plus.h>
#include <vis.h>

template <typename T>
auto make_cpu_tensor_from(const T &t)
{
    using R = typename T::value_type;
    constexpr auto r = T::rank;
    ttl::tensor<R, r> x(t.shape());
    ttl::copy(ttl::ref(x), ttl::view(t));
    return x;
}

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
    {
        TRACE_SCOPE("inference all");
        int repeats = 5;
        for (auto i : ttl::range(repeats)) {
            printf("inference %d\n", i);
            ttl::tensor_view<float, 4> x_batch(x.data(), 1, openpose.h,
                                               openpose.w, 3);
            // auto [l_conf, l_paf] = openpose(x_batch);
            ttl::cuda_tensor<float, 4> x_batch_gpu(x_batch.shape());
            ttl::copy(ttl::ref(x_batch_gpu), x_batch);
            const auto [l_conf_gpu, l_paf_gpu] =
                openpose(ttl::view(x_batch_gpu));

            const auto l_conf = make_cpu_tensor_from(*l_conf_gpu);
            const auto l_paf = make_cpu_tensor_from(*l_paf_gpu);

            auto conf = ttl::nn::invoke(ttl::nn::ops::to_channels_first(),
                                        ttl::view(l_conf));
            auto paf = ttl::nn::invoke(ttl::nn::ops::to_channels_first(),
                                       ttl::view(l_paf));

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
    }
    return 0;
}
