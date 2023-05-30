//
// Created by slimakanzer on 29.03.19.
//

#ifndef BENCHMARK_BENCHMARK_H
#define BENCHMARK_BENCHMARK_H

#include "helpers/cuda_helper.h"
#include "helpers/cudnn_helper.h"
#include "tensor.h"

enum benchmarkStatus {
    BENCHMARK_SUCCESS = 0,
    BENCHMARK_NOT_SUPPORTED = 1,
    BENCHMARK_ERROR = 2
};


struct benchmarkResult {
    double time;
    size_t workspace_size;
    benchmarkStatus status;
};

struct benchmarkRow {
    int w, h, c, n, k, s, r, pad_w, pad_h, stride_w, stride_h, out_w, out_h;
    cudnnTensorFormat_t
            inputTensorFormat,
            outputTensorFormat,
            filterFormat;
    benchmarkResult
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,

            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,

            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
};

template<typename T>
class Benchmark {
    cudnnHandle_t cudnn;
    Tensor<T> *inputTensor;
    Tensor<T> *outputTensor;
    Tensor<T> *kernelTensor;
    const float alpha = 1, beta = 0;

    TensorDescriptor *inputTensorDescriptor;
    TensorDescriptor *outputTensorDescriptor;
    FilterDescriptor *filterDescriptor;
    cudnnConvolutionDescriptor_t convolutionDescriptor_;
    curandGenerator_t curand_gen;

    size_t fwd_workspace_size(cudnnConvolutionFwdAlgo_t algo);

    benchmarkResult forward(cudnnConvolutionFwdAlgo_t algo, uint32_t num_repeats);

    void forward_algorythms(uint32_t num_repeats);

    void calculate_workspace_benchmark(uint32_t num_repeats);

    void create_cudnn();

    void create_curand_generator();

public:
    benchmarkRow *benchmark_row;

    Benchmark();

    void benchmark(benchmarkRow &benchmarkInput, uint32_t num_repeats);

    static void run(std::string file_name, std::string output_file_name, bool all_formats, uint32_t num_repeats, cudnnTensorFormat_t input_format, cudnnTensorFormat_t output_format, cudnnTensorFormat_t kernel_format);
};

#endif //BENCHMARK_BENCHMARK_H
