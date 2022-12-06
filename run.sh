export CUDA_PATH=/usr/local/cuda
export CUDNN_PATH=/usr/local/cuda

make
$CUDA_PATH/bin/nvcc -std=c++11 -I $CUDNN_PATH/include -L $CUDNN_PATH/lib64 src/benchmark.cu -o bin/benchmark -lcudnn -lcurand

./bin/benchmark conv_example.txt out_example.csv fp32 0 100 NHWC NHWC NHWC
./bin/benchmark conv_example.txt out_example.csv fp16 0 100 NHWC NHWC NHWC
./bin/benchmark conv_example.txt out_example.csv int8 0 100 NHWC NHWC NHWC
./bin/benchmark conv_example.txt out_example.csv int8x4 0 100 NCHW_VECT_C NCHW_VECT_C NCHW_VECT_C
./bin/benchmark conv_example.txt out_example.csv int8x32 0 100 NCHW_VECT_C NCHW_VECT_C NCHW_VECT_C