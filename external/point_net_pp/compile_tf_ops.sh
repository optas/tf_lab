#!/bin/bash

#set CUDA_DIR to cuda path you'd like to link with; should match the version used for tensorflow
# run ./compile_tf_ops with no arguments to compile all three libraries (tf_interpolate, tf_grouping, tf_sampling.)

# to compile specific library:
# ./compile_tf_ops [-g] LIB_NAME [LIB_NAME2 LIB_NAME3 ...]
# [-g]: whether to compile with gpu instructions or not.

# ./compile_tf_ops is equivalent to running (./compile_tf_ops -g tf_grouping tf_sampling && ./compile_tf_ops tf_interpolate)

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_DIR="/usr/local/cuda-8.0"

compile() {
   echo "compiling $1"
   
   g++ -std=c++11 -shared src/${1}.cpp -o lib/${1}_so.so -fPIC \
   -I${TF_INC} -I${TF_INC}/external/nsync/public -L${TF_LIB} -ltensorflow_framework \
   -I ${CUDA_DIR}/include -lcudart -L ${CUDA_DIR}/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
}

compile_with_gpu() {
   echo "compiling ${1} with gpu"

   ${CUDA_DIR}/bin/nvcc -std=c++11 src/${1}_g.cu -o lib/${1}_g.cu.o -c -O2 \
   -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

   g++ -std=c++11 -shared src/${1}.cpp bin/${1}_g.cu.o -o lib/${1}_so.so -fPIC \
   -I${TF_INC} -I${TF_INC}/external/nsync/public -L${TF_LIB} -ltensorflow_framework \
   -I ${CUDA_DIR}/include -lcudart -L ${CUDA_DIR}/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

}

compile_gpu=false
DEFAULT_NARGS=0

if [ $# -ne "$DEFAULT_NARGS" ]; then
   while getopts ":g" opt; do
      case $opt in 
         g) echo "Using gpu compilation"; compile_gpu=true; shift ;;
         \?) echo "Unknown option -$OPTARG"; exit 1;;
      esac
   done

   if ${compile_gpu} ; then
         for LIB in "$@" ; do
         compile_with_gpu "${LIB}"
      done
   else
      for LIB in "$@" ; do
         compile "${LIB}"
      done
   fi
else
   compile_with_gpu tf_grouping
   compile_with_gpu tf_sampling
   compile tf_interpolate
fi
