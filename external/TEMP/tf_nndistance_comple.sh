#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_DIR="/usr/local/cuda-10.1"

echo 'Using for include:', ${TF_INC}
echo 'Using for library:', ${TF_LIB}
   
#${CUDA_DIR}/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I${TF_INC} -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
${CUDA_DIR}/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC


g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I ${TF_INC} -I ${CUDA_DIR}/include -I ${TF_INC}/external/nsync/public -lcudart -L ${CUDA_DIR}/lib64/ -L${TF_LIB} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1


#g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I${TF_INC} -I${TF_INC}/external/nsync/public -L${TF_LIB} -ltensorflow_framework -I ${CUDA_DIR}/include -lcudart -L ${CUDA_DIR}/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
