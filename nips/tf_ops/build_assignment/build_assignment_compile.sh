# .so file cannot have the same name as .py file!
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared build_assignment.cc -o build_assignment_so.so -fPIC -I $TF_INC -O2
