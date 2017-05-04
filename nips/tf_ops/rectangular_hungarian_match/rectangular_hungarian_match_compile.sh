# .so file cannot have the same name as .py file!
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared rectangular_hungarian_match.cc -o rectangular_hungarian_match_so.so -fPIC -I $TF_INC -O2
