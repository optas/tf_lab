# .so file cannot have the same name as .py file!
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared hungarian_match_cost.cc -o hungarian_match_cost_so.so -fPIC -I $TF_INC -O2
