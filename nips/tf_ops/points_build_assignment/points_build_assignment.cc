#include <string.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

/*
Build an assignment cost matrix from image pixel probabilities.

x: BxMxN, 3-D probability map of a point cloud for per-point class prediction.
y: BxM, 2-D label for each point.
cost_matrix: BxNxN, 3-D cost matrix for assignment matrix,
  cost_matrix[:,i,j] is the score that pred-class-i matches pred-class-j
*/
REGISTER_OP("PointsBuildAssignment")
    .Input("x: float32")
    .Input("y: int32")
    .Output("cost_matrix: float32");
REGISTER_OP("PointsBuildAssignmentGrad")
    .Input("x: float32")
    .Input("y: int32")
    .Output("grad: float32"); // grad is for x

void build_assignment_matrix(int b, int m, int n, const float* x, const int* y, float* cost_matrix) {
	for (int k=0;k<b;k++){
		for (int i=0;i<m;i++){
			for (int l=0;l<n;l++){
				cost_matrix[l*n+y[i]] += x[i*n+l];
			}
		}
		x += m*n;
		y += m;
		cost_matrix += n*n;
	}
}

class PointsBuildAssignmentOp : public OpKernel {
	public:
		explicit PointsBuildAssignmentOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& x_tensor = context->input(0);
			OP_REQUIRES(context, x_tensor.dims() == 3, errors::InvalidArgument("input must be 3-D BxMxN."));
			auto x_flat = x_tensor.flat<float>();
			const float* x = &(x_flat(0));
			int b = x_tensor.shape().dim_size(0);
			int m = x_tensor.shape().dim_size(1);
			int n = x_tensor.shape().dim_size(2);

			const Tensor& y_tensor = context->input(1);
			OP_REQUIRES(context, y_tensor.dims() == 2, errors::InvalidArgument("input must be 2-D BxM."));
			auto y_flat = y_tensor.flat<int>();
			const int* y = &(y_flat(0));

			Tensor* cost_matrix_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,n}, &cost_matrix_tensor)); 
			auto cost_matrix_flat = cost_matrix_tensor->flat<float>();
			float* cost_matrix = &(cost_matrix_flat(0));
         		memset(cost_matrix, 0, sizeof(float)*b*n*n);

			build_assignment_matrix(b,m,n,x,y,cost_matrix);
		}
};

REGISTER_KERNEL_BUILDER(Name("PointsBuildAssignment").Device(DEVICE_CPU), PointsBuildAssignmentOp);
