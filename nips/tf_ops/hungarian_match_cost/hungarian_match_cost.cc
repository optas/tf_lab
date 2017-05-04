#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

/*
Compute match cost from cost matrix and match matrix.

cost_matrix: BxNxN 3-D cost matrix for the assignment problem
match: BxNxN 3-D match matrix with values in 0 or 1 (one per row per column), can also be weights in theory

cost: B, 1-D cost computed as sum(cost_matrix .* match)
*/
REGISTER_OP("HungarianMatchCost")
    .Input("cost_matrix: float32")
    .Input("match: float32")
    .Output("cost: float32");
REGISTER_OP("HungarianMatchCostGrad")
    .Input("cost_matrix: float32")
    .Input("match: float32")
    .Output("grad: float32"); // grad is for cost_matrix

void match_cost(int b, int n, const float* cost_matrix, const float* match, float* cost) {
	for (int k=0;k<b;k++){
          	float s = 0;
		for (int i=0;i<n;i++){
			for (int j=0;j<n;j++){
				s += (cost_matrix[i*n+j]*match[i*n+j]);
			}
		}
		cost[0] = s;
		cost_matrix += n*n;
		match += n*n;
		cost += 1;
	}
}

class HungarianMatchCostOp : public OpKernel {
	public:
		explicit HungarianMatchCostOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& cost_matrix_tensor = context->input(0);
			OP_REQUIRES(context, cost_matrix_tensor.dims() == 3, errors::InvalidArgument("input must be 3-D BxNxN."));
			auto cost_matrix_flat = cost_matrix_tensor.flat<float>();
			const float* cost_matrix = &(cost_matrix_flat(0));
			int b = cost_matrix_tensor.shape().dim_size(0);
			int n = cost_matrix_tensor.shape().dim_size(1);

			const Tensor& match_tensor = context->input(1);
			OP_REQUIRES(context, match_tensor.dims() == 3, errors::InvalidArgument("input must be 3-D BxNxN."));
			auto match_flat = match_tensor.flat<float>();
			const float* match = &(match_flat(0));

			Tensor* cost_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b}, &cost_tensor)); 
			auto cost_flat = cost_tensor->flat<float>();
			float* cost = &(cost_flat(0));

			match_cost(b,n,cost_matrix,match,cost);
		}
};

REGISTER_KERNEL_BUILDER(Name("HungarianMatchCost").Device(DEVICE_CPU), HungarianMatchCostOp);
