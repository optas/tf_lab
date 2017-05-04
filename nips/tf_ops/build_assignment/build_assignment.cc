#include <string.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

/*
Build an assignment cost matrix from image pixel probabilities.

x: BxHxWxN, 4-D probability map of an image for per-pixel class prediction.
y: BxHxW, 3-D label for each pixel.
cost_matrix: BxNxN, 3-D cost matrix for assignment matrix,
  cost_matrix[:,i,j] is the score that pred-class-i matches pred-class-j
*/
REGISTER_OP("BuildAssignment")
    .Input("x: float32")
    .Input("y: int32")
    .Output("cost_matrix: float32");
REGISTER_OP("BuildAssignmentGrad")
    .Input("x: float32")
    .Input("y: int32")
    .Output("grad: float32"); // grad is for x

void build_assignment_matrix(int b, int h, int w, int n, const float* x, const int* y, float* cost_matrix) {
	for (int k=0;k<b;k++){
		for (int i=0;i<h;i++){
			for (int j=0;j<w;j++){
				for (int l=0;l<n;l++){
					cost_matrix[l*n+y[i*w+j]] += x[i*(w*n)+j*n+l];
				}
			}
		}
		x += h*w*n;
		y += h*w;
		cost_matrix += n*n;
	}
}

//void build_assignment_matrix_grad(int b, int h, int w, int n, const float* x, const int* y, float* grad) {
//	for (int k=0;k<b;k++){
//		for (int i=0;i<h;i++){
//			for (int j=0;j<w;j++){
//				for (int l=0;l<n;l++){
//					//grad[i*(w*n)+j*n+l]= cost_matrix_grad[l*n+y[i*w+j]];
//				}
//			}
//		}
//		x += h*w*n;
//		y += h*w;
//		grad += h*w*n;
//	}
//}

class BuildAssignmentOp : public OpKernel {
	public:
		explicit BuildAssignmentOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& x_tensor = context->input(0);
			OP_REQUIRES(context, x_tensor.dims() == 4, errors::InvalidArgument("input must be 4-D BxHxWxN."));
			auto x_flat = x_tensor.flat<float>();
			const float* x = &(x_flat(0));
			int b = x_tensor.shape().dim_size(0);
			int h = x_tensor.shape().dim_size(1);
			int w = x_tensor.shape().dim_size(2);
			int n = x_tensor.shape().dim_size(3);

			const Tensor& y_tensor = context->input(1);
			OP_REQUIRES(context, y_tensor.dims() == 3, errors::InvalidArgument("input must be 3-D BxHxW."));
			auto y_flat = y_tensor.flat<int>();
			const int* y = &(y_flat(0));

			Tensor* cost_matrix_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,n}, &cost_matrix_tensor)); 
			auto cost_matrix_flat = cost_matrix_tensor->flat<float>();
			float* cost_matrix = &(cost_matrix_flat(0));
         		memset(cost_matrix, 0, sizeof(float)*b*n*n);

			build_assignment_matrix(b,h,w,n,x,y,cost_matrix);
		}
};

REGISTER_KERNEL_BUILDER(Name("BuildAssignment").Device(DEVICE_CPU), BuildAssignmentOp);


//class BuildAssignmentGradOp: public OpKernel {
//  public:
//    explicit BuildAssignmentGradOp(OpKernelConstruction* context) : OpKernel(context) {}
//    void Compute(OpKernelContext* context) override {
//	const Tensor& x_tensor = context->input(0);
//	OP_REQUIRES(context, x_tensor.dims() == 4, errors::InvalidArgument("input must be 4-D BxHxWxN."));
//      	auto x_flat = x_tensor.flat<float>();
//      	const float* x = &(x_flat(0));
//      	int b = x_tensor.shape().dim_size(0);
//      	int h = x_tensor.shape().dim_size(1);
//      	int w = x_tensor.shape().dim_size(2);
//      	int n = x_tensor.shape().dim_size(3);
//
//      	const Tensor& y_tensor = context->input(1);
//      	OP_REQUIRES(context, y_tensor.dims() == 3, errors::InvalidArgument("input must be 3-D BxHxW."));
//      	auto y_flat = y_tensor.flat<int>();
//      	const int* y = &(y_flat(0));
//
//	Tensor* grad_tensor = NULL;
//	OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,h,w,n}, &grad_tensor));
//	auto grad_flat = grad_tensor->flat<float>();
//	float* grad = &(grad_flat(0));
//        build_assignment_matrix_grad(b,h,w,n,x,y,grad);
//    }
//};
//
//REGISTER_KERNEL_BUILDER(Name("BuildAssignmentGrad").Device(DEVICE_CPU), BuildAssignmentGradOp);
