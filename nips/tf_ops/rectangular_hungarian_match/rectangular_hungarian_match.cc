#include <cstring>
#include <string.h>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

// using namespace std;
using namespace tensorflow;

#define INF 100000000 //just infinity

/*
Compute **MAX** cost matching for an assignment problem using Hungarian algorithm.

x: BxNxM 3-D cost matrix
match: BxNxM 3-D match matrix of 0 or 1

Grad op will be combined with HungarianMatchCost
*/
REGISTER_OP("RectangularHungarianMatch")
    .Input("x: float32")
    .Output("match: int32");

// inputs:
// 	int b: batch size
// 	int n: number of workers (predicted binary masks)
// 	int m: number of tasks (ground truth binary masks)
// 	const float* x: bxnxm cost matrix

void rectangular_hungarian_match(const int b, const int n, const int m, const float* x, int* match);
void hungarian(int n, int m, const float* const_cost, int* match);
void copy_cost(const float* cost, float* copied_cost, const int n, const int m);
void step_one(int* step, float* C, int n, int m);
void step_two(int* step, int n, int m, const float* C, int* M, int* RowCover, int* ColCover);
void step_three(int* step, int n, int m, int* ColCover, const int* M);
void step_four(int* step, int* M, int* RowCover, int* ColCover, int* path_row_0, int* path_col_0, float* C, int n, int m);
void step_five(int* step, int* path, const int path_row_0, const int path_col_0, int* M, const int n, const int m, int* path_count, int* RowCover, int* ColCover);
void step_six(int* step, const int n, const int m, const int* RowCover, const int* ColCover, float* C);
void step_seven(int* step);
void ShowCostMatrix(const float* C, int n, int m);
void ShowMatchMatrix(const int* M, int n, int m);
void ShowRowCover(const int* RowCover, int n);
void ShowColCover(const int* ColCover, int m);
// methods for step 4
void find_a_zero(int* row, int* col, const float* C, int* RowCover, int* ColCover, int n, int m);
bool star_in_row(int row, const int* M, int m);
void find_star_in_row(int row, int* col, int m, const int* M);
// methods for step 5
void find_star_in_col(const int c, int* r, const int n, const int m, const int* M);
void find_prime_in_row(const int r, int* c, const int n, const int m, const int* M);
void augment_path(const int path_count, int* M, const int* path, const int m);
void clear_covers(const int n, const int m, int* RowCover, int* ColCover);
void erase_primes(const int n, const int m, int* M);
// method for step 6
void find_smallest(float* minval, const int n, const int m, const int* RowCover, const int* ColCover, const float* C);



void rectangular_hungarian_match(const int b, const int n, const int m, const float* x, int* match) {
  for (int k=0; k < b; k++) {
    hungarian(n, m, x, match);
    x += n*m;
    match += n*m;
  }
}

void copy_cost(const float* cost, float* copied_cost, const int n, const int m){
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < m; c++) {
      copied_cost[r*m+c] = cost[r*m+c];
    }
  }
}

void hungarian(int n, int m, const float* const_cost, int* match) {
	bool done = false;
	int step = 1;
	int RowCover[n];
	memset(RowCover, 0, sizeof(RowCover));
	int ColCover[m];
	memset(ColCover, 0, sizeof(ColCover));
	int path_row_0 = -1;
	int path_col_0 = -1;
	int path[(2*m+1)*2]; // TODO be careful with the size of path
	memset(path, 0, sizeof(path));
	int path_count = 0;
  float cost[n*m];
  memset(cost, 0.0, sizeof(cost));
  copy_cost(const_cost, cost, n, m);

  //ShowCostMatrix(cost, n, m);
  //ShowMatchMatrix(match, n, m);
  //ShowRowCover(RowCover, n);
  //ShowColCover(ColCover, m);
	while (!done)
	{
    /*
		ShowCostMatrix(cost, n, m);
		ShowMatchMatrix(match, n, m);
		ShowRowCover(RowCover, n);
		ShowColCover(ColCover, m);
    */
   
		switch (step)
		{
			case 1:
				step_one(&step, cost, n, m);
				break;
			case 2:
			 	step_two(&step, n, m, cost, match, RowCover, ColCover);
				break;
			case 3:
				step_three(&step, n, m, ColCover, match);
				break;
			case 4:
				step_four(&step, match, RowCover, ColCover, &path_row_0, &path_col_0, cost, n, m);
				break;
			case 5:
				step_five(&step, path, path_row_0, path_col_0, match, n, m, &path_count, RowCover, ColCover);
				break;
			case 6:
				step_six(&step, n, m, RowCover, ColCover, cost);
				break;
			case 7:
			 	step_seven(&step);
			  done = true;
			  break;
		}
		// sleep(3);
	}
	//ShowCostMatrix(cost, n, m);
	//ShowMatchMatrix(match, n, m);
	//ShowRowCover(RowCover, n);
	//ShowColCover(ColCover, m);
}

//For each row of the cost matrix, find the smallest element and subtract
//it from every element in its row.  When finished, Go to Step 2.
void step_one(int* step, float* C, int n, int m) {
	//std::cout << "!!!!!!!!!!!!! STEP " << *step << std::endl;
	int min_in_row;
	
	for (int r = 0; r < n; r++)
	{
		min_in_row = C[r*m+0];
		for (int c = 0; c < m; c++)
	 	{
			if (C[r*m+c] < min_in_row)
			{
				min_in_row = C[r*m+c];
			}
		}
		for (int c = 0; c < m; c++) {
			C[r*m+c] -= min_in_row;
		}
	}
	//ShowCostMatrix(C, n, m);
	*step = 2;
	//std::cout << "****************" << std::endl;
}

//Find a zero (Z) in the resulting matrix.  If there is no starred 
//zero in its row or column, star Z. Repeat for each element in the 
//matrix. Go to Step 3.
void step_two(int* step, int n, int m, const float* C, int* M, int* RowCover, int* ColCover) {
	//std::cout << "!!!!!!!!!!!!! STEP " << *step << std::endl;
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			if (C[r*m+c] == 0 && RowCover[r] == 0 && ColCover[c] == 0) {
				M[r*m+c] = 1;
				RowCover[r] = 1;
				ColCover[c] = 1;
			}
		}
	}
	//ShowMatchMatrix(M, n, m);
	//ShowRowCover(RowCover, n);
	//ShowColCover(ColCover, m);

	for (int r = 0; r < n; r++){
		RowCover[r] = 0;
	}
	for (int c = 0; c < m; c++){
		ColCover[c] = 0;
	}
	*step = 3;
	//std::cout << "****************" << std::endl;
}

void step_three(int* step, int n, int m, int* ColCover, const int* M) {
	//std::cout << "!!!!!!!!!!!!! STEP " << *step << std::endl;
	int colcount = 0;
	//ShowMatchMatrix(M, n, m);
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			if (M[r*m+c] == 1) {
				ColCover[c] = 1;
			}
		}
	}
	for (int c = 0; c < m; c++) {
		if (ColCover[c] == 1) {
			colcount += 1;
		}
	}
	if (colcount >= m || colcount >= n) {
		*step = 7;
	} else {
		*step = 4;
	}
	//ShowColCover(ColCover, m);
	//std::cout << "col count " << colcount << std::endl;
	//std::cout << "****************" << std::endl;
}

void step_four(int* step, int* M, int* RowCover, int* ColCover, int* path_row_0, int* path_col_0, float* C, int n, int m) {
	//std::cout << "!!!!!!!!!!!!! STEP " << *step << std::endl;
	int row = -1;
	int col = -1;
	bool done;

	done = false;
	while (!done) {
		find_a_zero(&row, &col, C, RowCover, ColCover, n, m); // TODO Test
		//std::cout << "back to step four" << std::endl;
		if (row == -1) {
			done = true;
			*step = 6;
		} else {
			//std::cout << row << ' ' << col << std::endl;
			M[row*m+col] = 2;
			if (star_in_row(row, M, m)) { // TODO Test
				find_star_in_row(row, &col, m, M); // TODO Test
				//std::cout << "find star in row " << row << " in col " << col << std::endl;
				RowCover[row] = 1;
				ColCover[col] = 0;
				//ShowRowCover(RowCover, n);
				//ShowColCover(ColCover, m);
			} else {
				done = true;
				*step = 5;
				*path_row_0 = row;
				*path_col_0 = col;
			}
		}
	}
	//std::cout << "****************" << std::endl;
}

// find an uncovered zero
void find_a_zero(int* row, int* col, const float* C, int* RowCover, int* ColCover, int n, int m) {
	//std::cout << "finding zero ..." << std::endl;
	int r = 0;
	int c;
	bool done;
	*row = -1;
	*col = -1;
	done = false;
	while (!done) {
		c = 0;
		while (true) {
			if (C[r*m+c] == 0 && RowCover[r] == 0 && ColCover[c] == 0) {
				*row = r;
				*col = c;
				done = true;
			}
			c += 1;
			if (c >= m || done) {
				break;
			}
		}
		r += 1;
		if (r >= n) {
			done = true;
		}
	}
	//std::cout << *row << ' ' << *col << std::endl;
}

bool star_in_row(int row, const int* M, int m) {
	bool tmp = false;
	for (int c = 0; c < m; c++) {
		if (M[row*m+c] == 1) {
			tmp = true;
		}
	}
	return tmp;
}

void find_star_in_row(int row, int* col, int m, const int* M) {
	*col = -1;
	for (int c = 0; c < m; c++) {
		if (M[row*m+c] == 1) {
			*col = c;
		}
	}
}

void step_five(int* step, int* path, const int path_row_0, const int path_col_0, int* M, const int n, const int m, int* path_count, int* RowCover, int* ColCover) {
	//std::cout << "!!!!!!!!!!!!! STEP " << *step << std::endl;
	//std::cout << "path_row_0 " << path_row_0 << std::endl;
	//std::cout << "path_col_0 " << path_col_0 << std::endl;
	bool done;
	int r = -1;
	int c = -1;

	*path_count = 1;
	path[(*path_count - 1)*2+0] = path_row_0;
	path[(*path_count - 1)*2+1] = path_col_0;
	done = false;
	while (!done) {
		find_star_in_col(path[(*path_count-1)*2+1], &r, n, m, M); // TODO
		if (r > -1) {
			*path_count += 1;
			path[(*path_count-1)*2+0] = r;
			path[(*path_count-1)*2+1] = path[(*path_count-2)*2+1];
		} else {
			done = true;
		}
		if (!done) {
			find_prime_in_row(path[(*path_count-1)*2+0], &c, n, m, M); // TODO
			*path_count += 1;
			path[(*path_count-1)*2+0] = path[(*path_count-2)*2+0];
			path[(*path_count-1)*2+1] = c;
		}
	}
	augment_path(*path_count, M, path, m); // TODO
	clear_covers(n, m, RowCover, ColCover); // TODO
	erase_primes(n, m, M); // TODO
	*step = 3;
	//std::cout << "****************" << std::endl;
}

void find_star_in_col(const int c, int* r, const int n, const int m, const int* M) {
	*r = -1;
	for (int i = 0; i < n; i++) {
		if (M[i*m+c] == 1) {
			*r = i;
		}
	} 
}

void find_prime_in_row(const int r, int* c, const int n, const int m, const int* M) {
	for (int j = 0; j<m; j++) {
		if (M[r*m+j] == 2) {
			*c = j;
		}
	}
}

void augment_path(const int path_count, int* M, const int* path, const int m) {
	for (int p = 0; p < path_count; p++) {
		if (M[path[p*2+0]*m+path[p*2+1]] == 1) {
			M[path[p*2+0]*m+path[p*2+1]] = 0;
		} else {
			M[path[p*2+0]*m+path[p*2+1]] = 1;
		}
	}
}

void clear_covers(const int n, const int m, int* RowCover, int* ColCover) {
	for (int r = 0; r < n; r++) {
		RowCover[r] = 0;
	}
	for (int c = 0; c < m; c++) {
		ColCover[c] = 0;
	}
}

void erase_primes(const int n, const int m, int* M) {
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			if (M[r*m+c] == 2) {
				M[r*m+c] = 0;
			}
		}
	}
}

void step_six(int* step, const int n, const int m, const int* RowCover, const int* ColCover, float* C) {
  //std::cout << "!!!!!!!!!!!!! STEP " << *step << std::endl;
	//ShowCostMatrix(C, n, m);
	float minval = 100000000.0;
	find_smallest(&minval, n, m, RowCover, ColCover, C); // TODO
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			if (RowCover[r] == 1) {
				C[r*m+c] += minval;
			}
			if (ColCover[c] == 0) {
				C[r*m+c] -= minval;
			}
		}
	}
	//ShowCostMatrix(C, n, m);
	*step = 4;
	//std::cout << "****************" << std::endl;
}

void find_smallest(float* minval, const int n, const int m, const int* RowCover, const int* ColCover, const float* C) {
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			if (RowCover[r] == 0 && ColCover[c] == 0) {
				if (*minval > C[r*m+c]) {
					*minval = C[r*m+c];
				}
			}
		}
	}
}

void step_seven(int* step) {
	//std::cout << "----------Run Complete----------" <<std::endl;
}


void ShowCostMatrix(const float* C, int n, int m){
	std::cout << "Cost matrix: " << std::endl;
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			std::cout << C[r*m+c] << ' ';
		}
		std::cout << std::endl;
	}
}

void ShowMatchMatrix(const int* M, int n, int m){
	std::cout << "Match matrix: " << std::endl;
	for (int r = 0; r < n; r++) {
		for (int c = 0; c < m; c++) {
			std::cout << M[r*m+c] << ' ';
		}
		std::cout << std::endl;
	}
}

void ShowRowCover(const int* RowCover, int n){
	std::cout << "Row cover: " << std::endl;
	for (int r = 0; r < n; r++) {
		std::cout << RowCover[r] << ' ';
	}
	std::cout << std::endl;
}

void ShowColCover(const int* ColCover, int m){
	std::cout << "Col cover: " << std::endl;
	for (int c = 0; c < m; c++) {
		std::cout << ColCover[c] << ' ';
	}
	std::cout << std::endl;
}



class RectangularHungarianMatchOp : public OpKernel {
  public:
    explicit RectangularHungarianMatchOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      const Tensor& x_tensor = context->input(0);
      OP_REQUIRES(context, x_tensor.dims() == 3, errors::InvalidArgument("input must be 3-D BxNxM."));
      auto x_flat = x_tensor.flat<float>();
      const float* x = &(x_flat(0));
      int b = x_tensor.shape().dim_size(0);
      int n = x_tensor.shape().dim_size(1);
      int m = x_tensor.shape().dim_size(2);

      Tensor* match_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,n,m}, &match_tensor)); 
      auto match_flat = match_tensor->flat<int>();
      int* match = &(match_flat(0));
      memset(match, 0, sizeof(int)*b*n*m);

      rectangular_hungarian_match(b,n,m,x,match);
    }
};

REGISTER_KERNEL_BUILDER(Name("RectangularHungarianMatch").Device(DEVICE_CPU), RectangularHungarianMatchOp);
