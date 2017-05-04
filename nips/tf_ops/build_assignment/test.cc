#include <stdio.h>
#include <string.h>

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

int main() {
float x_data[12] = {0.18463433,0.77481091,0.59018219,0.11772865,0.39442706,0.14569713,0.02296862,0.19299343,0.80632013,0.29088834,0.75286949,0.94963497};
int y_data[4] = {0,1,2,0};
float cost_matrix[9];
memset(cost_matrix, 0, sizeof(cost_matrix));
build_assignment_matrix(1,2,2,3,x_data,y_data,cost_matrix);
for (int i=0; i<9; i++)
	printf("%f\n", cost_matrix[i]);
}
