#include <stdio.h>
#include "common.h"
#include <list>
#include <thrust/device_vector.h>

using namespace std;

#define N 5000000
#define DIM 2
#define R_MAX 1
#define SEED 5

struct Point{
	int index;
	float x;
	float y;
};

std::list<Point> listHull;

struct is_one {
  __host__ __device__
  bool operator()(const int &x)
  {
    return x == 1;
  }
};

__host__ void printConvexHull(){
	int counter = listHull.size();
	printf("\nThe points in Convex Hull are:\n");
	while (!listHull.empty()){
		printf("(%f, %f) ", listHull.begin()->x, listHull.begin()->y);
		listHull.erase(listHull.begin());
	}
	printf("\n\nTotal number of points in Convex Hull: %d\n\n", counter);
}

__host__ void printConvexHullDim(){
	printf("\nTotal number of points in Convex Hull: %d\n\n", (int) listHull.size());
}

__host__ void generatePoints(float* x, float* y, float* dist, int* flag, int n){
    for (int i = 0; i < n; i++) {
		x[i] = ((float) rand()) / ((float) RAND_MAX) * R_MAX;
		y[i] = ((float) rand()) / ((float) RAND_MAX) * R_MAX;
		dist[i] = 0;
		flag[i] = 0;
	}
}

__device__ int deviceFindSide(float p1_x, float p1_y, float p2_x, float p2_y, float p_x, float p_y){

	float val = (p_y - p1_y) * (p2_x - p1_x) - (p2_y - p1_y) * (p_x - p1_x);

	if (val > 0)
		return 1;

	if (val < 0)
		return -1;

	return 0;
}

__device__ float deviceLineDist(float p1_x, float p1_y, float p2_x, float p2_y, float p_x, float p_y){
	return abs((p_y - p1_y) * (p2_x - p1_x) - (p2_y - p1_y) * (p_x - p1_x));
}

__global__ void computeFlagsFromLine(float* x, float* y, int* flag, thrust::pair<float, float> p_left, thrust::pair<float, float> p_right, int n){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	if (flag[idx] == 0)
		return;

	flag[idx] = deviceFindSide(p_left.first, p_left.second, p_right.first, p_right.second, x[idx], y[idx]);
}

__global__ void computeSides(float* x, float* y, int* flag, thrust::pair<float, float> p_left, thrust::pair<float, float> p_right, int n){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	flag[idx] = deviceFindSide(p_left.first, p_left.second, p_right.first, p_right.second, x[idx], y[idx]);
}

__global__ void computeDistFromLine(float* x, float* y, int* flag, float* dist, thrust::pair<float, float> p_left, thrust::pair<float, float> p_right, int n){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	if (flag[idx] == 0)
	{
		dist[idx] = 0;
	}
	else
	{
		dist[idx] = deviceLineDist(p_left.first, p_left.second, p_right.first, p_right.second, x[idx], y[idx]);
	}

}

__global__ void isInsidePolygon(float p1_x, float p1_y, float p2_x, float p2_y, float p3_x, float p3_y, float p4_x, float p4_y, float *x, float *y, int *flag, int n){

	int temp1, temp2, temp3, temp4;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
	{
		return;
	}

	// Given the segment, finds the side where the point is located given the segment.
	temp1 = deviceFindSide(p1_x, p1_y, p2_x, p2_y, x[idx], y[idx]);
	temp2 = deviceFindSide(p2_x, p2_y, p3_x, p3_y, x[idx], y[idx]);
	temp3 = deviceFindSide(p3_x, p3_y, p4_x, p4_y, x[idx], y[idx]);
	temp4 = deviceFindSide(p4_x, p4_y, p1_x, p1_y, x[idx], y[idx]);

	// If the side is the same for every segment, then the point is inside the polygon. Else is out or is one point of the segment.
	if ((temp1 == temp2) && (temp2 == temp3) && (temp3 == temp4) && (temp4 == temp1))
    {
		flag[idx] = 0;
	}
    else
    {
		if ((x[idx]==p2_x && y[idx]==p2_y) || (x[idx]==p4_x && y[idx]==p4_y))
        {
			flag[idx] = 0;
		} 
        else
        {
			flag[idx] = 1;
		}
	}

}

__global__ void isInsideTriangle(float p1_x, float p1_y, float p2_x, float p2_y, float p3_x, float p3_y, float *x, float *y, int *flag, int n){
	
	int temp1, temp2, temp3;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	if (flag[idx] == 0)
		return;

	// Given the segment, finds the side where the point is located given the segment.
	temp1 = deviceFindSide(p1_x, p1_y, p2_x, p2_y, x[idx], y[idx]);
	temp2 = deviceFindSide(p2_x, p2_y, p3_x, p3_y, x[idx], y[idx]);
	temp3 = deviceFindSide(p3_x, p3_y, p1_x, p1_y, x[idx], y[idx]);

	// If the side is the same for every segment, then the point is inside the triangle. Else is out or is one point of the segment.
	if ((temp1 == temp2) && (temp2 == temp3) && (temp3 == temp1))
    {
		flag[idx] = 0;
	}
    else
    {
		if ((x[idx]==p2_x && y[idx]==p2_y))
        {
			flag[idx] = 0;
		}
        else
        {
            flag[idx] = 1;
        }
	}

}

__host__ void insertPoint(thrust::pair<float, float> point){
	Point new_point;
	new_point.x = point.first;
	new_point.y = point.second;
	listHull.push_back(new_point);
}

__host__ void convexHull(thrust::pair<float, float> p_left, thrust::pair<float, float> p_right, float* x, float* y, int* flag, float* dist, int start_index, int end_index){

	int num_points = end_index + 1 - start_index;
	int blocks = 1024;

	// If there are no points, the function will exit.
	if (num_points == 0)
		return;

	// If there is a point and is not a solution then, the function will exit.
	if (num_points == 1 && flag[start_index] == 0)
		return;
	
	// If there is a point and it is a solution then, it is added.
	if (num_points == 1 && flag[start_index] == 1)
    {
		thrust::pair<float, float> new_point;
		new_point.first = x[start_index];
		new_point.second = y[start_index];
		insertPoint(new_point);
		flag[start_index] = 0;
		return;
	}

	// Compute distance of every point contained from start_index to end index.
	computeDistFromLine<<<(num_points / blocks) + 1, blocks>>>(x + start_index, y + start_index, flag + start_index, dist + start_index, p_left, p_right, num_points);
	CHECK(cudaDeviceSynchronize());

	// Finds the point with the maximum distance.
	float *temp_dist = thrust::max_element(thrust::device, dist + start_index, dist + end_index + 1);

	// If the point with the maximum is zero then, the function will exit.
	if (*temp_dist == 0)
        return;

	// If not, the points inside the triangle formed by the point with max distance and the two starting points are checked.
	int index = temp_dist - dist;
	isInsideTriangle<<<(num_points / blocks) + 1, blocks>>>(p_left.first, p_left.second, x[index], y[index], p_right.first, p_right.second, x + start_index, y + start_index, flag + start_index, num_points);
	CHECK(cudaDeviceSynchronize());

	// The point with max distance is finally added to the solution.
	thrust::pair<float, float> max_dist_point;
	max_dist_point.first = x[index];
	max_dist_point.second = y[index];
	insertPoint(max_dist_point);

	// Recursive call for the two segments formed by the max dist point.
	// The first one checks the left segment, the second checks the right one.
	convexHull(p_left, max_dist_point, x, y, flag, dist, start_index, index - 1);
	convexHull(max_dist_point, p_right, x, y, flag, dist, index + 1, end_index);

}

__global__ void compactKernel(float* x, float* y, int* flag, float* nx, float* ny, int n, int n_points_left){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n || idx == 0)
		return;

	if (flag[idx-1]!=flag[idx])
	{
		nx[flag[idx-1]] = x[idx];
		ny[flag[idx-1]] = y[idx];
	}

}


int main(){

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.maxThreadsPerBlock;
	int blocksPerGrid = (N / blocks) + 1;
    double start, stopGPU, pStart, pStop;

	float *x, *y, *dist, *nx, *ny;
	int *flag;
	size_t sizeInt = N * sizeof(int);
	size_t sizeFloat = N * sizeof(float);

	printf("\n\tGPU:\tN %d - R_MAX %d - SEED %d\n\n", N, R_MAX, SEED);


	/* Memory allocation */
	CHECK(cudaMallocManaged(&x, sizeFloat));
	CHECK(cudaMallocManaged(&y, sizeFloat));
	CHECK(cudaMallocManaged(&dist, sizeFloat));
	CHECK(cudaMallocManaged(&flag, sizeInt));
	CHECK(cudaMallocManaged(&nx, sizeFloat));
	CHECK(cudaMallocManaged(&ny, sizeFloat));


    /* Generation of random points (CPU) */
	srand(SEED);
    generatePoints(x, y, dist, flag, N);
	

	start = seconds();


	/* 
	/ STEP 1
	/ Order the points in ascending order of x.
	*/
	pStart = seconds();
	thrust::stable_sort_by_key(thrust::device, x, x + N, y);
	pStop = seconds() - pStart;
	printf("\n\tSTEP 1: %f sec \n\n", pStop);


	/*
	/ STEP 2
	/ Extraction of 4 points, the points with the maximum and the minumum x and y values.
	/ Then the leftmost and rightmost points are inserted in the solution.
	*/
	pStart = seconds();
	thrust::pair<float *, float *> temp_x = thrust::minmax_element(thrust::device, x, x + N);
	thrust::pair<float *, float *> temp_y = thrust::minmax_element(thrust::device, y, y + N);
	int max_x = temp_x.second - x;
	int min_x = temp_x.first - x;
	int max_y = temp_y.second - y;
	int min_y = temp_y.first - y;
	thrust::pair<float, float> p_left, p_right;
	p_left.first = x[min_x];
	p_left.second = y[min_x];
	p_right.first = x[max_x];
	p_right.second = y[max_x];
	insertPoint(p_left);
	insertPoint(p_right);
	pStop = seconds() - pStart;
	printf("\n\tSTEP 2: %f sec \n\n", pStop);


	/*
	/ STEP 3 
	/ This kernel checks all the points inside the polygon formed by the points extracted above,
	/ it launches the maximum number of thread in a block.
	*/
	pStart = seconds();
	isInsidePolygon<<<blocksPerGrid, blocks>>>(x[min_y], y[min_y], p_left.first, p_left.second, x[max_y], y[max_y], p_right.first, p_right.second, x, y, flag, N);
	CHECK(cudaDeviceSynchronize());
	pStop = seconds() - pStart;
	printf("\n\tSTEP 3: %f sec \n\n", pStop);


	/*
	/ STEP 4
	/ The reduce step is used to calculate the number of points removed,
	/ necessary for the launch of the following kernels.
	*/
	pStart = seconds();
	int n_points_left = thrust::reduce(thrust::device, flag, flag + N);
	pStop = seconds() - pStart;
	printf("\n\tSTEP 4: %f sec \n\n", pStop);


	/*
	/ STEP 5
	/ A scan operation is performed on the flag array, then the kernel takes care of flattening
	/ the points contained in the array, eliminating those that wont be definitely part of the solution.
	*/
	pStart = seconds();
	thrust::inclusive_scan(thrust::device, flag, flag + N, flag);
	compactKernel<<<blocksPerGrid, blocks>>>(x, y, flag, nx, ny, N, n_points_left);
	CHECK(cudaDeviceSynchronize());
	pStop = seconds() - pStart;
	printf("\n\tSTEP 5: %f sec \n\n", pStop);


	/*
	/ STEP 6
	/ Calculates which side of the line the points are on and updates the corresponding flag value.
	*/
	pStart = seconds();
	computeSides<<<(n_points_left / blocks) + 1, blocks>>>(nx, ny, flag, p_left, p_right, n_points_left);
	CHECK(cudaDeviceSynchronize());
	pStop = seconds() - pStart;
	printf("\n\tSTEP 6: %f sec \n\n", pStop);


	/*
	/ STEP 7
	/ Divide the array between upper hull and lower hull, aligning all the values of the array belonging
	/ to the upper hull in the initial part of the array, then those belonging to the lower hull.
	*/
	pStart = seconds();
	thrust::stable_partition(thrust::device, nx, nx + n_points_left, flag, is_one());
	thrust::stable_partition(thrust::device, ny, ny + n_points_left, flag, is_one());
	pStop = seconds() - pStart;
	printf("\n\tSTEP 7: %f sec \n\n", pStop);


	/*
	/ STEP 8
	/ The reduce operation is performed to calculate where the first segment ends and where the new one begins.
	*/
	pStart = seconds();
	int temp_count = thrust::reduce(thrust::device, flag, flag + n_points_left);
	int segm = n_points_left - ((n_points_left - temp_count) / 2);
	pStop = seconds() - pStart;
	printf("\n\tSTEP 8: %f sec \n\n", pStop);


	/*
	/ STEP 9
	/ The function convexHull is called for both upper hull and lower hull, recursively until the problem is solved.
	*/
	pStart = seconds();
	convexHull(p_left, p_right, nx, ny, flag, dist, 0, segm-1);
	convexHull(p_left, p_right, nx, ny, flag, dist, segm, n_points_left-1);
	pStop = seconds() - pStart;
	printf("\n\tSTEP 9: %f sec \n\n", pStop);

	stopGPU = seconds() - start;


	/* Final result print */
	//printConvexHullDim();
	printConvexHull();
	printf("\n\tQuickhull GPU elapsed time %f sec \n\n", stopGPU);


	CHECK(cudaFree(x));
	CHECK(cudaFree(y));
	CHECK(cudaFree(dist));
	CHECK(cudaFree(flag));
	CHECK(cudaFree(nx));
	CHECK(cudaFree(ny));


	/*
	/ cudaDeviceReset must be called before exiting in order for profiling and
	/ tracing tools such as Nsight and Visual Profiler to show complete traces. 
	*/
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}