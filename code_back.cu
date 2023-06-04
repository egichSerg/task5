#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <exception>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

// error checks
#define CUDACHECK(name) if (cudaGetLastError() != cudaSuccess) { throw std::runtime_error(name); } 
#define MPI_CHECK(code, name) if (code != MPI_SUCCESS) { throw std::runtime_error(name); }

// macros for average interpolation calculation
#define AVG_CALC(A, Anew, netSize, i, j) Anew[i * netSize + j] = 0.25 * (A[i * netSize + j - 1] + A[(i - 1) * netSize + j] + A[(i + 1) * netSize + j] + A[i * netSize + j + 1]);

// pointers for error and other matrixes
double 	*A 		= nullptr,  // buffer for main matrix
	*Anew		= nullptr,  // buffer for matrix where we store our interpolations
	*dev_A 	        = nullptr,  // A on device
	*dev_Anew	= nullptr,  // Anew on device
        *buff           = nullptr,  // buffer for abs_diff calculation
	*d_out 		= nullptr,  // buffer for error on device
	*d_temp_storage = nullptr;  // temporary buffer for cub max reduction

// handler funnction which executes before end of program execution and frees memory allocated dynamically
void free_pointers()
{
	std::cout << "End of execution" << std::endl;
	
	// free memory section
	if (A) 	            cudaFreeHost(A); 		CUDACHECK("free A")
	if (Anew) 	    cudaFreeHost(Anew); 	CUDACHECK("free Anew")
	if (dev_A)	    cudaFree(dev_A); 		CUDACHECK("free dev_A")
	if (dev_Anew) 	    cudaFree(dev_Anew); 	CUDACHECK("free dev_Anew")
    	if (buff)           cudaFree(buff); 		CUDACHECK("free buff")
	if (d_out) 	    cudaFree(d_out);	        CUDACHECK("free d_out")
	if (d_temp_storage) cudaFree(d_temp_storage);   CUDACHECK("free d_temp_storage")
		
	std::cout << "Memory has been freed" << std::endl;
}

// interpolation on matrix field
__global__ void iterateMatrix(double* A, double* Anew, size_t netSize, size_t rowsToProcess)
{
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if(!(x_idx < 1 || y_idx < 2 || x_idx > netSize - 2 || y_idx > rowsToProcess - 2)) {
		AVG_CALC(A, Anew, netSize, y_idx, x_idx)
	}	
}

// interpolation on the matrix edges between devices
__global__ void interpolate_boundaries(double* A, double* Anew, size_t netSize, size_t rowsToProcess){
	unsigned int up_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int down_idx = blockIdx.x * blockDim.x + threadIdx.x;

	// check if horizontal index between 1 and (netSize - 2) then calculates result
	if (!(up_idx == 0 || up_idx > netSize - 2)) {
		AVG_CALC(A, Anew, netSize, 1, up_idx)
		AVG_CALC(A, Anew, netSize, (rowsToProcess - 2), down_idx)
	}
}

// modular difference between A and Anew stored in buff
__global__ void abs_diff(double* A, double* Anew, double* buff, size_t netSize, size_t rowsToProcess) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t idx = y * netSize + x;
	
	// check if idx in allocated area then calculate result
	if(!(x <= 0 || y <= 0 || x >= (netSize - 1) || y >= (rowsToProcess - 1)))
	{
		buff[idx] = std::abs(A[idx] - Anew[idx]);
	}
}

int find_nearest_power_of_two(size_t num) {
    int power = 1;
    while (power < num) {
        power <<= 1;
    }
    return power;
}

int main(int argc, char* argv[])
{
	auto atExitStatus = std::atexit(free_pointers);
	if (atExitStatus != 0)
	{
		std::cout << "Register error" << std::endl;
		exit(-1);
	}

	if (argc != 4)
	{
		std::cout << "Invalid parameters count" << std::endl;
		std::exit(-1);
	}
	
	try {
		//arguments  netSize maxError max_iterations
		int netSize = std::stoi(argv[1]);
		double maxError = std::stod(argv[2]);
		int max_iterations = std::stoi(argv[3]);

		int matrixSize = netSize * netSize;  // total matrix netSize

		// rank - number of device, deviceAvailable - number of devices used by MPI, error_code - buffer for error message
		int rank, deviceAvailable, error_code;

		error_code = MPI_Init(&argc, &argv);
		MPI_CHECK(error_code, "mpi initialization")

		error_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_CHECK(error_code, "mpi communicator rank_init")

		// get device count used by MPI
		error_code = MPI_Comm_size(MPI_COMM_WORLD, &deviceAvailable);
		MPI_CHECK(error_code, "mpi communicator size_init")

		// check if program uses enough number of devices for next calculations
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		printf("%d - number of devices\n", deviceCount);
		if (deviceCount < deviceAvailable || deviceAvailable < 1) {
			std::cout << "INvalid number of devices!";
			std::exit(-1);
		}

		// set device
		cudaSetDevice(rank);
		CUDACHECK("cuda set device")
		printf("device rank: %d\n", rank);

		size_t rowsPerOneProcess = netSize / deviceAvailable;
		size_t start_y_idx = rowsPerOneProcess * rank;

		cudaMallocHost((void**)&A, matrixSize * sizeof(double));
		CUDACHECK("A host alloc")
		cudaMallocHost((void**)&Anew, matrixSize * sizeof(double));
		CUDACHECK("Anew host alloc")

		std::memset(A, 0, matrixSize * sizeof(double));
		std::memset(Anew, 0, matrixSize * sizeof(double));


		// matrix edge interpolation
		A[0] = 10.0;
		A[netSize - 1] = 20.0;
		A[netSize * netSize - 1] = 30.0;
		A[netSize * (netSize - 1)] = 20.0;

		Anew[0] = 10.0;
		Anew[netSize - 1] = 20.0;
		Anew[netSize * netSize - 1] = 30.0;
		Anew[netSize * (netSize - 1)] = 20.0;

		double step = 10.0 / (netSize - 1);
		for (int i = 1; i < netSize - 1; i++) {
			A[i] = A[0] + i * step;
			A[i * netSize] = A[0] + i * step;
			A[netSize - 1 + netSize * i] = A[netSize - 1] + i * step;
			A[netSize * (netSize - 1) + i] = A[netSize * (netSize - 1)] + i * step;

			Anew[i] = Anew[0] + i * step;
			Anew[i * netSize] = Anew[0] + i * step;
			Anew[netSize - 1 + netSize * i] = Anew[netSize - 1] + i * step;
			Anew[netSize * (netSize - 1) + i] = Anew[netSize * (netSize - 1)] + i * step;
		}

		// calculate used area for each process
		if (rank != 0 && rank != deviceAvailable - 1)
		{
			rowsPerOneProcess += 2;
		}
		else if (deviceAvailable != 1)
		{
			rowsPerOneProcess += 1;
		}

		// memory netSize for one device
		size_t alloc_memsize = netSize * rowsPerOneProcess;
		if (rank == deviceAvailable - 1 && deviceAvailable != 1) {
			alloc_memsize += netSize % deviceAvailable == 0 ? 0 : netSize * ( netSize - (netSize / deviceAvailable) * deviceAvailable );
			rowsPerOneProcess += netSize % deviceAvailable == 0 ? 0 : netSize - (netSize / deviceAvailable) * deviceAvailable;
		}

		cudaMalloc((void**)&buff, alloc_memsize * sizeof(double));
		CUDACHECK("alloc buff")
		cudaMalloc((void**)&dev_A, alloc_memsize * sizeof(double));
		CUDACHECK("alloc dev_A")
		cudaMalloc((void**)&dev_Anew, alloc_memsize * sizeof(double));
		CUDACHECK("alloc dev_Anew")


		size_t offset = (rank != 0) ? netSize : 0;
		cudaMemset(dev_A, 0, sizeof(double) * alloc_memsize);
		CUDACHECK("memset dev_A")
		cudaMemset(dev_Anew, 0, sizeof(double) * alloc_memsize);
		CUDACHECK("memset dev_Anew")
		cudaMemcpy(dev_A, A + (start_y_idx * netSize) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice);
		CUDACHECK("memcpy from A to dev_A from start_y_idx coordinate with offset")
		cudaMemcpy(dev_Anew, Anew + (start_y_idx * netSize) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice);
		CUDACHECK("memcpy from Anew to dev_Anew with from start_y_idx coordinate offset")


		double* d_out;
		cudaMalloc((void**)&d_out, sizeof(double));
		CUDACHECK("alloc d_out")

		// init max reduction
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, netSize * rowsPerOneProcess);
		CUDACHECK("get temp_storage_bytes")
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		CUDACHECK("temp storage memory allocation")

		double h_error = maxError + 1.0;  // current accuracy
		int iteration = 0; 

		// cuda_stream - for blocks to sync them, matrix_calc_stream - for other operation
		cudaStream_t cuda_stream, matrix_calc_stream;
		cudaStreamCreate(&cuda_stream);
		CUDACHECK("cuda_stream creation")
		cudaStreamCreate(&matrix_calc_stream);
		CUDACHECK("matrix_calc_stream creation")

		// params for cuda functions
		unsigned int threads_x = std::min(find_nearest_power_of_two(netSize), 1024);
		unsigned int blocks_y = rowsPerOneProcess;
		unsigned int blocks_x = netSize / threads_x + 1;

		dim3 blockDim(threads_x, 1);
		dim3 gridDim(blocks_x, blocks_y);

		while (iteration < max_iterations && h_error > maxError) {

			interpolate_boundaries<<<netSize, 1, 0, cuda_stream>>>(dev_A, dev_Anew, netSize, rowsPerOneProcess);

			iterateMatrix<<<gridDim, blockDim, 0, matrix_calc_stream>>>(dev_A, dev_Anew, netSize, rowsPerOneProcess);
			
			// updates h_error 1/100 times of main cycle iterations and on the last iteration
			if (iteration % 100 == 0 || iteration + 1 == max_iterations) {
				
				// synchronize to understand either we can make operations with matrix or not
				cudaStreamSynchronize(cuda_stream);
				CUDACHECK("cuda_stream synchronize after interpolation")

				abs_diff<<<gridDim, blockDim, 0, matrix_calc_stream>>>(dev_A, dev_Anew, buff, netSize, rowsPerOneProcess);


				cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, alloc_memsize, matrix_calc_stream);
				CUDACHECK("cub max reduction")

				// synchronize streams to receive actual d_out max values from all device
				cudaStreamSynchronize(matrix_calc_stream);
				CUDACHECK("matrix_calc_stream synchronization (inside error calculations)")

				// receive max d_out values from all devices
				error_code = MPI_Allreduce((void*)d_out, (void*)d_out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				MPI_CHECK(error_code, "mpi reduction")

				// copy values from d_out on GPU to h_error on CPU
				cudaMemcpyAsync(&h_error, d_out, sizeof(double), cudaMemcpyDeviceToHost, matrix_calc_stream);
				CUDACHECK("copy to h_error")
			}

			// receive top edge
			if (rank != 0)
			{
				error_code = MPI_Sendrecv(
					dev_Anew + netSize + 1, 
					netSize - 2,
					MPI_DOUBLE,
					rank - 1,
					0,
					dev_Anew + 1,
					netSize - 2,
					MPI_DOUBLE,
					rank - 1,
					0, 
					MPI_COMM_WORLD,
					MPI_STATUS_IGNORE
				);
				MPI_CHECK(error_code, "top edge receiving")
			}

			// receive bottom edge
			if (rank != deviceAvailable - 1)
			{
				error_code = MPI_Sendrecv(
					dev_Anew + (rowsPerOneProcess - 2) * netSize + 1, 
					netSize - 2, 
					MPI_DOUBLE, 
					rank + 1,
					0,
					dev_Anew + (rowsPerOneProcess - 1) * netSize + 1, 
					netSize - 2, 
					MPI_DOUBLE, 
					rank + 1, 
					0, 
					MPI_COMM_WORLD, 
					MPI_STATUS_IGNORE
				);
				MPI_CHECK(error_code, "bottom edge receiving")
			}

			// synchronize streams before starting next calculations
			cudaStreamSynchronize(matrix_calc_stream);
			CUDACHECK("matrix_calc_stream synchronization (main loop after MPI_Sendrecv)")

			++iteration;
			std::swap(dev_A, dev_Anew); // swap pointers for next calculations
		}

		cudaStreamDestroy(cuda_stream);
		CUDACHECK("Destroy cuda_stream")
		cudaStreamDestroy(matrix_calc_stream);
		CUDACHECK("Destroy matrix_calc_stream")

		MPI_Barrier(MPI_COMM_WORLD);
		error_code = MPI_Finalize();
		MPI_CHECK(error_code, "mpi finalize")

		if (rank == 0) {
			printf("\n\n\n\n=========================\nprogram ended on iteration %d with error %0.20g\n=========================\n\n\n\n", iteration, h_error);
		}

		std::cout << "MPI engine was shut down" << std::endl;

	}
	catch (std::runtime_error& error) {
		std::cout << error.what() << std::endl;
		std::cout << "Program execution stops" << std::endl;
		exit(-1);	
	}

	return 0;
}
