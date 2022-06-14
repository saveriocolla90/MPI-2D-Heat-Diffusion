#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define MASTER 0
#define TAG_IN 0
#define TAG_OUT 1

/* -- Global variables -- */
float **global_T;
int numtasks, rank;
FILE* fp;

/* -- Problem parameters structure -- */
typedef struct {
	int rows;
	int cols;
	float Cx;
	float Cy;
	int n_iter;
} Param;
Param param;


/* -- Functions prototypes -- */
void print_data();
float **malloc_2d_f(int, int);
void init_temperature(float*);
void compute_temperature(float*, float*, int);
void copy(float*, float*, int); 



/* -------- MAIN FUNCTION -------- */
int main(int argc, char *argv[]){

	// Paths to data file for writing
	char fdata_init[] = "./output/initial_data.txt";
	char fdata_out[] = "./output/final_data.txt";
	
	// Local variables
	int offset = 0;
	int rc = 911;
	
	// Checks if MPI initialization has been achieved
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS){
		perror("MPI initialization error!\nExit.\n");
		exit(1);
	}
	
	// Get total number of avaiable tasks
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	if (numtasks < 2 || numtasks > 16){
		printf("Error: number of tasks out of possible range. (# input tasks = %d)\n", numtasks);
		printf("Possible range: 2 <= tasks <= 16\nExit.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
		exit(0);	
	}
	
	// Get rank of current task
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Comunications controllers
	MPI_Status Status;
	MPI_Request Request;
	
	// Get input problem parameters
	param.rows = atoi(argv[1]);
	param.cols = atoi(argv[2]);
	param.Cx = atof(argv[3]);
	param.Cy = atof(argv[4]);
	param.n_iter = atoi(argv[5]);
	
	/* ------- Control the size limits of the code ------- */
	if (rank == MASTER){
		// Grid size limit
		if (param.rows <= 2 || param.cols <= 2){
			printf("Error: grid size too low !\n");
			printf("Minimum grid size [ ROWS x COLS ]: 4 x 3\nExit.");
			MPI_Abort(MPI_COMM_WORLD, rc);
			exit(0);				
		}
		
		// Number of tasks compared to grid size limit
		if (param.rows < numtasks + 2){
			printf("Error: number of rows lower then avaiable tasks !\n");
			printf("Insert a number of rows >= number of tasks + 2.\nExit.");
			MPI_Abort(MPI_COMM_WORLD, rc);
			exit(0);				
		}
	}
	/* --------------------------------------------------- */

	if (rank == MASTER){
		printf("=======================================================================\n");
		printf("----------------- MPI Heat diffusion 2D grid problem ------------------\n");
		printf("=======================================================================\n");
		printf("Total number of tasks: %d\n", numtasks);
		printf("Grid size [ ROWS x COLS ]: %d x %d\n", param.rows, param.cols);
		printf("Specific heat: Cx = %.2f\tCy = %.2f\n", param.Cx, param.Cy);
		printf("Time iterations: %d\n", param.n_iter);
		printf("-----------------------------------------------------------------------\n");
	}
	
	
	/* Divide work-load based on total number of tasks */
	int n_rows[numtasks];
	int rows_per_task = param.rows / numtasks;
	int rest_rows = param.rows % numtasks;
	for (int i=0; i < numtasks; i++){
		n_rows[i] = rows_per_task;
	}
	int k=0;
	while (rest_rows != 0){
		n_rows[k]++;
		rest_rows--;
		k++;
	}

	/* Initialize local grid based on the rows number per current task */
	float **old_local_T = malloc_2d_f(n_rows[rank], param.cols);
	float **new_local_T = malloc_2d_f(n_rows[rank], param.cols);




	/* ================================================================== */
	/* ------------- SPLIT WORK BETWEEN MASTER AND WORKERS -------------- */
	/* ================================================================== */		
	if (rank == MASTER) {
	 
		// Initialize whole grid temperature and write it to file
		global_T = malloc_2d_f(param.rows, param.cols);
		init_temperature(&(global_T[0][0]));
		print_data(fdata_init, &(global_T[0][0]), param.rows, param.cols);
		
		// Store MASTER initial part of workload 
		for (int row = 0; row < n_rows[rank]; row++){
			for(int col = 0; col < param.cols; col++){
				old_local_T[row][col] = global_T[row][col];
			}
		}
				
		// Sending initial data portions to all workers tasks
		offset = 0;
		for (int p = 1; p < numtasks; p++){
			offset += n_rows[p-1];
			MPI_Isend(&(global_T[offset][0]), n_rows[p] * param.cols, MPI_FLOAT, p, TAG_OUT, MPI_COMM_WORLD, &Request);
		}
	}
	else {	// If task is a Worker
		
		// Recieve initial part of data from MASTER task
		MPI_Irecv(&(old_local_T[0][0]), n_rows[rank] * param.cols, MPI_FLOAT, MASTER, TAG_IN + 1, MPI_COMM_WORLD, &Request);
	}
	// Wait first comunications to be completed
	MPI_Wait(&Request, &Status);




	/* ================================================================== */
	/* --------------- HEAT DIFFUSION TIME INTEGRATION ------------------ */
	/* ================================================================== */

	// Start time iteration
	int iterations = param.n_iter;
	float percent_bar = 20.;
	while (iterations > 0){
		iterations--;
		
		// Compute heat diffusion from old data to a new data grid ( N.B.: rows depending on task rank ) 
		compute_temperature(&(new_local_T[0][0]), &(old_local_T[0][0]), n_rows[rank]);
		
		// Copy new computed data to the old grid
		copy(&(new_local_T[0][0]), &(old_local_T[0][0]), n_rows[rank]);	
		
		// Master print log progress bar
		if (rank == MASTER && iterations % (int)(param.n_iter * percent_bar/100.) == 0){
			printf("Completed: %.1f %%\n", 100.*(1. - (float)iterations / param.n_iter));
		}
		
		// Wait for all tasks beafore the next time cycle
		MPI_Barrier(MPI_COMM_WORLD);			
	}



	/* ================================================================== */
	/* --------------- DATA COLLECTION & FINAL PRINTING ----------------- */
	/* ================================================================== */
	if (rank == MASTER) {
		// Set the new values for the computed part of the grid
		for (int row = 0; row < n_rows[rank]; row++){
			for(int col = 0; col < param.cols; col++){
				global_T[row][col] = new_local_T[row][col]; 
			}
		}
		
		// Reciveing all computed data portions from workers 
		offset = 0;
		for (int p = 1; p < numtasks; p++){
			offset += n_rows[p-1];
			MPI_Irecv(&(global_T[offset][0]), n_rows[p] * param.cols, MPI_FLOAT, p, TAG_IN, MPI_COMM_WORLD, &Request);
		}
		
	}
	else {	// Workers

		// Send the computed local grid to the master
		MPI_Isend(&(new_local_T[0][0]), n_rows[rank] * param.cols, MPI_FLOAT, MASTER, TAG_OUT - 1, MPI_COMM_WORLD, &Request);
	
	}

	// Wait final comunications to be completed
	MPI_Wait(&Request, &Status);


	// MASTER writes final computed grid temperature to file	
	if (rank == MASTER) {
		print_data(fdata_out, &(global_T[0][0]), param.rows, param.cols);
	}
	
	printf("Task %d: Finish comunication !\n", rank);	
	MPI_Finalize();
	
	return 0;
}








/* ============================================= */
/* ---------- FUNCTIONS DEFINITIONS ------------ */
/* ============================================= */

float **malloc_2d_f(int rows, int cols){
	/* Create a 2D matrix (rows x cols) allocating contigous memory cells,
	   and set initial values to zero. */
	
	float* data = (float *)malloc(rows*cols*sizeof(float));
	float **matrix = (float **)malloc(rows*sizeof(float*));
	
	// Set elements to zero
	for (int i=0; i<rows*cols; i++){
		*(data + i) = 0.;
	}

	// Split memory reference into rows
	for (int i=0; i<rows; i++){
		matrix[i] = &(data[cols*i]);
	}
	return matrix;
}

void copy(float* data, float* copied, int Nx){
	/* Copy given input data */
	
	int Ny = param.cols;
	for (int row = 0; row < Nx; row++){
		for (int col = 0; col < Ny; col++){
			*(copied + row*Ny + col) = *(data + row*Ny + col);
		}
	}
}

void print_data(char *path, float *data, int Nx, int Ny){
	/* Writing input data to the given path file */
	  
	printf("Writing data to file: %s\n", path);  
	
	fp = fopen(path, "w");
	for(int i=0; i < Nx; i++){
		for(int j=0; j < Ny; j++){
			fprintf(fp, "%f\t", *(data + (i * Ny) + j));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void init_temperature(float *data){
	/* Initialize the temperature values of the input grid:
	 	- zero boundary values
	   	- max temperature at the center max 
	*/
	
	printf("Initializing grid values...");
	for(int i=0; i < param.rows; i++){
		for(int j=0; j < param.cols; j++){
			*(data + i*param.cols + j) = (float)(i * (param.rows - i - 1) * j * (param.cols - j -1));
		}
	}	    
	printf("Done!\n");
}

void compute_temperature(float *new_data, float *old_data, int Nx){
	/* Compute heat diffusion equation from old data grid to the new one. 
	   Comunication protocol:
	   	- each task send using non-blocking comunication with TAG = rank
	   	- each task recive using blocking comunication with TAG = rank +/- 1 ( depending on neighbour/s )
	*/

	MPI_Status Status[param.cols];
	MPI_Request Request[param.cols];
	
	int Ny = param.cols;
	float up, down, left, right;
	int min_row, max_row;
	
	// Pair ranks compute in reverse rows order ( MASTER always does it! )
	if (rank % 2 == 0){
		
		// Master hasn't to perform computation for 1st row
		min_row = rank == MASTER;
		max_row = rank == numtasks - 1;
		
		for (int row = Nx - max_row - 1; row >= min_row; row--) {
			for (int col = 1; col < Ny - 1; col++) {
			
				// Master and last task comunicate with only 1 neighbour
				if (rank == MASTER && row == Nx - 1) {	
					MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, MASTER + 1, rank, MPI_COMM_WORLD, &Request[col]);
					up = *(old_data + (row - 1)*Ny + col);
					MPI_Recv(&down, 1, MPI_FLOAT, MASTER + 1, rank + 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception
				}
				else if (rank == numtasks - 1 && row == 0) {
					MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &Request[col]);
					down = *(old_data + (row + 1)*Ny + col);
					MPI_Recv(&up, 1, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception
				}
				else {
					
					// Check for rows that need data comunication from neighbours 
					if (rank != MASTER && row == 0){
						MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &Request[col]);
						down = *(old_data + (row + 1)*Ny + col);
						MPI_Recv(&up, 1, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception
					}
					else if (rank != numtasks - 1 && row == Nx - 1) {
						MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &Request[col]);
						up = *(old_data + (row - 1)*Ny + col);
						MPI_Recv(&down, 1, MPI_FLOAT, rank + 1, rank + 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception				
					}
					else {
						up = *(old_data + (row - 1)*Ny + col);
						down = *(old_data + (row + 1)*Ny + col);
					}				
				}
								
				// Set data from left and right directions
				left = *(old_data + row*Ny +(col - 1));
				right = *(old_data + row*Ny + (col + 1));
				
				// Compute heat diffusion equation
				*(new_data + row*Ny + col) = *(old_data + row*Ny + col)
								+ param.Cx*( down + up - 2.0*( *(old_data + row*Ny + col) )) 
								+ param.Cy*( right + left - 2.0*( *(old_data + row*Ny + col) ));  
			}
		}
	}
	else {
		// The last task has to leave out the bottom row 
		max_row = rank == numtasks - 1;
		
		for (int row = 0; row < Nx - max_row; row++) {
			for (int col = 1; col < Ny - 1; col++) {
		
				// Last task compute just 1 row by asking data from neighbour
				if (rank == numtasks - 1 && row == 0) {
					MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &Request[col]);
					down = *(old_data + (row + 1)*Ny + col);
					MPI_Recv(&up, 1, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception
				}

				// If data from 2 neighbours are needed...
				else {
					if (row == 0) {
						MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, rank - 1, rank, MPI_COMM_WORLD, &Request[col]);
						down = *(old_data + (row + 1)*Ny + col);
						MPI_Recv(&up, 1, MPI_FLOAT, rank - 1, rank - 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception
					}
					else if (rank != numtasks - 1 && row == Nx - 1) {
						MPI_Isend(old_data + row*Ny + col, 1, MPI_FLOAT, rank + 1, rank, MPI_COMM_WORLD, &Request[col]);
						up = *(old_data + (row - 1)*Ny + col);
						MPI_Recv(&down, 1, MPI_FLOAT, rank + 1, rank + 1, MPI_COMM_WORLD, &Status[col]);	// Block on reception				
					}
					else {
						up = *(old_data + (row - 1)*Ny + col);
						down = *(old_data + (row + 1)*Ny + col);
					}
				}

				// Set data from left and right directions
				left = *(old_data + row*Ny +(col - 1));
				right = *(old_data + row*Ny + (col + 1));
				
				// Compute heat diffusion equation
				*(new_data + row*Ny + col) = *(old_data + row*Ny + col)
								+ param.Cx*( down + up - 2.0*( *(old_data + row*Ny + col) )) 
								+ param.Cy*( right + left - 2.0*( *(old_data + row*Ny + col) ));  
			}
		}
	}
}



