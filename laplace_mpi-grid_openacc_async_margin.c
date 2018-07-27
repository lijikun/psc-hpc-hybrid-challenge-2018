// Requires OpenMPI
// OpenACC and OpenMP are optional
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#ifdef _OPENACC
    #include "openacc.h"
    #ifndef ACC_VECLEN
        #define ACC_VECLEN 32
    #endif
    #define ACC_WORKERS (1024/ACC_VECLEN)
#endif
#ifndef MARGIN
    #define MARGIN 1024
#endif
// Global data size.
#define COLUMNS_GLOBAL 10000
#define ROWS_GLOBAL 10000
// Position to monitor.
#define WATCHI 7500
#define WATCHJ 9950
// Default is 1 node with 2 PEs. To change, use compiler flags: 
// -DNODES=$m -DPES_PER_NODE=$n -DNODE_ROWS=$p -DPE_ROWS=$q
#ifndef NODES
    #define NODES 1
#endif
#ifndef PES_PER_NODE
    #define PES_PER_NODE 2
#endif
#define NPES (NODES*PES_PER_NODE)
#ifndef NODE_ROWS 
    #define NODE_ROWS 1
#endif
#define NODE_COLS (NODES/NODE_ROWS)
#ifndef PE_ROWS
    #define PE_ROWS 2
#endif
#define PE_COLS (PES_PER_NODE/PE_ROWS)
// Communication tags
#define UP 256
#define DOWN 128
#define LEFT 64
#define RIGHT 32  
// Constants of the science & numerical aspects.
#define HEATER_MAX 100.0
#define MAX_TEMP_ERROR 0.01
#define MAX_ITERATIONS 5000
// Divides data into equal pieces accordingly. Only stores local data.
#define ROWS (ROWS_GLOBAL/NODE_ROWS/PE_ROWS)
#define COLUMNS (COLUMNS_GLOBAL/NODE_COLS/PE_COLS)
#define WATCH_PE_ROW (WATCHI-1)/ROWS
#define WATCH_PE_COL (WATCHJ-1)/COLUMNS
#define WATCH_LOCALI (WATCHI-1)%ROWS+1
#define WATCH_LOCALJ (WATCHJ-1)%COLUMNS+1
double Temperature[ROWS+2][COLUMNS+2];
double Temperature_last[ROWS+2][COLUMNS+2];
// Use this array to map PE location to global rank.
static int PE_Grid[NODE_ROWS*PE_ROWS][NODE_COLS*PE_COLS]={0};

void initialize(int const my_PE_row, int const my_PE_col);

    
int main(int argc, char *argv[]) {

    int i, j, k;
    int iteration=1;
    double dt;
    double dt_global=10000.0;
    double start_time, finish_time;
    
    int npes, my_PE_num, my_node_num, my_local_PE_num, my_PE_row, my_PE_col;
    int PE_rows[NPES], PE_cols[NPES];
    int has_left=1, has_right=1, has_up=1, has_down=1;

    MPI_Request request0, request1, requests[8];
    MPI_Status status0, status1, statuses[8];
    MPI_Datatype row_vec, col_vec;

    // MPI startup
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);
    // Verifies PE numbers and prints PE topology.
    if (npes != NPES) {
        if (my_PE_num == 0)
            printf("Incorrect number of PE's. Must be %d; has %d.\n", NPES, npes);
        MPI_Finalize();
        return 1;
    }
    /*char my_hostname[MPI_MAX_PROCESSOR_NAME+1];
    int my_hostname_len;
    MPI_Get_processor_name(my_hostname, &my_hostname_len);*/
    
    // Obtains node's "rank" among all nodes and this PE's rank on local node.
    my_node_num = my_PE_num % NODES;
    my_local_PE_num = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    // Grabs GPU if using OpenACC
    #ifdef _OPENACC
        int ngpus, my_gpu=0;
        ngpus = acc_get_num_devices(acc_device_nvidia);
        if (ngpus > 0) {
            my_gpu = my_local_PE_num % ngpus;
            acc_set_device_num(my_gpu, acc_device_nvidia);
        }
        else
            acc_set_device_type(acc_device_host);
    #endif
    // Calculates the location of this PE in the grid.
    my_PE_row = (my_node_num / NODE_COLS) * PE_ROWS + my_local_PE_num / PE_COLS;
    my_PE_col = (my_node_num % NODE_COLS) * PE_COLS + my_local_PE_num % PE_COLS;
    // And broadcast this information to all PEs. 
    MPI_Iallgather(&my_PE_row, 1, MPI_INT, PE_rows, 1, MPI_INT, MPI_COMM_WORLD, &request0);
    MPI_Iallgather(&my_PE_col, 1, MPI_INT, PE_cols, 1, MPI_INT, MPI_COMM_WORLD, &request1);
    // MPI types for vector messaging
    MPI_Type_contiguous(COLUMNS, MPI_DOUBLE, &row_vec);
    MPI_Type_commit(&row_vec);
    MPI_Type_vector(ROWS, 1, COLUMNS+2, MPI_DOUBLE, &col_vec);
    MPI_Type_commit(&col_vec);
    // Makes sure all grid information has arrived before continuing.
    MPI_Wait(&request0, &status0);
    MPI_Wait(&request1, &status1);
    for (i = 0; i < NPES; ++i)
        PE_Grid[PE_rows[i]][PE_cols[i]] = i;
    /*if (my_PE_num == NPES - 1) {
        printf("PE Arrangement:\n");
        for (i = 0; i < NODE_ROWS*PE_ROWS; ++i) {
            for (j = 0; j < NODE_COLS*PE_COLS; ++j) 
                printf(" %d", PE_Grid[i][j]);
            printf("\n");
        }
    }*/

    // Timer starts here. Use MPI_Wtime() for portable timer.
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_PE_num == PE_Grid[WATCH_PE_ROW][WATCH_PE_COL]) {
        //printf("PE %d will hold the timer.\n", my_PE_num);
        start_time = MPI_Wtime();
    }
    
    // The meat of the program.
    #pragma acc enter data create(Temperature_last)
    // Initializes the array.
    initialize(my_PE_row, my_PE_col);
    
    // Initializes message passing. Can't use GPU pointers b/c hw/driver doesn't supports RDMA.
    // #pragma acc host_data use_device(Temperature, Temperature_last)
    #pragma acc enter data create(Temperature)
    {
        // left column
        if (my_PE_col == 0) {
            requests[0] = MPI_REQUEST_NULL;
            requests[1] = MPI_REQUEST_NULL;
            has_left = 0;
        }
        else {
            MPI_Send_init(&Temperature[1][1], 1, col_vec, 
                PE_Grid[my_PE_row][my_PE_col-1], LEFT, MPI_COMM_WORLD, &requests[0]);
            MPI_Recv_init(&Temperature_last[1][0], 1, col_vec, 
                PE_Grid[my_PE_row][my_PE_col-1], RIGHT, MPI_COMM_WORLD, &requests[1]);
        }
        // right column
        if (my_PE_col == NODE_COLS*PE_COLS-1) {
            requests[2] = MPI_REQUEST_NULL;
            requests[3] = MPI_REQUEST_NULL;
            has_right = 0;
        }
        else {
            MPI_Send_init(&Temperature[1][COLUMNS], 1, col_vec, 
                PE_Grid[my_PE_row][my_PE_col+1], RIGHT, MPI_COMM_WORLD, &requests[2]);
            MPI_Recv_init(&Temperature_last[1][COLUMNS+1], 1, col_vec, 
                PE_Grid[my_PE_row][my_PE_col+1], LEFT, MPI_COMM_WORLD, &requests[3]);
        }
        // top row
        if (my_PE_row == 0) {
            requests[4] = MPI_REQUEST_NULL;
            requests[5] = MPI_REQUEST_NULL;
            has_up = 0;
        }
        else {
            MPI_Send_init(&Temperature[1][1], 1, row_vec, 
                PE_Grid[my_PE_row-1][my_PE_col], UP, MPI_COMM_WORLD, &requests[4]);
            MPI_Recv_init(&Temperature_last[0][1], 1, row_vec, 
                PE_Grid[my_PE_row-1][my_PE_col], DOWN, MPI_COMM_WORLD, &requests[5]);
        }
        // bottom row
        if (my_PE_row == NODE_ROWS*PE_ROWS-1) {
            requests[6] = MPI_REQUEST_NULL;
            requests[7] = MPI_REQUEST_NULL;
            has_down = 0;
        }
        else {
            MPI_Send_init(&Temperature[ROWS][1], 1, row_vec, 
                PE_Grid[my_PE_row+1][my_PE_col], DOWN, MPI_COMM_WORLD, &requests[6]);
            MPI_Recv_init(&Temperature_last[ROWS+1][1], 1, row_vec,
                PE_Grid[my_PE_row+1][my_PE_col], UP, MPI_COMM_WORLD, &requests[7]);
        }
    }
    #pragma acc wait

    #pragma acc data present(Temperature,Temperature_last)
    while ( dt_global > MAX_TEMP_ERROR && iteration < MAX_ITERATIONS ) {
        
        // Determine the size of the margin.
        int const large_enough = (ROWS > 2 && COLUMNS > MARGIN*2) ? 1:0;
        int const upper_bound = large_enough? 2:1;
        int const lower_bound = large_enough? ROWS-1:ROWS;
        // Calculate one ACC_VECLEN so that cycles are not wasted.
        int const left_bound = large_enough? MARGIN+1:1;
        int const right_bound = large_enough? COLUMNS-(COLUMNS-1)%MARGIN-1: COLUMNS;

        // main calculation: average my four neighbors
        // 4 boundaries first: async queue 1
        if (large_enough == 1) { 
            #pragma acc parallel loop async(1) gang worker \
                    device_type(nvidia) num_workers(ACC_WORKERS) vector_length(ACC_VECLEN)
            for (i = 2; i < ROWS; ++i) {
                #pragma acc loop worker vector
                for (j = 1; j <= left_bound - 1; ++j) {
                    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
                }
                #pragma acc loop worker vector
                for (j = right_bound + 1; j <= COLUMNS; ++j) {
                    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
                }
            }
            i = 1;
            #pragma acc parallel loop async(1) gang worker vector device_type(nvidia) vector_length(ACC_VECLEN)
            for (j = 1; j<= COLUMNS; ++j) {
                    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
            i = ROWS;
            #pragma acc parallel loop async(1) gang worker vector device_type(nvidia) vector_length(ACC_VECLEN)
            for (j = 1; j<= COLUMNS; ++j) {
                    Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        // inner region second: async queue 2
        #pragma acc parallel loop async(2) gang worker \
                device_type(nvidia) num_workers(ACC_WORKERS) vector_length(ACC_VECLEN)
        for(i = upper_bound; i <= lower_bound ; i++) {
            #pragma acc loop worker vector
            for(j = left_bound; j <= right_bound; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        #pragma acc wait(2) if(large_enough==0)
        #pragma acc update host(Temperature[1:1][1:COLUMNS]) if(has_up) async(1)
        #pragma acc update host(Temperature[ROWS:1][1:COLUMNS]) if(has_down) async(1)
        #pragma acc update host(Temperature[1:ROWS][1:1]) if(has_left) async(1)
        #pragma acc update host(Temperature[1:ROWS][COLUMNS:1]) if(has_right) async(1)

        // find global dt
        dt = 0.0;
        #pragma acc parallel loop reduction(max:dt) async(2) gang worker \
                device_type(nvidia) num_workers(ACC_WORKERS) vector_length(ACC_VECLEN)
        for(i = 1; i <= ROWS; i++){
            #pragma acc loop worker vector reduction(max:dt)
            for(j = 1; j <= COLUMNS; j++){
                dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
                Temperature_last[i][j] = Temperature[i][j];
            }
        }
        
        // COMMUNICATION PHASE: send and receive ghost rows for next iteration
        #pragma acc wait(1)
        for (k = 0; k < 8; ++k) 
            if (requests[k] != MPI_REQUEST_NULL)
                MPI_Start(&requests[k]);
            
        #pragma acc wait(2)
        MPI_Iallreduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &request0);
        
        MPI_Waitall(8, requests, statuses);            

        #pragma acc update device(Temperature_last[0:1][1:COLUMNS]) if (has_up) async(1)
        #pragma acc update device(Temperature_last[ROWS+1:1][1:COLUMNS]) if (has_down) async(1)
        #pragma acc update device(Temperature_last[1:ROWS][0:1]) if (has_left) async(1)
        #pragma acc update device(Temperature_last[1:ROWS][COLUMNS+1:1]) if (has_right) async(1)            

        MPI_Wait(&request0, &status0);
        
        /*// periodically print test values 
        #pragma acc update host(Temperature_last[ROWS-5:][COLUMNS-5:]) wait(1)
        if((iteration % 256) == 0 && my_PE_num == PE_Grid[NODE_ROWS*PE_ROWS-2][NODE_COLS*PE_COLS-1]) {
            printf("PE %d Last ---------- Iteration number: %d ------------\n", my_PE_num, iteration);
            for(int k1 = 5; k1 >= -1; --k1) {
                for (int k2 = 5; k2 >=-1; --k2)
                    printf("[%d,%d]: %5.3f  ", ROWS-k1, COLUMNS-k2, Temperature_last[ROWS-k1][COLUMNS-k2]);
                printf("\n");
            }
            printf("dt: %f  Global dt: %f\n", dt, dt_global);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if((iteration % 256) == 0 && my_PE_num == PE_Grid[NODE_ROWS*PE_ROWS-1][NODE_COLS*PE_COLS-1]) {
            printf("PE %d Last ---------- Iteration number: %d ------------\n", my_PE_num, iteration);
            for(int k1 = 0; k1 <= 6; ++k1) {
                for (int k2 = 0; k2 <= 6; ++k2)
                    printf("[%d,%d]: %5.3f  ", k1, COLUMNS-5+k2, Temperature_last[k1][COLUMNS-5+k2]);
                printf("\n");
            }
            printf("dt: %f  Global dt: %f\n", dt, dt_global);
        }*/
        
        ++iteration;
        #pragma acc wait(1)
    }
    #pragma acc exit data copyout(Temperature) delete(Temperature_last)

    // Slightly more accurate timing and cleaner output 
    MPI_Barrier(MPI_COMM_WORLD);

    // That one PE output timing and output values
    if (my_PE_num == PE_Grid[WATCH_PE_ROW][WATCH_PE_COL]) {
        finish_time = MPI_Wtime();
        printf("Temperature[%d][%d] = %f\n", WATCHI, WATCHJ, Temperature[WATCH_LOCALI][WATCH_LOCALJ]);
	    printf("Max error at iteration %d was %f\n", iteration-1, dt_global);
	    printf("Total time was %f seconds.\n", finish_time - start_time);
    }

    MPI_Finalize();
    return 0;
}

void initialize(int const my_PE_row, int const my_PE_col) {

    int i, j;
    double row_start, col_start;
    double const row_increment = HEATER_MAX / COLUMNS_GLOBAL; 
    double const col_increment = HEATER_MAX / ROWS_GLOBAL;

    #pragma acc data present(Temperature_last)
    {
        #pragma acc parallel loop gang worker \
                device_type(nvidia) num_workers(ACC_WORKERS) vector_length(ACC_VECLEN)
        for(i = 0; i <= ROWS+1; ++i){
            #pragma acc loop worker vector
            for (j = 0; j <= COLUMNS+1; ++j){
                Temperature_last[i][j] = 0.0;
            }
        }
        // bottom row
        if (my_PE_row == NODE_ROWS*PE_ROWS-1) {
            row_start = my_PE_col * HEATER_MAX / (NODE_COLS*PE_COLS);    
            #pragma acc parallel loop async(1) gang worker vector \
                    device_type(nvidia) num_workers(ACC_WORKERS) vector_length(ACC_VECLEN) 
            for (j = 1; j <= COLUMNS; ++j) {
                Temperature_last[ROWS+1][j] = row_start + row_increment * j;
            }
        }
        // right column
        if (my_PE_col == NODE_COLS*PE_COLS-1) {
            col_start = my_PE_row * HEATER_MAX / (NODE_ROWS*PE_ROWS);
            #pragma acc parallel loop async(2) gang worker vector \
                    device_type(nvidia) num_workers(ACC_WORKERS) vector_length(ACC_VECLEN) 
            for (i = 1; i <= ROWS; ++i) {
                Temperature_last[i][COLUMNS+1] = col_start + col_increment * i;
            }
        }
    }
}
