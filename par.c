#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <stdbool.h>
#include <mpi.h>


void init(FILE* in, bool* A, int X) {
        int tmp;
        for (size_t i = 0; i < X; ++i) {
                for (size_t j = 0; j < X; ++j) {
                        fscanf(in, "%d", &tmp);
                        A[i*X + j] = tmp;
                }
        }
}

void swap_fields(bool** A, bool** B) {
        bool* tmp = *A;
        *A = *B;
        *B = tmp;
}

int check_alive(bool* A, int x_cell, int X) {
        int number_live_cells = 0;
        bool* row1 = A;
        bool* row2 = A+X;
        bool* row3 = A+2*X;
        for (int i = x_cell - 1; i <= x_cell + 1; ++i) {
                number_live_cells += row1[(i + X) % X];
        }
        for (int i = x_cell - 1; i <= x_cell + 1; ++i) {
                number_live_cells += row2[(i + X) % X];
        }
        for (int i = x_cell - 1; i <= x_cell + 1; ++i) {
                number_live_cells += row3[(i + X) % X];
        }
        number_live_cells -= row2[x_cell];
        return number_live_cells;
}

int find_first_alive_cell(bool* field, int X, int Y) {
        for (int i = 0, end = X*Y; i < end; ++i) {
                if (field[i] == 1) return i;
        }
        return -1;
}

void print_field(bool* field, int X, int Y) {
        for (int i = 0; i < Y; ++i) {
                int ix = i*X;
                for (int j = 0; j < X; ++j) {
                        printf("%d ", field[ix + j]);
                }
                printf("\n");
        }
        printf("\n");
}

void print_result(FILE* out, bool* field, int X) {
        for (int i = 0; i < X; ++i) {
                int ix = i*X;
                for (int j = 0; j < X; ++j) {
                        fprintf(out, "%d ", field[ix + j]);
                }
                fprintf(out, "\n");
        }
        fprintf(out, "\n");
}

void get_sizes_and_offsets(int* sizes, int* offsets, int* parts, int size_MPI, int X) {
        for (size_t i = 0; i < size_MPI; ++i) {
                parts[i] = X / size_MPI;
                sizes[i] = parts[i]*X;
                if (i < X % size_MPI) {
                        sizes[i] += X;
                        parts[i] += 1;
                }
        }
        int offset = 0;
        for (size_t i = 0; i < size_MPI; ++i) {
                offsets[i] = offset;
                offset += sizes[i];
        }
}

void fill_row(bool* A, bool* B, int X) {
        for (int j = 0; j < X; ++j) {
                int number_live_cells = check_alive(A, j, X);
                if (number_live_cells == 3) {
                        B[j] = 1;
                } else if ((number_live_cells < 2) || (number_live_cells > 3)) {
                        B[j] = 0;
                } else {
                        B[j] = A[X+j];
                }
        }
}

int main(int argc, char** argv) {
        if (argc < 4) {
                printf("Not enough arguments\n");
                return -1;
        }
        int X = atol(argv[1]);

        FILE* in = fopen(argv[2], "r");
        if (in == NULL) {
                printf("Input file not found\n");
                return(-1);
        }

        FILE* out = fopen(argv[3], "wr");
        if (out == NULL) {
                printf("Output file not found\n");
                return(-1);
        }
        int number_of_iters = atol(argv[4]);

        MPI_Init(&argc, &argv);
        int rank_MPI;
        int size_MPI;
        MPI_Comm_size(MPI_COMM_WORLD, &size_MPI);
        MPI_Comm RING_COMM;

        int dims[1] = {size_MPI};
        int periods[1] = {1};
        MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 1, &RING_COMM);
        MPI_Comm_rank(RING_COMM, &rank_MPI);

        int up_rank, down_rank;
        MPI_Cart_shift(RING_COMM, 0, 1, &up_rank, &down_rank);
        if (size_MPI > X) {
                printf("Number of processes is more than X size of field!\n");
                return -1;
        }

        double start;
        double end;
        bool* A = NULL;
        bool* A_part = NULL;
        bool* B_part = NULL;

        int* sizes = (int*)calloc(size_MPI, sizeof(int));
        int* offsets = (int*)calloc(size_MPI, sizeof(int));
        int* parts = (int*)calloc(size_MPI, sizeof(int));
        get_sizes_and_offsets(sizes, offsets, parts, size_MPI, X);

        if (rank_MPI == 0) {
                for (int i = 0; i < size_MPI; ++i) {
                        printf("%d ", parts[i]);
                }
                printf("\n");
                A = (bool*)calloc(X * X, sizeof(bool));
                init(in, A, X);
                start = MPI_Wtime();
        }
        A_part = (bool*)calloc(X * (sizes[rank_MPI]+2), sizeof(bool));
        B_part = (bool*)calloc(X * (sizes[rank_MPI]+2), sizeof(bool));
        if (A_part == NULL || B_part == NULL) {
                printf("Not enough memory");
                if (A_part != NULL) free(A_part);
                if (B_part != NULL) free(B_part);
                MPI_Finalize();
                return(-1);
        }

        MPI_Scatterv(A, sizes, offsets, MPI_C_BOOL, A_part+X, sizes[rank_MPI]+2*X, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        if (rank_MPI == 0) print_field(A, X, X);

        MPI_Request request_send_up;
        MPI_Request request_send_down;
        MPI_Request request_recv_up;
        MPI_Request request_recv_down;

        for (int k = 0; k < number_of_iters; ++k) {
                MPI_Isend(A_part+X, X, MPI_C_BOOL, up_rank, rank_MPI+100, RING_COMM, &request_send_up);
                MPI_Isend(A_part + X*(parts[rank_MPI]), X, MPI_C_BOOL, down_rank, rank_MPI, RING_COMM, &request_send_down);
                MPI_Irecv(A_part, X, MPI_C_BOOL, up_rank, up_rank, RING_COMM, &request_recv_up);
                MPI_Irecv(A_part+X*(parts[rank_MPI]+1), X, MPI_C_BOOL, down_rank, down_rank+100, RING_COMM, &request_recv_down);

                int is_empty = find_first_alive_cell(A_part, X, parts[rank_MPI]+2);
                if (is_empty == -1) {
                        memset(B_part+2*X, 0, X*(parts[rank_MPI]-1)*sizeof(bool));
                } else {
                        for (int i = 1; i < parts[rank_MPI] - 1; ++i) {
                                fill_row(A_part+i*X, B_part+(1+i)*X, X);
                        }
                }

                MPI_Wait(&request_send_up, MPI_STATUS_IGNORE);
                MPI_Wait(&request_recv_up, MPI_STATUS_IGNORE);
                fill_row(A_part, B_part+X, X);

                MPI_Wait(&request_send_down, MPI_STATUS_IGNORE);
                MPI_Wait(&request_recv_down, MPI_STATUS_IGNORE);
                fill_row(A_part+(parts[rank_MPI]-1)*X, B_part+parts[rank_MPI]*X, X);

                swap_fields(&A_part, &B_part);
        }
        MPI_Gatherv(A_part+X, parts[rank_MPI]*X, MPI_C_BOOL, A, sizes, offsets, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        if (rank_MPI == 0) {
                end = MPI_Wtime();
                //print_field(A, X, X);
                print_result(out, A, X);
                printf("Time of computation: %lf\n", end - start);
                if (A != NULL) free(A);
        }

        if (A_part != NULL) free(A_part);
        if (B_part != NULL) free(B_part);

        MPI_Comm_free(&RING_COMM);
        MPI_Finalize();
        return 0;
}
