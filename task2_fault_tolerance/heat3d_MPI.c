/*

Запуск из командной строки:
    1. Без параметров
    mpicc heat3d_MPI.c
    mpirun --oversubscribe -n 7 a.out

    2. С параметрами t_steps и n
    mpicc heat3d_MPI.c
    mpirun --oversubscribe -n 7 a.out 10 24

Для сравнения файлов на идентичность/различность:
    diff output_calc_basic.txt output_calc_basic.txt

 */




#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define mpi_printf if (rank == 0) printf // чтобы в консоль выводил только 0й процесс
#define MSG_PLANE 111                    // тэг для сообщения пересылки плоскостей
#define RESULTS 777                      // тэг для сообщения при сборе результатов вычислений


// Иницилизация массивов A и B (для каждого процесса выделенно n_plane плоскостей массивов)
static void init_array(int rank, int num_processes, int n, int n_plane,
                        double A[n_plane + 2][n][n], double B[n_plane + 2][n][n]) {
    int start_plane = (rank - 1) * n / (num_processes - 1); // номер первой плоскости в распоряжении процесса
    for (int i = 1; i <= n_plane; i++) {
        int i_global = start_plane + i - 1;
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                A[i][j][k] = B[i][j][k] = (double) (i_global + j + (n - k)) * 10 / n;
            }
        }
    }
}


// Основная функция (MPI - параллельных) вычислений над массивами A и B
static void kernel_heat_3d_parallel(int rank, int num_processes, int t_steps, int n, int n_plane,
                                    double A[n_plane + 2][n][n], double B[n_plane + 2][n][n],
                                    MPI_Request req[4], MPI_Status status[4]) {

    for (int t = 1; t <= t_steps; t++) { // t_steps раз вычисляем

        MPI_Request request_isend1, request_isend2;
        MPI_Status status_recv1, status_recv2;

        // Массив B вычисляется через массив A
        if (rank != 1) {
            MPI_Isend(&A[1][0][0], n * n, MPI_DOUBLE, rank - 1, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend1);
        }
        if (rank != num_processes - 1) {
            MPI_Isend(&A[n_plane][0][0], n * n, MPI_DOUBLE, rank + 1, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend2);
        }
        if (rank != 1) {
            MPI_Recv(&A[0][0][0], n * n, MPI_DOUBLE, rank - 1, MSG_PLANE,
                     MPI_COMM_WORLD, &status_recv1);
        }
        if (rank != num_processes - 1) {
            MPI_Recv(&A[n_plane + 1][0][0], n * n, MPI_DOUBLE, rank + 1, MSG_PLANE,
                     MPI_COMM_WORLD, &status_recv2);
        }

        for (int i = 1; i <= n_plane; i++) {
            if ((i == 1 && rank == 1) || (i == n_plane && rank == num_processes - 1)) {
                continue;
            }
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    double residue;
                    B[i][j][k] = A[i][j][k] * 0.25;
                    residue = A[i + 1][j][k] + A[i - 1][j][k]
                              + A[i][j + 1][k] + A[i][j - 1][k]
                              + A[i][j][k + 1] + A[i][j][k - 1];
                    residue *= 0.125;
                    B[i][j][k] += residue + 1.0;
                }
            }
        }

        // Массив A вычисляется через массив B
        if (rank != 1) {
            MPI_Isend(&B[1][0][0], n * n, MPI_DOUBLE, rank - 1, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend1);
        }
        if (rank != num_processes - 1) {
            MPI_Isend(&B[n_plane][0][0], n * n, MPI_DOUBLE, rank + 1, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend2);
        }
        if (rank != 1) {
            MPI_Recv(&B[0][0][0], n * n, MPI_DOUBLE, rank - 1, MSG_PLANE,
                     MPI_COMM_WORLD, &status_recv1);
        }
        if (rank != num_processes - 1) {
            MPI_Recv(&B[n_plane + 1][0][0], n * n, MPI_DOUBLE, rank + 1, MSG_PLANE,
                     MPI_COMM_WORLD, &status_recv2);
        }

        for (int i = 1; i <= n_plane; i++) {
            if ((i == 1 && rank == 1) || (i == n_plane && rank == num_processes - 1)) {
                continue;
            }
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    double residue;
                    A[i][j][k] = B[i][j][k] * 0.25;
                    residue = B[i + 1][j][k] + B[i - 1][j][k]
                              + B[i][j + 1][k] + B[i][j - 1][k]
                              + B[i][j][k + 1] + B[i][j][k - 1];
                    residue *= 0.125;
                    A[i][j][k] += residue + 2.0;
                }
            }
        }

    }

}

// Рабочие процессы отправляют результаты процессу-мастеру
static void send_results(int n, int n_plane, double A[n_plane + 2][n][n], double B[n_plane + 2][n][n]) {
    MPI_Send(&A[1][0][0], n_plane * n * n, MPI_DOUBLE, 0, RESULTS, MPI_COMM_WORLD);
    MPI_Send(&B[1][0][0], n_plane * n * n, MPI_DOUBLE, 0, RESULTS, MPI_COMM_WORLD);
}

// Процесс-мастер получает результаты от рабочих процессов
static void receive_results(int num_processes, int n, int n_plane, double A_all[n][n][n], double B_all[n][n][n]) {
    MPI_Status status_result;
    for (int i = 1; i < num_processes; i++) {
        int start_recv_plane = ((i - 1) * n) / (num_processes - 1);
        MPI_Recv(&A_all[start_recv_plane][0][0], n_plane * n * n,
                 MPI_DOUBLE, i, RESULTS, MPI_COMM_WORLD, &status_result);
    }
    for (int i = 1; i < num_processes; i++) {
        int start_recv_plane = ((i - 1) * n) / (num_processes - 1);
        MPI_Recv(&B_all[start_recv_plane][0][0], n_plane * n * n,
                 MPI_DOUBLE, i, RESULTS, MPI_COMM_WORLD, &status_result);
    }
}


// Печать массива
static void print_array(int n, double A[n][n][n], FILE *file_for_arrays) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fprintf(file_for_arrays, "%0.2lf ", A[i][j][k]);
            }
            fprintf(file_for_arrays, "\n");
        }
    }
}


int main(int argc, char** argv) {

    int num_processes, rank;

    MPI_Init(&argc, &argv);                        // инициализация параллельной части приложения
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes); // определение общего числа параллельных процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);          // определение номера процесса (от 0 до num_processes - 1)

    mpi_printf("-----MPI-Parallel program heat3d-----\n");
    mpi_printf("\t-----START-----\n\n");

    int n = 120;                                   // размер измерения 3х-мерной кубической матрицы
    int t_steps = 100;                             // число шагов в kernel_heat_3d

    if (argc == 3) {                               // если передали параметры t_steps и n в командную строку
        t_steps = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    mpi_printf("Число шагов в kernel_heat_3d - t = %d\n", t_steps);
    mpi_printf("Размер измерения 3х-мерной кубической матрицы - n = %d\n", n);
    mpi_printf("Число процессов - num_processes = %d\n", num_processes);

    if (num_processes > n) {
        mpi_printf("Процессов больше, чем необходимо для параллельной работы программы.\n");
        mpi_printf("Пожалуйста, задайте число процессов < n.\n");
        MPI_Finalize(); // завершение параллельной части приложения
        return 0;
    }

    if (n % (num_processes - 1) != 0) {
        mpi_printf("Невозможно разделить плоскости по рабочим процессам\n");
        mpi_printf("Пожалуйста, задайте число рабочих процессов так, чтобы n делилось на него.\n");
        MPI_Finalize(); // завершение параллельной части приложения
        return 0;
    }

    int n_plane = n / (num_processes - 1); // число плоскостей в распоряжении процессов

    if (rank == 0) {
        double (*A_all)[n][n][n];
        double (*B_all)[n][n][n];

        // Выделение памяти для массивов A_all и B_all
        A_all = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
        B_all = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));

        receive_results(num_processes, n, n_plane, *A_all, *B_all);

        // Записываем вычисленные массивы A_all и B_all в файл "output_calc_MPI.txt"
        FILE *file_for_calc_arrays;
        file_for_calc_arrays = fopen("output_calc_MPI.txt", "w");
        print_array(n, *A_all, file_for_calc_arrays);
        print_array(n, *B_all, file_for_calc_arrays);
        fclose(file_for_calc_arrays);

        free((void *) A_all);
        free((void *) B_all);
    }

    if (rank != 0) {
        MPI_Request req[4];
        MPI_Status status[4];

        double (*A)[n_plane + 2][n][n];
        double (*B)[n_plane + 2][n][n];

        // Выделение памяти для массивов A и B - частей A_all и B_all
        A = (double (*)[n_plane + 2][n][n]) malloc((n_plane + 2) * (n) * (n) * sizeof(double));
        B = (double (*)[n_plane + 2][n][n]) malloc((n_plane + 2) * (n) * (n) * sizeof(double));

        init_array(rank, num_processes, n, n_plane, *A, *B); // иницилизация массивов A и B
        kernel_heat_3d_parallel(rank, num_processes, t_steps, n, n_plane, *A, *B, req, status); // основная функция
        send_results(n, n_plane, *A, *B);

        // Освобождение памяти, выделенной для массивов A и B
        free((void *) A);
        free((void *) B);
    }

    mpi_printf("\n\t-----FINISH-----\n");

    MPI_Finalize(); // завершение параллельной части приложения
    return 0;
}

