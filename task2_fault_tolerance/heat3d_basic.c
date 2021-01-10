/*

Запуск из командной строки:
    1. Без параметров
    mpicc heat3d_basic.c
    mpirun --oversubscribe -n 1 a.out

    2. С параметрами t_steps и n
    mpicc heat3d_basic.c
    mpirun --oversubscribe -n 1 a.out 10 24

Для сравнения файлов на идентичность/различность:
    diff output_calc_basic.txt output_calc_basic.txt

 */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


// Иницилизация массивов A и B
static void init_array (int n, double A[n][n][n], double B[n][n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                A[i][j][k] = B[i][j][k] = (double) (i + j + (n - k)) * 10 / n;
            }
        }
    }
}


// Основная функция (последовательных) вычислений над массивами A и B
static void kernel_heat_3d(int t_steps, int n, double A[n][n][n], double B[n][n][n]) {

    for (int t = 1; t <= t_steps; t++) { // t_steps раз вычисляем

        // Массив B вычисляется через массив A
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])
                                 + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])
                                 + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1])
                                 + A[i][j][k] + 1.0; // добавлено +1.0, иначе программа не изменяет значений массивов
                }
            }
        }

        // Массив A вычисляется через массив B
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])
                                 + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])
                                 + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1])
                                 + B[i][j][k] + 2.0; // добавлено +2.0, иначе программа не изменяет значений массивов
                }
            }
        }

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

    printf("-----Sequential program heat3d-----\n");
    printf("\t-----START-----\n\n");

    int n = 120;       // размер измерения 3х-мерной кубической матрицы
    int t_steps = 100; // число шагов в kernel_heat_3d

    if (argc == 3) {   // если передали параметры t_steps и n в командную строку
        t_steps = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    printf("Число шагов в kernel_heat_3d - t = %d\n", t_steps);
    printf("Размер измерения 3х-мерной кубической матрицы - n = %d\n", n);

    double (*A)[n][n][n];
    double (*B)[n][n][n];

    // Выделение памяти для массивов A и B
    A = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));
    B = (double(*)[n][n][n])malloc ((n) * (n) * (n) * sizeof(double));

    init_array(n, *A, *B);                                              // иницилизация массивов A и B

    // Записываем инициализированные массивы A и B в файл "output_init_basic.txt"
    FILE *file_for_init_arrays;
    file_for_init_arrays = fopen("output_init_basic.txt", "w");
    print_array(n, *A, file_for_init_arrays);
    print_array(n, *B, file_for_init_arrays);
    fclose(file_for_init_arrays);

    double time_start = MPI_Wtime();                                    // время старта в секундах
    kernel_heat_3d(t_steps, n, *A, *B);                                 // основная функция (последовательная)
    double time_finish = MPI_Wtime();                                   // время финиша в секундах
    double time_execution = time_finish - time_start;                   // время выполнения
    printf("Execution time = %0.3lf seconds\n", time_execution); // вывод времени выполнения

    // Записываем вычисленные массивы A и B в файл "output_calc_basic.txt"
    FILE *file_for_calc_arrays;
    file_for_calc_arrays = fopen("output_calc_basic.txt", "w");
    print_array(n, *A, file_for_calc_arrays);
    print_array(n, *B, file_for_calc_arrays);
    fclose(file_for_calc_arrays);

    // Освобождение памяти, выделенной для массивов A и B
    free((void*)A);
    free((void*)B);

    printf("\n\t-----FINISH-----\n");

    return 0;
}