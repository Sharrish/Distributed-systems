/*

Запуск из командной строки:
    1. Без параметров
    source dockervars.sh
    mpicc heat3d_MPI_FT.c
    mpirun --oversubscribe -n 8 a.out

    2. С параметрами t_steps и n
    source dockervars.sh
    mpicc heat3d_MPI_FT.c
    mpirun --oversubscribe -n 8 a.out 10 24

Для сравнения файлов на идентичность/различность:
    diff output_calc_basic.txt output_calc_MPI_FT.txt

 */


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h> // для использования bool
#include <time.h>    // time() для srand()
#include <signal.h> // для raise()
#include <sys/stat.h> // для stat и mkdir
#include <unistd.h> // для rmdir
#include "mpi.h"

#define mpi_printf if (rank == 0) printf // чтобы в консоль выводил только 0й процесс
#define MSG_PLANE 111     // тэг для сообщения пересылки плоскостей
#define RESULTS 777     // тэг для сообщения при сборе результатов вычислений
#define RECOVERY 666     // тэг для сообщения о запуске восстановления работы умершего процесса
#define TEST_DEAD 228
#define N_BLOCKS 8 // кол-во контрольных точек
#define RANDOM_SEED 3000

int rank_to_damage, block_to_damage, t_step_to_damage;

static void try_to_suicide(int rank, int curr_block, int curr_t_step) {
    if (rank == rank_to_damage) {
        if (curr_block == block_to_damage) {
            if (curr_block == 0 || curr_block == N_BLOCKS || curr_t_step == t_step_to_damage) {
                printf("Процесс №%d умер в блоке №%d на t-шаге №%d\n", rank, curr_block, curr_t_step);
                raise(SIGTERM);
            }
        }
    }
}

// Сохранение контрольной точки - запись в файл
static void save_control_point(int rank, int n, int n_plane,
                               double A[n_plane + 2][n][n], double B[n_plane + 2][n][n],
                               int last_t_step, int last_block) {
    char filename[30];
    // Задаем имя файла для контрольных точек
    snprintf(filename, 30, "control_point_%d_%d_%d.txt", rank, last_t_step, last_block);
    FILE *control_point_file = fopen(filename, "w");

    // Записываем массив A в файл
    for (int i = 0; i < n_plane + 2; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fprintf(control_point_file, "%0.16lf ", A[i][j][k]);
            }
            fprintf(control_point_file, "\n");
        }
    }

    // Записываем массив B в файл
    for (int i = 0; i < n_plane + 2; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fprintf(control_point_file, "%0.16lf ", B[i][j][k]);
            }
            fprintf(control_point_file, "\n");
        }
    }

    fclose(control_point_file);
}


// Восстановление данных из последней контрольной точки - чтение из файла
static void load_control_point(int rank, int n, int n_plane,
                               double A[n_plane + 2][n][n], double B[n_plane + 2][n][n],
                               int *last_t_step, int *last_block) {
    char filename[30];
    int max_t_step = *last_t_step, max_block = *last_block;
    // Задаем имя файла для контрольных точек
    bool flag_founded_control_point_file = false;
    for (*last_t_step = max_t_step; *last_t_step >= 1 && !flag_founded_control_point_file; (*last_t_step)--) {
        for(*last_block = max_block; *last_block >= 0 && !flag_founded_control_point_file; (*last_block)--) {
            snprintf(filename, 30, "control_point_%d_%d_%d.txt", rank, *last_t_step, *last_block);
            FILE *last_control_point_file = fopen(filename, "r");
            if (last_control_point_file != NULL) {
                flag_founded_control_point_file = true;
                (*last_t_step)++;
                (*last_block)++;
                fclose(last_control_point_file);
            }
        }
    }
    if (!flag_founded_control_point_file) { // если не нашли контрольной точки
        *last_t_step = 1;
        *last_block = -1;
        return;
    }
    snprintf(filename, 30, "control_point_%d_%d_%d.txt", rank, *last_t_step, *last_block);
    FILE *control_point_file = fopen(filename, "r");

    // Читаем массив A из файла
    for (int i = 0; i < n_plane + 2; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fscanf(control_point_file, "%lf", &A[i][j][k]);
            }
        }
    }

    // Читаем массив B из файла
    for (int i = 0; i < n_plane + 2; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fscanf(control_point_file, "%lf", &B[i][j][k]);
            }
        }
    }

    fclose(control_point_file);

    printf("Контрольная точка успешно загружена! t_step %d block %d\n", *last_t_step, *last_block);
}


// Иницилизация массивов A и B (для каждого процесса выделенно n_plane плоскостей массивов)
static void init_array(int rank, int active_processes, int n, int n_plane,
                       double A[n_plane + 2][n][n], double B[n_plane + 2][n][n]) {
    int start_plane = (rank - 1) * n / (active_processes - 1); // номер первой плоскости в распоряжении процесса
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
static void kernel_heat_3d_parallel(int rank, int active_processes, int t_steps, int n, int n_plane,
                                    double A[n_plane + 2][n][n], double B[n_plane + 2][n][n],
                                    int start_t_step, int start_block) {
    int t = start_t_step;
    int rank_broken = -1; // номер умершего процесса
    int rc; // return code
    MPI_Request request_isend1, request_isend2, request_recovery;
    MPI_Status status_recv1, status_recv2;
    switch (start_block + 1) {
        case 0:
        case 1:
            goto block_A_prev;
        case 2:
            goto block_A_next;
        case 3:
            goto block_A_calc;
        case 4:
            goto block_B_prev;
        case 5:
            goto block_B_next;
        case 6:
            goto block_B_calc;
        default:
            goto block_t_step_end;
    }
    for (; t <= t_steps; t++) { // t_steps раз вычисляем

        // Массив B вычисляется через массив A
block_A_prev: // 1
        try_to_suicide(rank, 1, t);
        if (rank != 1) {
            MPI_Isend(&A[1][0][0], n * n, MPI_DOUBLE, (rank - 1 != rank_broken) ? (rank - 1) : active_processes, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend1);
            rc = MPI_Recv(&A[0][0][0], n * n, MPI_DOUBLE, (rank - 1 != rank_broken) ? (rank - 1) : active_processes, MSG_PLANE,
                     MPI_COMM_WORLD, &status_recv1);
            if (rc != 0) { // если сломался процесс, с которым взаимодействуем
                rank_broken = rank - 1;
                MPI_Isend(&rank_broken, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);
                printf("Процесс №%d заметил, что процесс №%d умер.\n", rank, rank - 1);
                printf("Сообщение с номером умершего процесса отправлено резервному процессу - №%d.\n", active_processes);
                goto block_A_prev;
            }
        }
        save_control_point(rank, n, n_plane, A, B, t, 1);

block_A_next: // 2
        try_to_suicide(rank, 2, t);
        if (rank != active_processes - 1) {
            MPI_Isend(&A[n_plane][0][0], n * n, MPI_DOUBLE, (rank + 1 != rank_broken) ? (rank + 1) : active_processes, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend2);
            rc = MPI_Recv(&A[n_plane + 1][0][0], n * n, MPI_DOUBLE, (rank + 1 != rank_broken) ? (rank + 1) : active_processes, MSG_PLANE,
                     MPI_COMM_WORLD, &status_recv2);
            if (rc != 0) { // если сломался процесс, с которым взаимодействуем
                rank_broken = rank + 1;
                MPI_Isend(&rank_broken, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);
                printf("Процесс №%d заметил, что процесс №%d умер.\n", rank, rank + 1);
                printf("Сообщение с номером умершего процесса отправлено резервному процессу - №%d.\n", active_processes);
                goto block_A_next;
            }
        }
        save_control_point(rank, n, n_plane, A, B, t, 2);

block_A_calc: // 3
        try_to_suicide(rank, 3, t);
        for (int i = 1; i <= n_plane; i++) {
            if ((i == 1 && rank == 1) || (i == n_plane && rank == active_processes - 1)) {
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
        save_control_point(rank, n, n_plane, A, B, t, 3);

        // Массив A вычисляется через массив B
block_B_prev: // 4
        try_to_suicide(rank, 4, t);
        if (rank != 1) {
            MPI_Isend(&B[1][0][0], n * n, MPI_DOUBLE, (rank - 1 != rank_broken) ? (rank - 1) : active_processes, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend1);
            rc = MPI_Recv(&B[0][0][0], n * n, MPI_DOUBLE, (rank - 1 != rank_broken) ? (rank - 1) : active_processes, MSG_PLANE,
                          MPI_COMM_WORLD, &status_recv1);
            if (rc != 0) { // если сломался процесс, с которым взаимодействуем
                rank_broken = rank - 1;
                MPI_Isend(&rank_broken, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);
//                printf("Процесс №%d заметил, что процесс №%d умер.\n", rank, rank - 1);
//                printf("Сообщение с номером умершего процесса отправлено резервному процессу - №%d.\n", active_processes);
                goto block_B_prev;
            }
        }
        save_control_point(rank, n, n_plane, A, B, t, 4);

block_B_next: // 5
        try_to_suicide(rank, 5, t);
        if (rank != active_processes - 1) {
            MPI_Isend(&B[n_plane][0][0], n * n, MPI_DOUBLE, (rank + 1 != rank_broken) ? (rank + 1) : active_processes, MSG_PLANE,
                      MPI_COMM_WORLD, &request_isend2);
            rc = MPI_Recv(&B[n_plane + 1][0][0], n * n, MPI_DOUBLE, (rank + 1 != rank_broken) ? (rank + 1) : active_processes, MSG_PLANE,
                          MPI_COMM_WORLD, &status_recv2);
            if (rc != 0) { // если сломался процесс, с которым взаимодействуем
                rank_broken = rank + 1;
                MPI_Isend(&rank_broken, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);
                printf("Процесс №%d заметил, что процесс №%d умер.\n", rank, rank + 1);
                printf("Сообщение с номером умершего процесса отправлено резервному процессу - №%d.\n", active_processes);
                goto block_B_next;
            }
        }
        save_control_point(rank, n, n_plane, A, B, t, 5);

block_B_calc: // 6
        try_to_suicide(rank, 6, t);
        for (int i = 1; i <= n_plane; i++) {
            if ((i == 1 && rank == 1) || (i == n_plane && rank == active_processes - 1)) {
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
        save_control_point(rank, n, n_plane, A, B, t, 6);

block_t_step_end: // 7
        try_to_suicide(rank, 7, t);
    }

}

// Рабочие процессы отправляют результаты процессу-мастеру
static void send_results(int rank, int n, int n_plane, double A[n_plane + 2][n][n], double B[n_plane + 2][n][n]) {
    try_to_suicide(rank, 8, 0);
    MPI_Send(&A[1][0][0], n_plane * n * n, MPI_DOUBLE, 0, RESULTS, MPI_COMM_WORLD);
    MPI_Send(&B[1][0][0], n_plane * n * n, MPI_DOUBLE, 0, RESULTS, MPI_COMM_WORLD);
}

// Процесс-мастер получает результаты от рабочих процессов
static void receive_results(int active_processes, int n, int n_plane, double A_all[n][n][n], double B_all[n][n][n]) {
    MPI_Request request_recovery;
    int rc;
    int rank_broken = -1;
    for (int i = 1; i < active_processes; i++) {
        int start_recv_plane = ((i - 1) * n) / (active_processes - 1);
block_A_recv:
        rc = MPI_Recv(&A_all[start_recv_plane][0][0], n_plane * n * n,
                 MPI_DOUBLE, (i != rank_broken) ? i : active_processes, RESULTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rc != 0) {
            rank_broken = i;
            MPI_Isend(&rank_broken, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);
            goto block_A_recv;
        }
    }
    for (int i = 1; i < active_processes; i++) {
        int start_recv_plane = ((i - 1) * n) / (active_processes - 1);
block_B_recv:
        rc = MPI_Recv(&B_all[start_recv_plane][0][0], n_plane * n * n,
                 MPI_DOUBLE, (i != rank_broken) ? i : active_processes, RESULTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rc != 0) {
            rank_broken = i;
            MPI_Isend(&rank_broken, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);
            goto block_B_recv;
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

    int num_processes, rank, active_processes;

    MPI_Init(&argc, &argv);                        // инициализация параллельной части приложения
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes); // определение общего числа параллельных процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);          // определение номера процесса (от 0 до num_processes - 1)
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); // возврат ошибок

    mpi_printf("-----MPI-FT-Parallel program heat3d-----\n");
    mpi_printf("\t-----START-----\n\n");

    int n = 120;                                   // размер измерения 3х-мерной кубической матрицы
    int t_steps = 100;                             // число шагов в kernel_heat_3d

    if (argc == 3) {                               // если передали параметры t_steps и n в командную строку
        t_steps = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    active_processes = num_processes - 1;
    srand(RANDOM_SEED);  // инициализация начала последовательности, генерируемой функцией rand()
    rank_to_damage = 3;//rand() % (active_processes - 1) + 1;
    block_to_damage = 4; //rand() % N_BLOCKS;
    t_step_to_damage = (rand() % (t_steps - 1)) + 1;
    printf("DAMGE rank %d block %d step %d\n", rank_to_damage, block_to_damage, t_step_to_damage);
    mpi_printf("Число шагов в kernel_heat_3d - t = %d\n", t_steps);
    mpi_printf("Размер измерения 3х-мерной кубической матрицы - n = %d\n", n);
    mpi_printf("Число процессов - num_processes = %d\n", num_processes);

    if (active_processes > n) {
        mpi_printf("Процессов больше, чем необходимо для параллельной работы программы.\n");
        mpi_printf("Пожалуйста, задайте число процессов < n.\n");
        MPI_Finalize(); // завершение параллельной части приложения
        return 0;
    }

    if (n % (active_processes - 1) != 0) {
        mpi_printf("Невозможно разделить плоскости по рабочим процессам\n");
        mpi_printf("Пожалуйста, задайте число рабочих процессов так, чтобы n делилось на него.\n");
        MPI_Finalize(); // завершение параллельной части приложения
        return 0;
    }

    int n_plane = n / (active_processes - 1); // число плоскостей в распоряжении процессов

    // Работа процесса-мастера
    if (rank == 0) {

        double (*A_all)[n][n][n];
        double (*B_all)[n][n][n];

        // Выделение памяти для массивов A_all и B_all
        A_all = (double (*)[n][n][n]) malloc((n) * (n) * (n) * sizeof(double));
        B_all = (double (*)[n][n][n]) malloc((n) * (n) * (n) * sizeof(double));

        // Собираем результаты вычислений от рабочих процессов в единые массивы
        receive_results(active_processes, n, n_plane, *A_all, *B_all);

        // Записываем вычисленные массивы A_all и B_all в файл "output_calc_MPI.txt"
        FILE *file_for_calc_arrays;
        file_for_calc_arrays = fopen("output_calc_MPI_FT.txt", "w");
        print_array(n, *A_all, file_for_calc_arrays);
        print_array(n, *B_all, file_for_calc_arrays);
        fclose(file_for_calc_arrays);

        free((void *) A_all);
        free((void *) B_all);

        // Отправляем резервному процессу, что никто не умер и вычисления прошли успешно
        int success = 0; // никто не умер
        MPI_Request request_recovery;
        MPI_Isend(&success, 1, MPI_INT, active_processes, RECOVERY, MPI_COMM_WORLD, &request_recovery);

    } else { // работа рабочих процессов

        double (*A)[n_plane + 2][n][n];
        double (*B)[n_plane + 2][n][n];

        // Выделение памяти для массивов A и B - частей A_all и B_all
        A = (double (*)[n_plane + 2][n][n]) malloc((n_plane + 2) * (n) * (n) * sizeof(double));
        B = (double (*)[n_plane + 2][n][n]) malloc((n_plane + 2) * (n) * (n) * sizeof(double));

        int start_t_step = 1, start_block = 0;

        if (rank == num_processes - 1) { // резервный процесс
            printf("Резервный процесс - №%d\n", rank);

            // Пытаемся понять от какого процесса ждать сообщения о умершем процессе
            MPI_Status status_test;
            MPI_Request request_test[active_processes];
            int test[active_processes];
            for (int i = 0; i < active_processes; i++) {
                MPI_Irecv(&test[i], 1, MPI_INT, i, TEST_DEAD, MPI_COMM_WORLD, &request_test[i]);
            }
            int idx_failure = -1;
            MPI_Waitany(active_processes, request_test, &idx_failure, &status_test);
            // в результате в переменной idx_failure будет содержаться индекс поломавшегося процесса
            printf("RECOVERY rank %d broken\n", idx_failure);

            if (idx_failure == 0) { // никто не умер
                goto block_end_active_process;
            }

            // Определяем от кого ждать сообщения
            int sender;
            if (idx_failure == 1) {
                sender = idx_failure + 1;
            } else {
                sender = idx_failure - 1;
            }

            MPI_Request request_rank[2];
            MPI_Status status;
            int new_ranks[2];
            MPI_Irecv(&new_ranks[0], 1, MPI_INT, 0, RECOVERY, MPI_COMM_WORLD, &request_rank[0]); // для 8 блока
            MPI_Irecv(&new_ranks[1], 1, MPI_INT, sender, RECOVERY, MPI_COMM_WORLD, &request_rank[1]);
            int idx_success = -1;
            MPI_Waitany(2, request_rank, &idx_success, &status);
            int new_rank = new_ranks[idx_success];
            printf("Резервный процесс получил сообщение от процесса №%d, что нужно заменить процесс №%d!\n",
                           status.MPI_SOURCE, new_rank);
            printf("Резервный процесс присоединился к работе!\n");
            rank_to_damage = -1;
            block_to_damage = -1;
            t_step_to_damage = -1;
            start_t_step = t_steps;
            start_block = N_BLOCKS;
            load_control_point(new_rank, n, n_plane, *A, *B, &start_t_step, &start_block);
            rank = new_rank;
            switch (start_block + 1) {
                case 0:
                    goto block_init;
                default:
                    goto block_kernel;
            }
        }
block_init: // 0
        try_to_suicide(rank, 0, 0);
        init_array(rank, active_processes, n, n_plane, *A, *B); // иницилизация массивов A и B
        save_control_point(rank, n, n_plane, *A, *B, 1, 0);
block_kernel:
        kernel_heat_3d_parallel(rank, active_processes, t_steps, n, n_plane, *A, *B, start_t_step, start_block); // основная функция
        send_results(rank, n, n_plane, *A, *B);

block_end_active_process:
        // Освобождение памяти, выделенной для массивов A и B
        free((void *) A);
        free((void *) B);
    }

    mpi_printf("\n\t-----FINISH-----\n");

    MPI_Finalize(); // завершение параллельной части приложения
    return 0;
}

