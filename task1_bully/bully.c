/*
    Для запуска в терминале:
    mpicc bully.c -o bully
    mpirun -np 25 --oversubscribe ./bully [пользовательские аргументы командной строки]
        (где 25 - число процессов)
        (Работает с --oversubscribe, а без него mpirun не позволяет использовать больше процессов, чем число ядер.)
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>    // time() для srand()
#include <stdbool.h> // для использования bool
#include "mpi.h"


#define ELECTION 111     // тэг для сообщения о выборах/голосовании
#define OK 222           // тэг для сообщения "ОК" в ответ задире на призыв к участию в голосовании
#define COORDINATOR 333  // тэг для сообщения, которое разошлет процесс, победивший в выборах и ставший координатором


double get_uniform_rand(double a, double b) {
    // Получение случайного числа, из равномерного распределения на отрезке [a, b].
    return (double) rand() / RAND_MAX * (b - a) + a;
}

double work_init() {
    // Принятие решения, будет ли процесс рабочим и принимать участие в выборах координатора.
    double r = get_uniform_rand(0, 1);
    if (r < 0.5) {
        return true;
    }
    return false;
}

void print_not_working_processes(int start_bully, int rank, bool work_state) {
    // Печатаем все мертвые процессы, которые не участвуют в голосовании
    if (rank == start_bully) {
        printf("Нерабочие процессы перед голосованием: ");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (work_state == false) {
        printf("%d ", rank);
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == start_bully) {
        printf("\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_working_processes(int start_bully, int rank, bool work_state) {
    // Печатаем все живые процессы, которые участвуют в голосовании
    if (rank == start_bully) {
        printf("В голосовании участвуют процессы: ");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (work_state == true) {
        printf("%d ", rank);
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == start_bully) {
        printf("\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {

    setvbuf( stdout, NULL, _IOLBF, BUFSIZ );
    setvbuf( stderr, NULL, _IOLBF, BUFSIZ );

    int size; // число параллельных процессов
    int rank; // номер процесса

    MPI_Init(&argc, &argv);               // инициализация параллельной части приложения
    MPI_Comm_size(MPI_COMM_WORLD, &size); // определение общего числа параллельных процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // определение номера процесса
    srand(time(NULL) * rank);  // инициализация начала последовательности, генерируемой функцией rand()

    if (size == 1) {  // если всего только один процесс, то он и является координатором
        printf("Координатором является процесс 0.\n");
        MPI_Finalize();  // завершение параллельной части приложения
        return 0;
    }

    int current_cordinator = size-1, start_bully = 0; // если не задать параметры командной строки
    if (argc == 3) {
        current_cordinator = atoi(argv[1]); // исходный координатор, который находится в нерабочем состоянии
                                                 // и перестал отвечать задире
        start_bully = atoi(argv[2]);        // изначальный задира, который инициирует голосование
    }

    // Выбор процессов, участвущих в голосовании
    bool work_state; // находится ли процесс в рабочем состоянии, чтобы участвовать в голосовании
    if (rank == start_bully) {
        work_state = true;
    } else if (rank == current_cordinator) {
        work_state = false;
    } else {
        work_state = work_init();
    }

    // Печатаем информацию о живых/мертвых процессах
    print_not_working_processes(start_bully, rank, work_state);
    MPI_Barrier(MPI_COMM_WORLD);
    print_working_processes(start_bully, rank, work_state);
    MPI_Barrier(MPI_COMM_WORLD);

    // Завершаем работу всех нерабочих процессов
    if (work_state == false) {
        MPI_Finalize();  // завершение параллельной части приложения
        return 0;
    }

    bool i_was_bully = false;
    bool i_am_bully;
    if (rank == start_bully) {
        i_am_bully = true;
    } else{
        i_am_bully = false;
    }

    MPI_Status status;
    MPI_Request request;

    bool election_go = true;
    while (election_go) {

        if (i_am_bully) {
            printf("Процесс %d - задира и начал голосование.\n", rank);
            fflush(stdout);
            for (int i = rank + 1; i < size; i++) { // отправляем приглашение на выборы всем старшим процессам
                MPI_Isend(&rank, 1, MPI_INT, i, ELECTION, MPI_COMM_WORLD, &request);
            }
            i_was_bully = true; // рассылал ли приглашение на голосование всем старшим процессам
        }

        // Принимаем сообщение для приглашения на выборы
        int election_msg;
        for (int i = 0; i < rank; i++) { // пытаемся принять приглашение на голосование от всех младших
            MPI_Irecv(&election_msg, 1, MPI_INT, i, ELECTION, MPI_COMM_WORLD, &request);
            int flag = false;
            double start_wait = MPI_Wtime();
            while (!flag) {  // пока сообщение "OK" не получено, flag == true -> получили сообщение
                MPI_Test(&request, &flag, &status); // Проверка завершенности асинхронной процедуры MPI_Irecv
                if (MPI_Wtime() - start_wait >= 1) {  // ждем 1 секунду
                    // Ничего не получили, поэтому отменяем неблокирующий Recieve
                    MPI_Cancel(&request);
                    MPI_Request_free(&request);
                    break;
                }
            }
            if (flag == true) { // если действительно получили сообщение-приглашение на выборы
                i_am_bully = true; // теперь этот процесс тоже станет задирой
                // Отправляем сообщение "OK" о готовности в выборах задире обратно
                MPI_Isend(&rank, 1, MPI_INT, status.MPI_SOURCE, OK, MPI_COMM_WORLD, &request);
                printf("Процесс %d получил приглашение на выборы от процесса %d и отправил ему OK.\n", rank,
                       status.MPI_SOURCE);
            }
        }

        if (i_am_bully == false) { // Если процесс имел номер меньше, чем у start_bully, то он не получил приглашение на
                                   //выборы и никогда не получит.
            break;
        }

        // Ждем "ОК" от процессов, которых приглашали на голосование
        if (i_was_bully == true) { // рассылал приглашение на голосование всем старшим процессам и жду от них "ОК"
            int ok_msg;
            // Неблокирующий Recieve для сообщения "OK" от всех процессов
            MPI_Irecv(&ok_msg, 1, MPI_INT, MPI_ANY_SOURCE, OK, MPI_COMM_WORLD, &request);
            int flag = false;
            double start_wait = MPI_Wtime();
            while (!flag) {  // пока сообщение "OK" не получено, flag == true -> получили сообщение
                MPI_Test(&request, &flag, &status); // Проверка завершенности асинхронной процедуры MPI_Irecv
                if (MPI_Wtime() - start_wait >= 1) {  // ждем 1 секунду
                    // Ничего не получили, поэтому отменяем неблокирующий Recieve
                    MPI_Cancel(&request);
                    MPI_Request_free(&request);
                    break;
                }
            }
            if (flag == true) { // процесс получил сообщение "ОК" от кого-то
                election_go = false;  // выборы для этого процесса заканчиваются
                printf("Процесс %d получил сообщение ОК от процесса %d и заканчивает участие в выборах.\n", rank, status.MPI_SOURCE);
            } else { // процесс не получил сообщение "ОК" от кого-то
                election_go = false;  // выборы для этого процесса заканчиваются
                current_cordinator = rank;
                // Отправляем всем остальным процессам сообщение "COORDINATOR".
                for (int i = 0; i < size; i++) {
                    if (i != rank) {
                        MPI_Isend(&rank, 1, MPI_INT, i, COORDINATOR, MPI_COMM_WORLD, &request);
                    }
                }
                printf("Процесс %d победил в выборах и стал новым координатором.\n", rank);
            }
        }
    }

    // Все процессы кроме нового координатора получают сообщение от него и обновляют current_cordinator
    int coordinator_msg;
    if (current_cordinator != rank) {
        MPI_Recv(&coordinator_msg, 1, MPI_INT, MPI_ANY_SOURCE, COORDINATOR, MPI_COMM_WORLD, &status);
        current_cordinator = coordinator_msg;
        printf("Процесс %d получил сообщение - новый координатор процесс %d\n", rank, current_cordinator);
        fflush(stdout);
    }

    MPI_Finalize(); // завершение параллельной части приложения
    return 0;
}