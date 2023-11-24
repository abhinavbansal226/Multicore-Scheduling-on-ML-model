//This progrsm simulates the multicore scheduling on SVM machine learning model
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <ctime>
#include <sys/wait.h> 
using namespace std;

struct Point {
    double x, y;
    int label;
};

struct SVMModel {
    double w1, w2; // Weights for the linear SVM
    double b;      // Bias term
};

// SVM training function (basic linear SVM)
SVMModel trainSVM(vector<Point>& data) {
    SVMModel model;
    
    double learningRate = 0.1;
    int epochs = 1000;

    // Initialize weights and bias
    model.w1 = 0.0;
    model.w2 = 0.0;
    model.b = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (const Point& point : data) {
            double margin = point.label * (model.w1 * point.x + model.w2 * point.y + model.b);

            if (margin < 1) {
                model.w1 += learningRate * (point.label * point.x - 2 * model.w1);
                model.w2 += learningRate * (point.label * point.y - 2 * model.w2);
                model.b += learningRate * point.label;
            }
        }
    }

    return model;
}

// Predict the class of a new point using the trained SVM model
int predictClass(const SVMModel& model, double x, double y) {
    double margin = model.w1 * x + model.w2 * y + model.b;
    return (margin >= 0) ? 1 : -1;
}

int main() {
    // Sample data
    vector<Point> data = {
        {1, 12, 1},
        {2, 5, -1},
        {5, 3, -1},
        {3, 2, -1},
        {3, 6, 1},
        {1.5, 9, -1},
        {7, 2, -1},
        {6, 1, -1},
        {3.8, 3, -1},
        {3, 10, 1},
        {5.6, 4, -1},
        {4, 2, -1},
        {3.5, 8, 1},
        {2, 11, 1},
        {2, 5, -1},
        {2, 9, 1},
        {1, 7, 1}
    };

    // Train the SVM model
    SVMModel model = trainSVM(data);
    pid_t pid;
 
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN); // Get the number of available CPU cores

    clock_t start_time = clock();  // Record the start time

    for (int core = 0; core < num_cores; core++) {
        pid = fork();

        if (pid == 0) {
            // This is the child process
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(core, &mask);

            if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
                perror("sched_setaffinity");
            }

            // Now each child process can work on a specific core
            printf("Process ID (PID): %d\n", getpid());
            printf("CPU Core Number: %d\n", sched_getcpu());

            // You can call classifyAPoint here with the data you want to classify
           
    double testX = 2.5;
    double testY = 4;
    int predictedClass = predictClass(model, testX, testY);
            printf("The predicted class for the value is => %d\n", predictedClass);
            return 0;
        } else if (pid < 0) {
            perror("fork");
        }
    }

    // The parent process waits for all child processes to finish
    int status;
    for (int i = 0; i < num_cores; i++) {
        wait(&status);
    }

    clock_t end_time = clock();  // Record the end time
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;  // Calculate elapsed time in seconds

    cout << "Total time taken: " << elapsed_time << " seconds" << endl;

  
  

    return 0;
}
