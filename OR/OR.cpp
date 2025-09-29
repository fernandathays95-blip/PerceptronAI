#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Definição do PerceptronModel e funções (do Bloco 1, 2 e 3)
// ... (Seu código PerceptronModel, activation_function e train_single_example aqui) ...

// Conjunto de dados para a função OR
struct ORDataset {
    vector<vector<double>> inputs = {
        {0.0, 0.0}, // (0, 0)
        {0.0, 1.0}, // (0, 1)
        {1.0, 0.0}, // (1, 0)
        {1.0, 1.0}  // (1, 1)
    };
    vector<int> expected_outputs = {0, 1, 1, 1};
    int size = 4;
};


// Inicializa o modelo com pesos e viés aleatórios
PerceptronModel initialize_perceptron(double learning_rate, int num_inputs) {
    srand(time(0)); // Inicializa o gerador de numeros aleatorios
    PerceptronModel p;
    p.learning_rate = learning_rate;
    p.bias = (double)(rand() % 200 - 100) / 100.0; // Viés inicial entre -1 e 1
    
    for (int i = 0; i < num_inputs; ++i) {
        // Pesos iniciais aleatórios entre -1 e 1
        p.weights.push_back((double)(rand() % 200 - 100) / 100.0);
    }
    return p;
}
