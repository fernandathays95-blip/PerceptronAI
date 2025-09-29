void train_perceptron(PerceptronModel& p, const ORDataset& data) {
    
    int epoch = 0;
    bool all_correct = false;
    const int MAX_EPOCHS = 100; // Limite para evitar loop infinito
    
    cout << "Iniciando Treinamento da Funcao OR..." << endl;
    
    while (!all_correct && epoch < MAX_EPOCHS) {
        
        all_correct = true; // Assume que esta correto no inicio da epoca
        int errors_in_epoch = 0;

        for (int i = 0; i < data.size; ++i) {
            
            // 1. PREDICAO (Calcula a soma ponderada)
            double weighted_sum = p.bias;
            for (size_t j = 0; j < p.weights.size(); ++j) {
                weighted_sum += data.inputs[i][j] * p.weights[j];
            }
            int prediction = activation_function(weighted_sum);
            
            // 2. CALCULO DO ERRO
            int expected = data.expected_outputs[i];
            int error = expected - prediction;

            // 3. AJUSTE DE PESOS
            if (error != 0) {
                train_single_example(p, data.inputs[i], expected);
                all_correct = false; // Se houve erro, a IA precisa de mais uma epoca
                errors_in_epoch++;
            }
        }

        cout << "  Epoca " << epoch + 1 << " completa. Erros: " << errors_in_epoch << endl;
        epoch++;
    }

    // Resultados finais
    cout << "\nTreinamento Concluido em " << epoch << " epocas." << endl;
    cout << "Pesos Finais: W1=" << p.weights[0] << ", W2=" << p.weights[1] << ", Bias=" << p.bias << endl;
}


// Funcao principal para iniciar o treinamento
int main() {
    
    // 1. Inicializa o modelo
    PerceptronModel or_perceptron = initialize_perceptron(0.1, 2); // 0.1 = Taxa de Aprendizagem, 2 = 2 entradas
    
    // 2. Define o dataset
    ORDataset or_data;
    
    // 3. Inicia o treinamento
    train_perceptron(or_perceptron, or_data);

    // 4. Teste final (Verifica se a IA aprendeu)
    cout << "\n--- TESTE DE PREDICAO ---" << endl;
    for (int i = 0; i < or_data.size; ++i) {
        double sum = or_perceptron.bias + (or_data.inputs[i][0] * or_perceptron.weights[0]) + (or_data.inputs[i][1] * or_perceptron.weights[1]);
        int result = activation_function(sum);
        cout << "(" << or_data.inputs[i][0] << ", " << or_data.inputs[i][1] << ") -> Predito: " << result << " (Esperado: " << or_data.expected_outputs[i] << ")" << endl;
    }
    
    return 0;
}
