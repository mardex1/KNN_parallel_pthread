#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <pthread.h>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace std;

// Estrutura para representar um ponto (com features e label)
struct Point {
    vector<double> features;
    int label;  // Para pontos de teste, pode ser -1 (não classificado)
};

// Estrutura para passar parâmetros para cada thread
struct ThreadData {
    int testIndex;                        // Índice do ponto de teste a ser processado
    int k;                                // Número de vizinhos
    int numTrain;                         // Número de pontos de treinamento
    int dim;                              // Número de dimensões
    const vector<Point>* trainingData;   // Ponteiro para o vetor de dados de treinamento
    const vector<Point>* testData;         // Ponteiro para o vetor de dados de teste
    const vector<int>* trainingLabels;     // Ponteiro para o vetor de labels dos pontos de treinamento
    int classificationResult;             // Resultado da classificação para o ponto de teste
};

// Função executada por cada thread para classificar um ponto de teste
void* knnThreadFunc(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    int testIdx = data->testIndex;
    int numTrain = data->numTrain;
    int k = data->k;
    int dim = data->dim;
    
    // Referências para facilitar o acesso aos dados
    const vector<Point>& trainingData = *(data->trainingData);
    const vector<Point>& testData = *(data->testData);
    const vector<int>& trainingLabels = *(data->trainingLabels);
    
    // Vetor para armazenar pares (distância, label)
    vector<pair<double, int>> distLabel;
    distLabel.reserve(numTrain);
    
    // Calcula a distância Euclidiana entre o ponto de teste e cada ponto de treinamento
    for (int i = 0; i < numTrain; i++) {
        double sum = 0.0;
        for (int d = 0; d < dim; d++) { // 1 -> todos do treino
            double diff = trainingData[i].features[d] - testData[testIdx].features[d];
            sum += diff * diff;
        }
        double distance = sqrt(sum);
        distLabel.push_back(make_pair(distance, trainingLabels[i]));
    }
    
    // Ordena os pares pela distância (crescente)
    sort(distLabel.begin(), distLabel.end(),
              [](const pair<double, int>& a, const pair<double, int>& b) {
                  return a.first < b.first;
              });
    // 30 instancias de teste
    // k = 32
    // Realiza a votação entre os k vizinhos mais próximos
    // (aqui, assumimos que os labels são inteiros não-negativos)
    vector<int> voteCount; // Tamanho = número de classes -> voteCount[0] = quantos pontos dos k mais próximos pertencem a classe 0.
    for (int i = 0; i < k; i++) {
        int label = distLabel[i].second;
        if (label >= (int)voteCount.size())
            voteCount.resize(label + 1, 0);
        voteCount[label]++;
    }
    
    // Determina a classe com o maior número de votos
    int bestLabel = -1;
    int maxVotes = 0;
    for (int i = 0; i < voteCount.size(); i++) {
        if (voteCount[i] > maxVotes) {
            maxVotes = voteCount[i];
            bestLabel = (int)i;
        }
    }
    data->classificationResult = bestLabel;
    
    pthread_exit(nullptr);
}

vector<Point> readIrisCSV(const string& filename) {
    vector<Point> dataset;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    // Lê o cabeçalho (se existir) e ignora
    if (getline(file, line)) {
        // Se não houver cabeçalho, comente essa linha
        // cout << "Cabeçalho: " << line << endl;
    }

    // Mapeamento dos labels para inteiros
    unordered_map<string, int> labelMap = {
        {"Iris-setosa", 0},
        {"Iris-versicolor", 1},
        {"Iris-virginica", 2}
    };

    // Processa cada linha do arquivo CSV
    while (getline(file, line)) {
        if (line.empty())
            continue;  // ignora linhas vazias

        istringstream iss(line);
        string token;
        Point p;

        // Lê os 4 atributos numéricos
        for (int i = 0; i < 4; i++) {
            if (!getline(iss, token, ',')) {
                cerr << "Formato inválido na linha: " << line << endl;
                break;
            }
            try {
                p.features.push_back(stod(token));
            } catch (const invalid_argument& e) {
                cerr << "Erro na conversão para double: " << token << endl;
            }
        }

        // Lê o label (último campo)
        if (getline(iss, token, ',')) {
            // Remove eventuais espaços em branco no início/fim
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            if (labelMap.find(token) != labelMap.end()) {
                p.label = labelMap[token];
            } else {
                cerr << "Label desconhecido: " << token << endl;
                p.label = -1;  // ou trate como erro
            }
        }
        dataset.push_back(p);
    }

    file.close();
    return dataset;
}

int main() {
    string train_file = "train.csv";
    string test_file = "train.csv";

    vector<Point> irisTrainData = readIrisCSV(train_file);
    vector<Point> irisTestData = readIrisCSV(test_file);

    // Parâmetros
    const int dim = 4;          // Número de features (dimensões)
    const int numTrain = 120;     // Número de pontos de treinamento
    const int numTest = 30;      // Número de pontos de teste
    const int k = 3;            // Número de vizinhos a considerar 
    
    vector<int> training_labels;

    for(int i = 0; i < irisTrainData.size(); i++) {
        training_labels.push_back(irisTrainData[i].label);
    }

    vector<int> test_labels;

    for(int i = 0; i < irisTestData.size(); i++) {
        test_labels.push_back(irisTestData[i].label);
    }

    for(int i = 0; i < irisTrainData.size(); i++) {
        cout << training_labels[i] << endl;
    }

    // Vetores para armazenar as threads e os dados de cada thread
    vector<pthread_t> threads(numTest);
    vector<ThreadData> threadData(numTest);

    // Cria uma thread para cada ponto de teste
    for (int i = 0; i < numTest; i++) {
        threadData[i].testIndex = i;
        threadData[i].k = k;
        threadData[i].numTrain = numTrain;
        threadData[i].dim = dim;
        threadData[i].trainingData = &irisTrainData;
        threadData[i].testData = &irisTestData;
        threadData[i].trainingLabels = &training_labels;

        int rc = pthread_create(&threads[i], nullptr, knnThreadFunc, (void*)&threadData[i]);
        if (rc) {
            cerr << "Erro ao criar a thread para o ponto de teste " << i << endl;
            return 1;
        }
    }

    // Aguarda a finalização de todas as threads
    for (int i = 0; i < numTest; i++) {
        pthread_join(threads[i], nullptr);
    }

    // Exibe os resultados da classificação para cada ponto de teste
    double accuracy = 0.0;
    for (int i = 0; i < numTest; i++) {
        if(threadData[i].classificationResult == test_labels[i]) {
            accuracy++;
        }
    }
    accuracy /= numTest;
    cout << "Acurácia do modelo: " << accuracy*100 << "%" << endl;
    return 0;
}
