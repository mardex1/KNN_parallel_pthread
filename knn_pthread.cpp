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
    int idxTest;                        // Índice do ponto de teste a ser processado
    int k;                                // Número de vizinhos
    int trainSize;                         // Número de pontos de treinamento
    int nFeatures;                              // Número de dimensões
    const vector<Point>* trainingData;   // Ponteiro para o vetor de dados de treinamento
    const vector<Point>* testData;         // Ponteiro para o vetor de dados de teste
    const vector<int>* trainingLabels;     // Ponteiro para o vetor de labels dos pontos de treinamento
    int classificacaoFinal;             // Resultado da classificação para o ponto de teste
};

// Função executada por cada thread para classificar um ponto de teste
void* knnThreadFunc(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int testIdx = data->idxTest;
    int trainSize = data->trainSize;
    int k = data->k;
    int nFeatures = data->nFeatures;
    
    // Referências para facilitar o acesso aos dados
    const vector<Point>& trainingData = *(data->trainingData);
    const vector<Point>& testData = *(data->testData);
    const vector<int>& trainingLabels = *(data->trainingLabels);
    
    // Vetor para armazenar pares (distância, label)
    vector<pair<double, int>> distLabel;
    distLabel.reserve(trainSize);
    
    // Calcula a distância Euclidiana entre o ponto de teste e cada ponto de treinamento
    for (int i = 0; i < trainSize; i++) {
        double sum = 0.0;
        for (int d = 0; d < nFeatures; d++) { // 1 -> todos do treino
            double diff = trainingData[i].features[d] - testData[testIdx].features[d];
            sum += diff * diff;
        }
        double distance = sqrt(sum);
        distLabel.push_back(make_pair(distance, trainingLabels[i]));
    }
    
    // Ordena os pares pela distância (crescente)
    sort(distLabel.begin(), distLabel.end());
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
    data->classificacaoFinal = bestLabel;
    
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
    // Lê o cabeçalho e não faz nada 
    getline(file, line);
    

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
            p.features.push_back(stod(token));
        }

        // Lê o label (último campo)
        if (getline(iss, token, ',')) { 
            p.label = labelMap[token];
        }
        dataset.push_back(p);
    }
    file.close();
    return dataset;
}

int main() {
    string train_file = "Datasets/train.csv";
    string test_file = "Datasets/test.csv";

    vector<Point> irisTrainData = readIrisCSV(train_file);
    vector<Point> irisTestData = readIrisCSV(test_file);

    // Parâmetros
    const int nFeatures = 4;          // Número de features (nFeaturesensões)
    const int trainSize = 120;     // Número de pontos de treinamento
    const int numTest = 30;      // Número de pontos de teste
    int k;            // Número de vizinhos a considerar 
    cout << "Quantos vizinhos serão considerados no knn?" << endl;
    cin >> k;
    
    vector<int> training_labels;

    for(int i = 0; i < irisTrainData.size(); i++) {
        training_labels.push_back(irisTrainData[i].label);
    }

    vector<int> test_labels;

    for(int i = 0; i < irisTestData.size(); i++) {
        test_labels.push_back(irisTestData[i].label);
    }

    // Vetores para armazenar as threads e os dados de cada thread
    vector<pthread_t> threads(numTest);
    vector<ThreadData> threadData(numTest);

    // Cria uma thread para cada ponto de teste
    for (int i = 0; i < numTest; i++) {
        threadData[i].idxTest = i;
        threadData[i].k = k;
        threadData[i].trainSize = trainSize;
        threadData[i].nFeatures = nFeatures;
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
        if(threadData[i].classificacaoFinal == test_labels[i]) {
            accuracy++;
        }
    }
    accuracy /= numTest;
    cout << "Acurácia do modelo: " << accuracy*100 << "%" << endl;
    return 0;
}
