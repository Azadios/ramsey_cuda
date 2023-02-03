#include <iostream>
#include <numeric>
#include <vector>
using std::vector;
#include <set>
using std::set;
#include <fstream>
#include <functional>
#include "definitions.cuh"
#include "runConfig.cuh"
#include "cudaConfig.cuh"

static set<int> getDifferences(int subgraphVertices[SUBGRAPH_VERTEX_COUNT]) {
    set<int> differences;
    for (int i = 0; i < SUBGRAPH_VERTEX_COUNT - 1; ++i) {
        for (int j = i + 1; j < SUBGRAPH_VERTEX_COUNT; ++j) {
            differences.insert(subgraphVertices[j] - subgraphVertices[i]);
        }
    }
    return differences;
}

static unsigned long long getCondition(int subgraphVertices[SUBGRAPH_VERTEX_COUNT]) {
    set<int> differences = getDifferences(subgraphVertices);
    unsigned long long condition = 0;
    for (int difference : differences) {
        condition |= (1ull << (VERTEX_COUNT - difference - 1));
    }
    return condition;
}

static void markSubconditions(vector<unsigned long long> &conditions, const int currentConditionIndex) {
    #pragma omp parallel for default(none) shared(conditions) firstprivate(currentConditionIndex)
    for (int subconditionCandidateIndex = currentConditionIndex + 1; subconditionCandidateIndex < conditions.size(); ++subconditionCandidateIndex) {
        const unsigned long long &currentCondition = conditions[currentConditionIndex];
        unsigned long long &subconditionCandidate = conditions[subconditionCandidateIndex];
        if ((currentCondition & subconditionCandidate) == currentCondition) {
            subconditionCandidate |= SUBCONDITION_FLAG;
        }
    }
}

static void removeExcessiveConditions(vector<unsigned long long> &conditions) {
    std::cout << "Marking subconditions..." << std::endl;
    for (int currentConditionIndex = 0; currentConditionIndex < conditions.size() - 1; ++currentConditionIndex) {
        if (conditions[currentConditionIndex] & SUBCONDITION_FLAG) {
            continue;
        }
        markSubconditions(conditions, currentConditionIndex);
    }
    std::cout << "Erasing excessive conditions..." << std::endl;

    auto erasedStart = std::partition(conditions.begin(), conditions.end(), [](unsigned long long condition) {
                                                    return !(condition & SUBCONDITION_FLAG);
                                                });
    conditions.erase(erasedStart, conditions.end());
    std::cout << "Excessive conditions erased." << std::endl;
}

void initConditions(vector<unsigned long long> &conditions) {
    int subgraphVertices[SUBGRAPH_VERTEX_COUNT];
    std::iota(std::begin(subgraphVertices), std::end(subgraphVertices), 0);
    constexpr int lastVertexIndex = SUBGRAPH_VERTEX_COUNT - 1;
    constexpr int firstSubgraphVertexMax = VERTEX_COUNT - SUBGRAPH_VERTEX_COUNT;

    std::cout << "Generating conditions..." << std::endl;
    int currentVertexIndex = lastVertexIndex;
    while (true) {
        unsigned long long condition = getCondition(subgraphVertices);
        conditions.push_back(condition);

        if (subgraphVertices[0] == firstSubgraphVertexMax) {
            break;
        }

        ++subgraphVertices[currentVertexIndex];
        while (subgraphVertices[currentVertexIndex] == firstSubgraphVertexMax + currentVertexIndex + 1) {
            --currentVertexIndex;
            ++subgraphVertices[currentVertexIndex];
        }
        while (currentVertexIndex < lastVertexIndex) {
            ++currentVertexIndex;
            subgraphVertices[currentVertexIndex] = subgraphVertices[currentVertexIndex - 1] + 1;
        }
    }
    std::cout << "Conditions generated." << std::endl;

    removeExcessiveConditions(conditions);
}

void checkForSolution(unsigned long long *solutions_h, const unsigned long long *solutions_d, cudaStream_t stream) {
    if (stream != nullptr) {
        cudaMemcpyAsync(solutions_h, solutions_d, TOTAL_THREADS_COUNT * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        cudaMemcpy(solutions_h, solutions_d, TOTAL_THREADS_COUNT * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }

    bool found = false;
    for (int i = 0; i < TOTAL_THREADS_COUNT; ++i) {
        if (solutions_h[i] != 0) {
            std::cout << "Solution by thread[" << i << "]: " << solutions_h[i] << std::endl;
            for (int j = VERTEX_COUNT - 2; j >= 0; --j) {
                std::cout << ((solutions_h[i] >> j) & 1);
            }
            std::cout << std::endl;
            found = true;
        }
    }
    if (!found) {
        std::cout << "No solution found" << std::endl;
    }
}

[[maybe_unused]]
void readConditions(vector<unsigned long long> &conditions, const char *filename) {
    std::ifstream file(filename);
    unsigned long long condition;
    while (file >> condition) {
        conditions.push_back(condition);
    }
    file.close();
}

[[maybe_unused]]
void printConditions(vector<unsigned long long> &conditions, std::ostream &out, bool binary) {
    for (auto condition: conditions) {
        if (binary) {
            for (int i = VERTEX_COUNT - 2; i >= 0; --i) {
                out << ((condition >> i) & 1);
            }
        } else {
            out << condition;
        }

        out << std::endl;
    }
}
