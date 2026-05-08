# KNN Core

Lightweight KNN implementation for embedded systems and microcontrollers. Zero dynamic memory allocation, pure integer arithmetic.

## Overview

KNN Core is a lightweight K-Nearest Neighbors implementation designed for embedded systems and resource-constrained environments.

## Features

- Zero dynamic memory allocation (malloc-free)
- Pure integer arithmetic (no floating point)
- Manhattan distance for faster computation
- Dimensional pruning optimization
- Highly portable (standard C99)

## Algorithm

### Manhattan Distance

Unlike Euclidean distance (which requires square root), Manhattan distance is the sum of absolute differences:

```
d(p1, p2) = |x1-x2| + |y1-y2| + ... + |zn-z2|
```

This avoids expensive floating-point operations, making it ideal for microcontrollers.

### Dimensional Pruning

The algorithm processes dimensions incrementally. If at any point the accumulated distance exceeds the current maximum distance in the heap, further computation is skipped:

```
For each training point:
    1. Calculate first dimension difference
    2. If diff >= current max distance in heap → Skip this point
    3. Otherwise, add to queue and continue to next dimension
    4. Repeat until all dimensions are processed or pruned
```

This significantly reduces computation, especially for high-dimensional data.

### Max-Heap Based kNN Selection

A max-heap of size k maintains the k nearest neighbors:
- Heap root contains the largest distance among k neighbors
- New points with smaller distances can replace the root
- Time complexity: O(n log k) instead of O(n log n)

## Files

- `knn_core.h` - API header file
- `knn_core.c` - Implementation
- `knn_test_minimal.c` - Minimal test file

## Configuration

Define these macros before including the header to customize:

```c
#define KNN_FEATURE_TYPE int16_t  // Feature data type
#define KNN_DISTANCE_TYPE int32_t // Distance accumulator type
#define KNN_LABEL_TYPE int8_t     // Label type
```

## Data Structures

### KnnPoint
```c
typedef struct {
    const KnnFeature* features;  // Pointer to feature array
    KnnLabel label;              // Class label
    uint8_t dim;                 // Number of dimensions
} KnnPoint;
```

### KnnHeap
```c
typedef struct {
    KnnNeighbor* heap;    // Heap buffer (user allocated)
    uint16_t size;        // Current size
    uint16_t capacity;    // Maximum capacity
} KnnHeap;
```

## API Reference

### knn_manhattan_distance
```c
KnnDistance knn_manhattan_distance(const KnnPoint* p1, const KnnPoint* p2);
```
Calculate Manhattan distance between two points.

### knn_predict
```c
void knn_predict(
    const KnnPoint* train_data,  // Training dataset
    uint16_t train_size,         // Number of training samples
    const KnnPoint* test_point,  // Point to classify
    uint8_t k,                   // Number of neighbors
    KnnLabel* predicted_label    // Output: predicted label
);
```
Perform KNN classification using:
1. Dimensional pruning to skip unlikely candidates
2. Max-heap to efficiently maintain k nearest neighbors
3. Majority voting for final prediction

## Usage Example

```c
#include "knn_core.h"

int main() {
    // Prepare training data
    KnnFeature f1[] = {10, 11};
    KnnFeature f2[] = {50, 10};
    KnnPoint train[] = {{f1, 0, 2}, {f2, 1, 2}};

    // Prepare test point
    KnnFeature test_features[] = {50, 9};
    KnnPoint test = {test_features, -1, 2};

    // Run prediction
    KnnLabel result;
    knn_predict(train, 2, &test, 1, &result);

    return result;
}
```

## Memory Requirements

| Component | Memory Size | Notes |
|-----------|-------------|-------|
| KnnPoint | dim * sizeof(KnnFeature) + sizeof(KnnLabel) + 1 | Per point |

## Platform Requirements

- Standard C99 compatible compiler
- No dynamic memory allocation (malloc/free)
- Integer arithmetic only (no floating point)

## Compilation

```bash
# Compile as object file
gcc -c knn_core.c -o knn_core.o

# Compile with test
gcc -DTEST_KNN knn_core.c knn_test_minimal.c -o knn_test
```

## License

MIT License - See LICENSE file for details.
