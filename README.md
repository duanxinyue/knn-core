# KNN Core

Lightweight KNN implementation for embedded systems and microcontrollers. Zero dynamic memory allocation, pure integer arithmetic.

## Overview

KNN Core is a lightweight K-Nearest Neighbors implementation designed for embedded systems and resource-constrained environments.

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
Perform KNN classification.

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
