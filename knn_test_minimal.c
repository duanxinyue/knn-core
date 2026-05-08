#include "knn_core.h"

#if defined(TEST_KNN) || defined(DEBUG)

#include <stdio.h>

static int run_basic_test() {
    const uint8_t dim = 2;
    KnnFeature f1[] = {10, 11};
    KnnFeature f2[] = {50, 10};
    KnnPoint train[] = {{f1, 0, dim}, {f2, 1, dim}};
    
    KnnFeature test[] = {50, 9};
    KnnPoint point = {test, -1, dim};
    
    KnnLabel result;
    knn_predict(train, 2, &point, 1, &result);
    
    return (result == 1) ? 0 : 1;
}

int main() {
    int ret = run_basic_test();
    if (ret == 0) {
        printf("KNN core test passed.\n");
    } else {
        printf("KNN core test failed.\n");
    }
    return ret;
}

#endif
