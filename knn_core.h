#ifndef KNN_CORE_H
#define KNN_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef KNN_FEATURE_TYPE
#define KNN_FEATURE_TYPE int16_t
#endif

#ifndef KNN_DISTANCE_TYPE
#define KNN_DISTANCE_TYPE int32_t
#endif

#ifndef KNN_LABEL_TYPE
#define KNN_LABEL_TYPE int8_t
#endif

typedef KNN_FEATURE_TYPE KnnFeature;
typedef KNN_DISTANCE_TYPE KnnDistance;
typedef KNN_LABEL_TYPE KnnLabel;

typedef struct {
    const KnnFeature* features;
    KnnLabel label;
    uint8_t dim;
} KnnPoint;

typedef struct {
    KnnPoint point;
    KnnDistance distance;
} KnnNeighbor;

typedef struct {
    KnnPoint train_point;
    KnnPoint test_point;
    KnnDistance current_sum;
    uint8_t current_dim;
} KnnQueueItem;

typedef struct {
    KnnQueueItem* buffer;
    uint16_t capacity;
    uint16_t front;
    uint16_t rear;
    uint16_t count;
} KnnQueue;

typedef struct {
    KnnNeighbor* heap;
    uint16_t size;
    uint16_t capacity;
} KnnHeap;

void knn_heap_init(KnnHeap* heap, KnnNeighbor* buffer, uint16_t capacity);
void knn_heap_insert(KnnHeap* heap, KnnPoint point, KnnDistance distance);
KnnNeighbor* knn_heap_top(KnnHeap* heap);
void knn_heap_pop(KnnHeap* heap);
uint16_t knn_heap_size(KnnHeap* heap);

void knn_queue_init(KnnQueue* queue, KnnQueueItem* buffer, uint16_t capacity);
int knn_queue_enqueue(KnnQueue* queue, KnnPoint train, KnnPoint test, KnnDistance sum, uint8_t dim);
int knn_queue_dequeue(KnnQueue* queue, KnnQueueItem* item);
uint16_t knn_queue_size(KnnQueue* queue);
int knn_queue_is_empty(KnnQueue* queue);

KnnDistance knn_manhattan_distance(const KnnPoint* p1, const KnnPoint* p2);

void knn_predict(
    const KnnPoint* train_data,
    uint16_t train_size,
    const KnnPoint* test_point,
    uint8_t k,
    KnnLabel* predicted_label
);

#ifdef __cplusplus
}
#endif

#endif
