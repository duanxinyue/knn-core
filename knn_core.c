#include "knn_core.h"
#include <string.h>

static void swap(KnnNeighbor* a, KnnNeighbor* b) {
    KnnNeighbor temp = *a;
    *a = *b;
    *b = temp;
}

static void heapify(KnnNeighbor* heap, uint16_t size, uint16_t parent) {
    uint16_t largest = parent;
    uint16_t left = 2 * parent + 1;
    uint16_t right = 2 * parent + 2;

    if (left < size && heap[left].distance > heap[largest].distance) {
        largest = left;
    }
    if (right < size && heap[right].distance > heap[largest].distance) {
        largest = right;
    }

    if (largest != parent) {
        swap(&heap[parent], &heap[largest]);
        heapify(heap, size, largest);
    }
}

static void build_max_heap(KnnNeighbor* heap, uint16_t size) {
    for (int i = size / 2 - 1; i >= 0; i--) {
        heapify(heap, size, i);
    }
}

void knn_heap_init(KnnHeap* heap, KnnNeighbor* buffer, uint16_t capacity) {
    heap->heap = buffer;
    heap->size = 0;
    heap->capacity = capacity;
}

void knn_heap_insert(KnnHeap* heap, KnnPoint point, KnnDistance distance) {
    if (heap->size < heap->capacity) {
        heap->heap[heap->size].point = point;
        heap->heap[heap->size].distance = distance;
        heap->size++;
        
        int i = heap->size - 1;
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap->heap[i].distance <= heap->heap[parent].distance) {
                break;
            }
            swap(&heap->heap[i], &heap->heap[parent]);
            i = parent;
        }
    } else if (distance < heap->heap[0].distance) {
        heap->heap[0].point = point;
        heap->heap[0].distance = distance;
        heapify(heap->heap, heap->size, 0);
    }
}

KnnNeighbor* knn_heap_top(KnnHeap* heap) {
    return heap->size > 0 ? &heap->heap[0] : NULL;
}

void knn_heap_pop(KnnHeap* heap) {
    if (heap->size == 0) return;
    
    heap->heap[0] = heap->heap[heap->size - 1];
    heap->size--;
    heapify(heap->heap, heap->size, 0);
}

uint16_t knn_heap_size(KnnHeap* heap) {
    return heap->size;
}

void knn_queue_init(KnnQueue* queue, KnnQueueItem* buffer, uint16_t capacity) {
    queue->buffer = buffer;
    queue->capacity = capacity;
    queue->front = 0;
    queue->rear = 0;
    queue->count = 0;
}

int knn_queue_enqueue(KnnQueue* queue, KnnPoint train, KnnPoint test, KnnDistance sum, uint8_t dim) {
    if (queue->count >= queue->capacity) {
        return 0;
    }
    
    queue->buffer[queue->rear].train_point = train;
    queue->buffer[queue->rear].test_point = test;
    queue->buffer[queue->rear].current_sum = sum;
    queue->buffer[queue->rear].current_dim = dim;
    
    queue->rear = (queue->rear + 1) % queue->capacity;
    queue->count++;
    return 1;
}

int knn_queue_dequeue(KnnQueue* queue, KnnQueueItem* item) {
    if (queue->count == 0) {
        return 0;
    }
    
    *item = queue->buffer[queue->front];
    queue->front = (queue->front + 1) % queue->capacity;
    queue->count--;
    return 1;
}

uint16_t knn_queue_size(KnnQueue* queue) {
    return queue->count;
}

int knn_queue_is_empty(KnnQueue* queue) {
    return queue->count == 0;
}

KnnDistance knn_manhattan_distance(const KnnPoint* p1, const KnnPoint* p2) {
    KnnDistance distance = 0;
    for (uint8_t i = 0; i < p1->dim; i++) {
        KnnDistance diff = p1->features[i] - p2->features[i];
        distance += (diff >= 0) ? diff : -diff;
    }
    return distance;
}

void knn_predict(
    const KnnPoint* train_data,
    uint16_t train_size,
    const KnnPoint* test_point,
    uint8_t k,
    KnnLabel* predicted_label
) {
    if (k == 0 || train_size == 0) {
        *predicted_label = -1;
        return;
    }

    uint8_t actual_k = (k > train_size) ? train_size : k;

    KnnNeighbor heap_buffer[actual_k];
    KnnHeap heap;
    knn_heap_init(&heap, heap_buffer, actual_k);

    KnnQueueItem queue_buffer[train_size];
    KnnQueue queue;
    knn_queue_init(&queue, queue_buffer, train_size);

    for (uint16_t i = 0; i < train_size; i++) {
        KnnDistance first_dim_diff = (train_data[i].features[0] >= test_point->features[0])
            ? (train_data[i].features[0] - test_point->features[0])
            : (test_point->features[0] - train_data[i].features[0]);

        if (knn_heap_size(&heap) < actual_k) {
            knn_heap_insert(&heap, train_data[i], 0);
        } else {
            KnnNeighbor* top = knn_heap_top(&heap);
            if (top && first_dim_diff < top->distance) {
                knn_queue_enqueue(&queue, train_data[i], *test_point, first_dim_diff, 1);
            }
        }
    }

    KnnNeighbor initial_heap[actual_k];
    uint16_t initial_size = 0;
    
    for (uint16_t i = 0; i < train_size && initial_size < actual_k; i++) {
        KnnDistance dist = knn_manhattan_distance(&train_data[i], test_point);
        initial_heap[initial_size].point = train_data[i];
        initial_heap[initial_size].distance = dist;
        initial_size++;
    }
    
    build_max_heap(initial_heap, initial_size);
    memcpy(heap_buffer, initial_heap, sizeof(KnnNeighbor) * initial_size);
    heap.size = initial_size;

    while (!knn_queue_is_empty(&queue)) {
        KnnQueueItem item;
        if (!knn_queue_dequeue(&queue, &item)) {
            break;
        }

        if (item.current_dim < item.train_point.dim) {
            KnnDistance dim_diff = (item.train_point.features[item.current_dim] >= item.test_point.features[item.current_dim])
                ? (item.train_point.features[item.current_dim] - item.test_point.features[item.current_dim])
                : (item.test_point.features[item.current_dim] - item.train_point.features[item.current_dim]);
            
            KnnDistance new_sum = item.current_sum + dim_diff;

            KnnNeighbor* top = knn_heap_top(&heap);
            if (top && new_sum < top->distance) {
                if (item.current_dim == item.train_point.dim - 1) {
                    knn_heap_insert(&heap, item.train_point, new_sum);
                } else {
                    knn_queue_enqueue(&queue, item.train_point, item.test_point, new_sum, item.current_dim + 1);
                }
            }
        }
    }

    KnnLabel label_count[256] = {0};
    for (uint16_t i = 0; i < heap.size; i++) {
        if (heap.heap[i].point.label >= 0 && heap.heap[i].point.label < 256) {
            label_count[heap.heap[i].point.label]++;
        }
    }

    KnnLabel max_label = -1;
    KnnLabel max_count = 0;
    for (int i = 0; i < 256; i++) {
        if (label_count[i] > max_count) {
            max_count = label_count[i];
            max_label = i;
        }
    }

    *predicted_label = max_label;
}
