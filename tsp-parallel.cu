#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>

#define ONE_BILLION (double)1000000000.0
#define TSP_ELT(tsp, n, i, j) *(tsp + (i * n) + j)

/* Print a TSP distance matrix. */
void
print_tsp(int *tsp, int n, int random_seed)
{
printf("TSP (%d cities - seed %d)\n    ", n, random_seed);
for (int j = 0;  j < n;  j++) {
printf("%3d|", j);
}
printf("\n");
for (int i = 0;  i < n;  i++) {
printf("%2d|", i);
for (int j = 0;  j < n;  j++) {
printf("%4d", TSP_ELT(tsp, n, i, j));
}
printf("\n");
}
printf("\n");
}

/* Print a permutation array */
__host__ __device__ void
print_perm(int *perm, int size)
{
  printf("Min Path: ");
  for (int k = 0; k < size; k++) {
	printf("%4d", perm[k]);
  }
  printf("\n");
}

/**** List ADT ****************/

typedef struct {
    int *values;					/* Values stored in list */
    int max_size;					/* Maximum size allocated */
    int cur_size;					/* Size currently in use */
} list_t;

/* Dump list, including sizes */
__host__ __device__ void
list_dump(list_t *list)
{
    printf("%2d/%2d", list->cur_size, list->max_size);
    for (int i = 0;  i < list->cur_size;  i++) {
        printf(" %d", list->values[i]);
    }
    printf("\n");
}

/* Allocate list that can store up to 'max_size' elements */
__host__ __device__ list_t *
list_alloc(int max_size)
{
    list_t *list = (list_t *)malloc(sizeof(list_t));
    list->values = (int *)malloc(max_size * sizeof(int));
    list->max_size = max_size;
    list->cur_size = 0;
    return list;
}

/* Free a list; call this to avoid leaking memory! */
__host__ __device__ void
list_free(list_t *list)
{
    free(list->values);
    free(list);
}

/* Add a value to the end of the list */
__host__ __device__ void
list_add(list_t *list, int value)
{
    if (list->cur_size >= list->max_size) {
        printf("List full");
        list_dump(list);
    } else {
        list->values[list->cur_size++] = value;
    }
}

/* Return the current size of the list */
__host__ __device__ int
list_size(list_t *list)
{
    return list->cur_size;
}

/* Validate index */
__host__ __device__ void
_list_check_index(list_t *list, int index)
{
    if (index < 0 || index > list->cur_size - 1) {
        printf("Invalid index %d\n", index);
        list_dump(list);
    }
}

/* Get the value at given index */
__host__ __device__ int
list_get(list_t *list, int index)
{
    _list_check_index(list, index);
    return list->values[index];
}

/* Remove the value at the given index */
__host__ __device__ void
list_remove_at(list_t *list, int index)
{
    _list_check_index(list, index);
    for (int i = index; i < list->cur_size - 1;  i++) {
        list->values[i] = list->values[i + 1];
    }
    list->cur_size--;
}

/* Retrieve a copy of the values as a simple array of integers. The returned
array is allocated dynamically; the caller must free the space when no
longer needed.
*/
__host__ __device__ int *
list_as_array(list_t *list)
{
    int *rtn = (int *)malloc(list->max_size * sizeof(int));
    for (int i = 0;  i < list->max_size;  i++) {
        rtn[i] = list_get(list, i);
    }
    return rtn;
}

/* Calculate n! iteratively */
__host__ __device__ unsigned long
factorial(int n)
{
    if (n < 1) {
        return 0;
    }

    unsigned long rtn = 1;
    for (int i = 1;  i <= n;  i++) {
        rtn *= i;
    }
    return rtn;
}

/* Return the kth lexographically ordered permuation of an array of k integers
   in the range [0 .. size - 1]. The integers are allocated dynamically and
   should be free'd by the caller when no longer needed.
*/
__host__ __device__ int *
kth_perm(unsigned long k, int size)
{
    unsigned long remain = k;

    list_t *numbers = list_alloc(size);
    for (int i = 0;  i < size;  i++) {
        list_add(numbers, i);
    }

    list_t *perm = list_alloc(size);

    for (int i = 1;  i < size;  i++) {
        unsigned long f = factorial(size - i);
        unsigned long j = remain / f;
        remain = remain % f;

        list_add(perm, list_get(numbers, j));
        list_remove_at(numbers, j);

        if (remain == 0) {
            break;
        }
    }

    /* Append remaining digits */
    for (int i = 0;  i < list_size(numbers);  i++) {
        list_add(perm, list_get(numbers, i));
    }

    int *rtn = list_as_array(perm);
    list_free(perm);

    return rtn;
}

/* Swap v[i] and v[j] */
__device__ void
swap(int *v, int i, int j)
{
    int t = v[i];
    v[i] = v[j];
    v[j] = t;
}

/* Given an array of size elements at perm, update the array in place to
   contain the lexographically next permutation. It is originally due to
   Dijkstra. The present version is discussed at:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
 */
__device__ void
next_perm(int *perm, int size)
{
    int i = size - 1;
    while (perm[i - 1] >= perm[i]) {
        i = i - 1;
    }

    int j = size;
    while (perm[j - 1] <= perm[i - 1]) {
        j = j - 1;
    }

    swap(perm, i - 1, j - 1);

    i++;
    j = size;
    while (i < j) {
        swap(perm, i - 1, j - 1);
        i++;
        j--;
    }
}

__device__ int 
calc_cost(int* tsp, int* perm, int num_cities) {
    int total = 0;
    for (int i = 0;  i < num_cities;  i++) {
        int j = (i + 1) % num_cities;
        int from = perm[i];
        int to = perm[j];
        int val = TSP_ELT(tsp, num_cities, from, to);
        total += val;
    }
    return total;
}

/* TSP Kernal */
__global__ void
TSP(int* tsp, int* mins, unsigned long* min_perms, unsigned long total_permutations, unsigned long num_threads, unsigned long num_cities) {
    int idx = threadIdx.x;
    unsigned long permutations_per_thread = total_permutations / num_threads;
    unsigned long perm_idx = idx * permutations_per_thread;
    unsigned long stop_idx = (idx + 1) * permutations_per_thread;
    if(idx == num_threads - 1) {
        stop_idx = total_permutations;
    }

    // printf("Thread %d | permutations_per_thread: %ld | perm_idx: %ld | stop_idx: %ld\n", idx, permutations_per_thread, perm_idx, stop_idx);

    int * perm = kth_perm(perm_idx, num_cities);
    int min = INT_MAX;
    unsigned long min_perm = perm_idx;
    
    while(perm_idx < stop_idx) {
        // printf("Thread %d | min: %d\n", idx, min);
        printf("Thread %d | permutations_per_thread: %ld | perm_idx: %ld | stop_idx: %ld\n", idx, permutations_per_thread, perm_idx, stop_idx);
        int cost = calc_cost(tsp, perm, num_cities);
        if(cost < min) {
            min = cost;
            min_perm = perm_idx;
        }
        next_perm(perm, num_cities);
        perm_idx = perm_idx + 1;
    }

    mins[idx] = min;
    min_perms[idx] = min_perm;
}

/* Create an instance of a symmetric TSP. */
int *
create_tsp(int size, int random_seed, int tsp_size)
{
    int *tsp = (int *)malloc(tsp_size);

    srandom(random_seed);
    for (int i = 0;  i < size;  i++) {
        for (int j = 0;  j <= i;  j++) {
            int val = (int)(random() / (RAND_MAX / 100));
            TSP_ELT(tsp, size, i, j) = val;
            TSP_ELT(tsp, size, j, i) = val;
        }
    }
    return tsp;
}

/* Return the current time. */
double now(void)
{
  struct timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

/* Print out help */
void
usage(char *prog_name)
{
  fprintf(stderr, "usage: %s [flags]\n", prog_name);
  fprintf(stderr, "   -h\n");
  fprintf(stderr, "   -t <number of threads>\n");
  fprintf(stderr, "   -c <number of cities>\n");
  fprintf(stderr, "   -s <random seed>\n");
  exit(1);
}

int 
main(int argc, char **argv) {

    int random_seed = time(NULL);
    int num_threads = 0;
    int num_cities = 0;

    int ch;
    while ((ch = getopt(argc, argv, "c:hs:t:")) != -1) {
        switch (ch) {
            case 'c':
                num_cities = atoi(optarg);
            break;
            case 't':
                num_threads = atoi(optarg);
            break;
            case 's':
                random_seed = atoi(optarg);
            break;
            case 'h':
            default:
                usage(argv[0]);
            }
    }

    if(num_cities <= 0 || num_threads <= 0) {
        fprintf(stderr, "Error, number of cities and threads must be above 0\n");
        usage(argv[0]);
    }

    double start_time = now();
    
    // Copy tsp cost map to GPU global memory
    int tsp_size = num_cities * num_cities * sizeof(int);
    int mins_size = num_threads * sizeof(int);
    int min_perms_size = num_threads * sizeof(unsigned long);

    int* h_tsp = create_tsp(num_cities, random_seed, tsp_size);
    int* h_mins = (int*)malloc(mins_size);
    unsigned long* h_min_perms = (unsigned long*)malloc(min_perms_size);

    int* d_tsp;
    int* d_mins;
    unsigned long* d_min_perms;

    cudaMalloc((void **)&d_tsp, tsp_size);
    cudaMalloc((void **)&d_mins, mins_size);
    cudaMalloc((void **)&d_min_perms, min_perms_size);

    cudaMemcpy(d_tsp, h_tsp, tsp_size, cudaMemcpyHostToDevice);

    unsigned long total_permutations = factorial(num_cities);
    fprintf(stderr, "fact: %ld\n", total_permutations);
    // Call the kernel.
    TSP<<<1, num_threads>>>(d_tsp, d_mins, d_min_perms, total_permutations, (unsigned long)num_threads, (unsigned long)num_cities);
    // cudaDeviceSynchronize();

    cudaMemcpy(h_mins, d_mins, mins_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_perms, d_min_perms, min_perms_size, cudaMemcpyDeviceToHost);

    int min = h_mins[0];
    printf("Min Path Cost for %d: %d\n",0, h_mins[0]);
    for(int i = 1; i < num_threads; i++) {
        printf("Min Path Cost for %d: %d\n",i, h_mins[i]);
        if(h_mins[i] < min) {
            min = h_mins[i];
        }
    }

    printf("Min Path Cost: %d\n", min);

    for(int i = 1; i < num_threads; i++) {
        if(h_mins[i] == min) {
            print_perm(kth_perm(h_min_perms[i], num_cities), num_cities);
        }
    }

    /* Report time. */
    printf("    TOOK %5.3f seconds\n", now() - start_time);
}