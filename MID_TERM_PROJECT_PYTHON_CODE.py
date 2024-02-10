import numpy as np
import time
import random
import matplotlib.pyplot as plt

def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    less = []
    equal = []
    greater = []

    for x in arr:
        if x < pivot:
            less.append(x)
        elif x == pivot:
            equal.append(x)
        elif x > pivot:
            greater.append(x)

    return quick_sort(less) + equal + quick_sort(greater)

def heap_sort(arr):
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[l] > arr[largest]:
            largest = l

        if r < n and arr[r] > arr[largest]:
            largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def radix_sort(arr):
    def counting_sort(arr, exp):
        output = [0] * len(arr)
        count = [0] * 10

        for i in range(len(arr)):
            index = arr[i] // exp
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = len(arr) - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1

        for i in range(len(arr)):
            arr[i] = output[i]

    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

def bucket_sort(arr):
    max_num = max(arr)
    min_num = min(arr)
    num_buckets = (max_num - min_num) // 100 + 1
    buckets = [[] for _ in range(num_buckets)]

    for num in arr:
        index = (num - min_num) // 100
        buckets[index].append(num)

    for bucket in buckets:
        heap_sort(bucket)

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr

def insertion_sort(arr, start, end):
    for i in range(start + 1, end + 1):
        key = arr[i]
        j = i - 1
        while j >= start and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def timsort(arr):
    min_run = 32
    n = len(arr)

    for i in range(0, n, min_run):
        insertion_sort(arr, i, min(i + min_run - 1, n - 1))

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min(left + 2 * size - 1, n - 1)
            arr[left:right + 1] = merge(arr[left:mid + 1], arr[mid + 1:right + 1])
        size *= 2

    return arr

def generate_inputs(size, input_type):
    if input_type == 1:
        return list(np.random.randint(0, size, size=size))
    elif input_type == 2:
        k = min(1000, size)
        return list(np.random.randint(0, k, size=size))
    elif input_type == 3:
        max_cube = np.iinfo(np.int32).max
        max_size = int((max_cube ** (1/3)) // 2)
        size = min(size, max_size)
        return list(np.random.randint(0, size**3, size=size))
    elif input_type == 4:
        return list(np.random.randint(0, int(np.log2(size)), size=size))
    elif input_type == 5:
        return list(np.random.randint(0, size, size=size) * 1000)
    elif input_type == 6:
        arr = list(range(size))
        for _ in range(int(np.log2(size)) // 2):
            i, j = random.sample(range(size), 2)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

def measure_time(algorithm, arr):
    start_time = time.time()
    algorithm(arr)
    end_time = time.time()
    return end_time - start_time

def compare_algorithms(input_sizes, input_type):
    algorithms = {
        'Quick Sort': quick_sort,
        'Heap Sort': heap_sort,
        'Merge Sort': merge_sort,
        'Radix Sort': radix_sort,
        'Bucket Sort': bucket_sort,
        'TimSort': timsort
    }
    results = {alg: [] for alg in algorithms}

    for size in input_sizes:
        arr = generate_inputs(size, input_type)
        for alg_name, alg_func in algorithms.items():
            # Measure time
            start_time = time.time()
            sorted_arr = alg_func(arr.copy())
            end_time = time.time()
            time_taken = end_time - start_time

            # Print execution time
            print(f"{alg_name} - Input Size: {size}, Input Type: {input_type}, Execution Time: {time_taken:.6f} seconds")
           
            results[alg_name].append(time_taken)

    return results

def plot_results(results, input_sizes, input_type):
    plt.figure(figsize=(12, 8))
    for alg_name, times in results.items():
        plt.plot(input_sizes, times, label=alg_name)
    plt.title(f'Sorting Algorithms Performance (Input Type {input_type})')
    plt.xlabel('Input Size')
    plt.ylabel('Time (seconds)')
   
    # Select a subset of input sizes for x-axis ticks
    num_ticks = 5  # Number of ticks to display
    tick_indices = np.linspace(0, len(input_sizes) - 1, num_ticks, dtype=int)
    plt.xticks([input_sizes[i] for i in tick_indices], [input_sizes[i] for i in tick_indices], rotation=45)  # Rotate the x-axis labels by 45 degrees
   
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()



if __name__ == '__main__':
    input_sizes = [100, 500, 1000, 5000, 10000]
    input_types = [1, 2, 3, 4, 5, 6]

    for input_type in input_types:
        results = compare_algorithms(input_sizes, input_type)
        plot_results(results, input_sizes, input_type)
