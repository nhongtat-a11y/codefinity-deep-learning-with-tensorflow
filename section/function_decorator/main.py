import tensorflow as tf
import time

# Function with `tf.function` decorator
@tf.function
def matrix_multiply_optimized(mat1, mat2):
    x = tf.matmul(mat1, mat2)
    for i in range(100):
        x = tf.matmul(x, mat2)
    return tf.reduce_mean(x)

# Function without `tf.function` decorator
def matrix_multiply_basic(mat1, mat2):
    x = tf.matmul(mat1, mat2)
    for i in range(100):
        x = tf.matmul(x, mat2)
    return tf.reduce_mean(x)

# Create random matrices
mat1 = tf.random.uniform((100, 100), minval=0., maxval=0.01)
mat2 = tf.random.uniform((100, 100), minval=0., maxval=0.01)

# Run once for the graph to set up
result_optimized = matrix_multiply_optimized(mat1, mat2)

# Measure execution time for basic function
start_time = time.time()
result_basic = matrix_multiply_basic(mat1, mat2)
basic_time = time.time() - start_time

# Measure execution time for optimized function
start_time = time.time()
result_optimized = matrix_multiply_optimized(mat1, mat2)
optimized_time = time.time() - start_time

# Print results
print(f"Time taken by basic function: {basic_time:.4f} seconds")
print(f"Time taken by optimized function: {optimized_time:.4f} seconds")