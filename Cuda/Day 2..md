---
cssclasses:
  - width-100
---
#### **목표: CUDA 프로그래밍의 기초 이해 및 간단한 CUDA 커널 작성**

- **CUDA 프로그래밍**
    - CUDA 프로그래밍 모델 이해 (스레드, 블록, 그리드)
    - 메모리 관리 (Global, Shared, Constant, Texture Memory)
    - 간단한 CUDA 커널 작성 및 실행

**활동:**

- 간단한 벡터 덧셈 CUDA 커널 작성
- PyTorch CUDA Tensors 사용해보기

### **관련 지식**

#### **CUDA 프로그래밍 모델 이해**

- **스레드 (Thread)**: CUDA 프로그램의 기본 실행 단위입니다.
- **블록 (Block)**: 여러 스레드로 구성되며, 블록 내의 모든 스레드는 공유 메모리를 통해 데이터를 공유할 수 있습니다.
- **그리드 (Grid)**: 여러 블록으로 구성되며, 동일한 커널을 실행하는 모든 스레드가 포함됩니다.

#### **메모리 모델**

- **Global Memory**: 모든 스레드가 접근 가능한 메모리입니다. 가장 느리지만 용량이 큽니다.
- **Shared Memory**: 같은 블록 내의 스레드들이 공유하는 메모리입니다. Global Memory보다 빠릅니다.
- **Register**: 각 스레드가 사용하는 가장 빠른 메모리입니다.
- **Constant and Texture Memory**: 읽기 전용 메모리로, 빠른 접근을 제공합니다.

### 예제 및 문제

#### 예제 1) vector_add는 두 개의 벡터를 더하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```C
#include <cuda_runtime.h>
#include <iostream>

// CUDA 벡터 덧셈 커널
__global__ void vector_add(float* A, float* B, float* C, int N) {
    // TODO: 두 벡터를 더하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Mismatch at index " << i << "!" << std::endl;
            break;
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Done!" << std::endl;
    return 0;
}

```

#### **문제 1: 벡터 덧셈 CUDA 커널 작성**

##### **문제 설명**

1. 위 예제에서 `vector_add` 커널 함수를 완성합니다.
2. 각 스레드는 벡터 `A`와 `B`의 요소를 더하여 결과를 벡터 `C`에 저장합니다.

**힌트**: `blockIdx.x`, `blockDim.x`, `threadIdx.x`를 사용하여 각 스레드의 인덱스를 계산하세요.

정답:
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 벡터 덧셈 커널
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

#### 예제 2) matrix_add는 두 개의 행렬을 더하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 행렬 덧셈 커널
__global__ void matrix_add(float* A, float* B, float* C, int N, int M) {
    // TODO: 두 행렬을 더하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    int M = 1000;
    size_t size = N * M * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_A[i * M + j] = static_cast<float>(i + j);
            h_B[i * M + j] = static_cast<float>(i - j);
        }
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    matrix_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N, M);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (h_C[i * M + j] != h_A[i * M + j] + h_B[i * M + j]) {
                std::cerr << "Mismatch at index (" << i << ", " << j << ")!" << std::endl;
                break;
            }
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Done!" << std::endl;
    return 0;
}

```

정답:

#### 예제 3) scalar_mult는 벡터의 모든 요소에 스칼라 값을 곱하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 스칼라 곱 커널
__global__ void scalar_mult(float* A, float scalar, float* B, int N) {
    // TODO: 벡터의 모든 요소에 스칼라 값을 곱하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    float scalar = 2.0f;
    size_t size = N * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    scalar_mult<<<blocks_per_grid, threads_per_block>>>(d_A, scalar, d_B, N);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        if (h_B[i] != h_A[i] * scalar) {
            std::cerr << "Mismatch at index " << i << "!" << std::endl;
            break;
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    std::cout << "Done!" << std::endl;
    return 0;
}

```

정답:

#### 예제 4) matrix_transpose는 행렬의 전치를 계산하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 행렬 전치 커널
__global__ void matrix_transpose(float* A, float* B, int N, int M) {
    // TODO: 행렬의 전치를 계산하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    int M = 1000;
    size_t size = N * M * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_A[i * M + j] = static_cast<float>(i * M + j);
        }
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    matrix_transpose<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, N, M);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (h_B[j * N + i] != h_A[i * M + j]) {
                std::cerr << "Mismatch at index (" << i << ", " << j << ")!" << std::endl;
                break;
            }
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    std::cout << "Done!" << std::endl;
    return 0;
}

```
