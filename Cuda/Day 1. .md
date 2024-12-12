---
cssclasses:
  - width-100
---
### 예시 문제

#### **문제 1: CUDA 환경 설정 및 벡터 덧셈 커널 작성**

##### **문제 설명**

1. CUDA Toolkit을 설치하고, nvcc 컴파일러가 올바르게 설치되었는지 확인합니다.
2. 간단한 벡터 덧셈 연산을 수행하는 CUDA 커널을 작성하세요.
3. 호스트 메모리에서 데이터를 초기화하고, 이를 디바이스 메모리로 복사한 뒤, CUDA 커널을 호출하여 벡터 덧셈을 수행합니다.
4. 결과를 다시 호스트 메모리로 복사하고 올바르게 수행되었는지 검증합니다.

**세부 단계**:

- CUDA Toolkit 설치
- 벡터 덧셈 커널 코드 작성 (`vector_add.cu`)
- nvcc를 사용하여 컴파일하고 실행

#### **문제 2: PyTorch CUDA Tensor 테스트**

##### **문제 설명**

1. PyTorch가 설치되어 있는지 확인하고, CUDA를 사용할 수 있는지 테스트합니다.
2. CUDA를 사용하여 텐서를 생성하고, 간단한 연산을 수행합니다.
3. GPU에서 연산이 제대로 수행되는지 확인합니다.

**세부 단계**:

- PyTorch 설치 확인
- CUDA 사용 가능 여부 확인
- CUDA 텐서 생성 및 연산 수행

---

### 예시 코드

#### **문제 1: CUDA 환경 설정 및 벡터 덧셈 커널 작성**
```C++
#include <cuda_runtime.h>
#include <iostream>

// CUDA 벡터 덧셈 커널
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
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


---

#### **문제 2: PyTorch CUDA Tensor 테스트**

##### **코드 예시**

python

코드 복사

`import torch  # CUDA 사용 가능한지 확인 print(torch.cuda.is_available())  # CUDA 디바이스 정보 출력 print(torch.cuda.get_device_name(0))  # 텐서 생성 및 GPU로 이동 x = torch.randn(3, 3) x = x.to('cuda') print(x)  # 간단한 연산 수행 y = x * 2 print(y)`

---

이렇게 문제와 코드를 따로 제공하였습니다. 각 문제의 목표를 이해하고 직접 코드를 작성해보는 것이 학습에 도움이 됩니다. 추가적인 질문이나 도움이 필요하면 언제든지 알려주세요!

4o