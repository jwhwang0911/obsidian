### 관련 지식

#### **PyTorch C++ 확장**

PyTorch는 C++ 확장 모듈을 통해 커스텀 연산을 정의하고, 이를 Python에서 사용할 수 있게 합니다. `torch/extension.h`를 사용하면 CUDA 커널을 포함한 C++ 코드를 PyTorch와 통합할 수 있습니다.

#### **기본 개념**

- **C++ 확장 모듈**: PyTorch는 `pybind11` 라이브러리를 사용하여 C++ 코드를 Python에 바인딩합니다.
- **CUDA 커널**: GPU에서 실행되는 CUDA 코드를 포함합니다.
- **torch::Tensor**: PyTorch의 텐서를 C++에서 사용합니다.

### 예제 및 문제

#### 예제 1) Custom Add Operation

#### **문제 1: 간단한 덧셈 연산을 수행하는 C++/CUDA 모듈 작성**

##### **문제 설명**

1. `add.cpp` 파일을 작성하여 두 개의 텐서를 더하는 C++ 모듈을 만듭니다.
2. `torch/extension.h`를 포함하여 PyTorch 텐서를 사용합니다.
3. CUDA 커널을 작성하여 두 텐서의 요소를 더합니다.

**파일 구조**:

arduino

코드 복사

`custom_op/     ├── CMakeLists.txt     ├── add.cpp     ├── setup.py     └── test.py`

**CMakeLists.txt**:

cmake

코드 복사

`cmake_minimum_required(VERSION 3.10 FATAL_ERROR) project(custom_op)  find_package(Torch REQUIRED) find_package(CUDA REQUIRED)  add_library(custom_op SHARED add.cpp) target_link_libraries(custom_op "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}") set_property(TARGET custom_op PROPERTY CXX_STANDARD 14)`

**setup.py**:

python

코드 복사

`from setuptools import setup from torch.utils.cpp_extension import BuildExtension, CUDAExtension  setup(     name='custom_op',     ext_modules=[         CUDAExtension('custom_op', [             'add.cpp',         ])     ],     cmdclass={         'build_ext': BuildExtension     } )`

**add.cpp**:

cpp

코드 복사

`#include <torch/extension.h>  // CUDA 커널 __global__ void add_kernel(float* x, float* y, float* out, int64_t N) {     int idx = blockIdx.x * blockDim.x + threadIdx.x;     if (idx < N) {         out[idx] = x[idx] + y[idx];     } }  // C++ 함수 정의 torch::Tensor add(torch::Tensor x, torch::Tensor y) {     auto out = torch::zeros_like(x);     int64_t N = x.size(0);      int threads = 1024;     int blocks = (N + threads - 1) / threads;     add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), N);      return out; }  // PyTorch 모듈 정의 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {     m.def("add", &add, "Add two tensors"); }`

**test.py**:

python

코드 복사

`import torch import custom_op  x = torch.randn(1000, device='cuda') y = torch.randn(1000, device='cuda') z = custom_op.add(x, y)  print(z) print(torch.allclose(z, x + y))`

##### **활동**

1. `add.cpp` 파일에서 `add_kernel` CUDA 커널을 작성하여 각 요소를 더하는 함수를 완성합니다.
2. `setup.py`를 사용하여 C++ 확장 모듈을 빌드합니다.
    
    bash
    
    코드 복사
    
    `python setup.py install`
    
3. `test.py`를 실행하여 결과를 확인합니다.
    
    bash
    
    코드 복사
    
    `python test.py`
    

---

#### 예제 2) Custom Scalar Multiplication

#### **문제 2: 텐서의 모든 요소에 스칼라 값을 곱하는 C++/CUDA 모듈 작성**

##### **문제 설명**

1. `scalar_mult.cpp` 파일을 작성하여 텐서의 모든 요소에 스칼라 값을 곱하는 모듈을 만듭니다.
2. CUDA 커널을 작성하여 각 요소에 스칼라 값을 곱합니다.

**파일 구조**:

arduino

코드 복사

`custom_op/     ├── CMakeLists.txt     ├── scalar_mult.cpp     ├── setup.py     └── test_scalar_mult.py`

**scalar_mult.cpp**:

cpp

코드 복사

`#include <torch/extension.h>  // CUDA 커널 __global__ void scalar_mult_kernel(float* x, float scalar, float* out, int64_t N) {     int idx = blockIdx.x * blockDim.x + threadIdx.x;     if (idx < N) {         out[idx] = x[idx] * scalar;     } }  // C++ 함수 정의 torch::Tensor scalar_mult(torch::Tensor x, float scalar) {     auto out = torch::zeros_like(x);     int64_t N = x.size(0);      int threads = 1024;     int blocks = (N + threads - 1) / threads;     scalar_mult_kernel<<<blocks, threads>>>(x.data_ptr<float>(), scalar, out.data_ptr<float>(), N);      return out; }  // PyTorch 모듈 정의 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {     m.def("scalar_mult", &scalar_mult, "Multiply tensor by scalar"); }`

**test_scalar_mult.py**:

python

코드 복사

`import torch import custom_op  x = torch.randn(1000, device='cuda') scalar = 2.0 z = custom_op.scalar_mult(x, scalar)  print(z) print(torch.allclose(z, x * scalar))`

##### **활동**

1. `scalar_mult.cpp` 파일에서 `scalar_mult_kernel` CUDA 커널을 작성하여 각 요소에 스칼라 값을 곱하는 함수를 완성합니다.
2. `setup.py`를 사용하여 C++ 확장 모듈을 빌드합니다.
    
    bash
    
    코드 복사
    
    `python setup.py install`
    
3. `test_scalar_mult.py`를 실행하여 결과를 확인합니다.
    
    bash
    
    코드 복사
    
    `python test_scalar_mult.py`
    

---

#### 예제 3) Custom Matrix Multiplication

#### **문제 3: 행렬 곱셈을 수행하는 C++/CUDA 모듈 작성**

##### **문제 설명**

1. `matrix_mult.cpp` 파일을 작성하여 두 개의 행렬을 곱하는 모듈을 만듭니다.
2. CUDA 커널을 작성하여 행렬 곱셈을 수행합니다.

**파일 구조**:

arduino

코드 복사

`custom_op/     ├── CMakeLists.txt     ├── matrix_mult.cpp     ├── setup.py     └── test_matrix_mult.py`

**matrix_mult.cpp**:

cpp

코드 복사

`#include <torch/extension.h>  // CUDA 커널 __global__ void matrix_mult_kernel(float* A, float* B, float* C, int M, int N, int K) {     int row = blockIdx.x * blockDim.x + threadIdx.x;     int col = blockIdx.y * blockDim.y + threadIdx.y;      if (row < M && col < K) {         float value = 0;         for (int i = 0; i < N; ++i) {             value += A[row * N + i] * B[i * K + col];         }         C[row * K + col] = value;     } }  // C++ 함수 정의 torch::Tensor matrix_mult(torch::Tensor A, torch::Tensor B) {     auto C = torch::zeros({A.size(0), B.size(1)}, A.options());      int M = A.size(0);     int N = A.size(1);     int K = B.size(1);      dim3 threads(16, 16);     dim3 blocks((M + threads.x - 1) / threads.x, (K + threads.y - 1) / threads.y);     matrix_mult_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);      return C; }  // PyTorch 모듈 정의 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {     m.def("matrix_mult", &matrix_mult, "Multiply two matrices"); }`

**test_matrix_mult.py**:

python

코드 복사

`import torch import custom_op  A = torch.randn(32, 64, device='cuda') B = torch.randn(64, 128, device='cuda') C = custom_op.matrix_mult(A, B)  print(C) print(torch.allclose(C, torch.matmul(A, B)))`

##### **활동**

1. `matrix_mult.cpp` 파일에서 `matrix_mult_kernel` CUDA 커널을 작성하여 행렬 곱셈을 수행하는 함수를 완성합니다.
2. `setup.py`를 사용하여 C++ 확장 모듈을 빌드합니다.
    
    bash
    
    코드 복사
    
    `python setup.py install`
    
3. `test_matrix_mult.py`를 실행하여 결과를 확인합니다.
    
    bash
    
    코드 복사
    
    `python test_matrix_mult.py`
    

---

#### 예제 4) Custom Element-wise Operation

#### **문제 4: 텐서의 요소별 연산을 수행하는 C++/CUDA 모듈 작성**

##### **문제 설명**

1. `elementwise_op.cpp` 파일을 작성하여 텐서의 각 요소에 대해 사용자 정의 연산을 수행하는 모듈을 만듭니다.
2. CUDA 커널을 작성하여 요소별 연산을 수행합니다.

**파일 구조**:

arduino

코드 복사

`custom_op/     ├── CMakeLists.txt     ├── elementwise_op.cpp     ├── setup.py     └── test_elementwise_op.py`

**elementwise_op.cpp**:

cpp

코드 복사

`#include <torch/extension.h>  // CUDA 커널 __global__ void elementwise_op_kernel(float* x, float* out, int64_t N) {     int idx = blockIdx.x * blockDim.x + threadIdx.x;     if (idx < N) {         out[idx] = x[idx] * x[idx] + 2 * x[idx] + 1;  // 예: (x^2 + 2x + 1)     } }  // C++ 함수 정의 torch::Tensor elementwise_op(torch::Tensor x) {     auto out = torch::zeros_like(x);     int64_t N = x.size(0);      int threads = 1024;     int blocks = (N + threads - 1) / threads;     elementwise_op_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);      return out; }  // PyTorch 모듈 정의 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {     m.def("elementwise_op", &elementwise_op, "Element-wise operation on tensor"); }`

**test_elementwise_op.py**:

python

코드 복사

`import torch import custom_op  x = torch.randn(1000, device='cuda') z = custom_op.elementwise_op(x)  print(z) print(torch.allclose(z, x * x + 2 * x + 1))`

##### **활동**

1. `elementwise_op.cpp` 파일에서 `elementwise_op_kernel` CUDA 커널을 작성하여 각 요소에 사용자 정의 연산을 수행하는 함수를 완성합니다.
2. `setup.py`를 사용하여 C++ 확장 모듈을 빌드합니다.
    
    bash
    
    코드 복사
    
    `python setup.py install`
    
3. `test_elementwise_op.py`를 실행하여 결과를 확인합니다.
    
    bash
    
    코드 복사
    
    `python test_elementwise_op.py`
    

---

이렇게 Day 3에 진행할 C++/CUDA 통합 문제 4개를 제공해드렸습니다. 각 문제를 해결하면서 PyTorch와 C++/CUDA 통합 방법을 이해하고 연습할 수 있습니다. 추가적인 질문이나 도움이 필요하면 언제든지 알려주세요!