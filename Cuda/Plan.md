---
cssclasses:
  - width-100
  - col-lines
  - row-alt
  - row-lines
  - table-small
---
### **Day 1: 환경 설정 및 기본 개념 학습**
#### **목표: 학습 환경 설정 및 CUDA와 PyTorch 기본 개념 이해**
- **설치 및 환경 설정**
    - [x] Visual Studio Code, CUDA Toolkit, PyTorch, CMake 설치 및 설정
    - [x] VSCode 확장 프로그램 설치 (C/C++, CUDA C++, Code Runner 등)
- **기본 개념 학습**
    - [x] CUDA 개념 이해: GPU 아키텍처, CUDA 프로그래밍 모델
    - [x] PyTorch 기본 개념 이해: Tensor, Autograd, Neural Networks
**활동:**
- [x] NVIDIA CUDA Programming Guide 읽기
- [x] PyTorch 공식 문서의 시작하기 튜토리얼 완료

### **Day 2: CUDA 프로그래밍 기본 학습**
#### **목표: CUDA 프로그래밍의 기초 이해 및 간단한 CUDA 커널 작성**
- **CUDA 프로그래밍**
    - [x] CUDA 프로그래밍 모델 이해 (스레드, 블록, 그리드)
    - [x] 메모리 관리 (Global, Shared, Constant, Texture Memory)
    - [x] 간단한 CUDA 커널 작성 및 실행
**활동:**
- [x] 간단한 벡터 덧셈 CUDA 커널 작성
- [x] PyTorch CUDA Tensors 사용해보기
### **Day 3: PyTorch와 C++/CUDA 통합**
#### **목표: PyTorch C++ 확장 (torch/extension.h) 사용법 학습**
- **PyTorch C++ 확장**
    - [ ] `torch/extension.h` 사용법 학습
    - [ ] PyTorch와 C++/CUDA 코드 통합 방법 이해
    - [ ] 간단한 C++ Extension 모듈 작성 및 빌드
**활동:**
- [ ] 간단한 C++ Extension 예제 작성 및 빌드
- [ ] PyTorch와 통합하여 Python에서 사용해보기
### **Day 4: Custom CUDA Kernel 작성**
#### **목표: Custom CUDA Kernel을 작성하고 PyTorch와 통합하기**
- **Custom CUDA Kernel 작성**
    - [ ] Custom CUDA 커널 작성 및 디버깅
    - [ ] PyTorch Tensors와 CUDA 커널 연동
    - [ ] CUDA 커널을 래핑하는 C++ 함수 작성
**활동:**
- [ ] 간단한 Custom CUDA 커널 작성 (예: 행렬 곱셈)
- [ ] PyTorch와 통합하여 실행
### **Day 5: 복잡한 Custom Op 구현**
#### **목표: 복잡한 Custom Operator 구현**
- **복잡한 Custom Op 구현**
    - [ ] 복잡한 연산 수행하는 Custom CUDA Op 작성 (예: Convolution)
    - [ ] 성능 최적화 및 디버깅
**활동:**
- [ ] Custom Convolution 커널 작성
- [ ] 성능 테스트 및 최적화
### **Day 6: 자동 미분 및 테스트**
#### **목표: Custom Operator에 대한 자동 미분 구현 및 테스트**
- **자동 미분**
    - [ ] PyTorch의 Autograd와 통합하여 Custom Op에 대한 자동 미분 구현
    - [ ] Forward 및 Backward 함수 작성
- **테스트**
    - [ ] Custom Op에 대한 단위 테스트 작성
    - [ ] 기존 PyTorch 연산과 결과 비교
**활동:**
- [ ] Custom Op의 Forward 및 Backward 함수 작성
- [ ] 단위 테스트 작성 및 결과 검증

### **Day 7: 최종 프로젝트 및 문서화**
#### **목표: Custom Operator 프로젝트 완료 및 문서화**
- **최종 프로젝트**
    - [ ] Custom Op 프로젝트 완성
    - [ ] 다양한 입력 데이터로 테스트 및 디버깅
- **문서화**
    - [ ] 프로젝트 문서화 (README 작성, 코드 주석 추가)
    - [ ] 학습 결과 정리
**활동:**
- [ ] 최종 프로젝트 코드 완성
- [ ] 프로젝트 문서 작성 및 정리

### 학습 자료
- NVIDIA CUDA Programming Guide: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- PyTorch 공식 문서: https://pytorch.org/docs/stable/index.html
- PyTorch C++ Extension 튜토리얼: https://pytorch.org/tutorials/advanced/cpp_extension.html