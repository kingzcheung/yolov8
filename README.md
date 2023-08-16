# yolov8

使用 yolov8 模型推理, 支持两种方式：

1. libtorch 推理
2. onnxruntime 推理

## 环境要求
### libtorch 推理
1. 安装 libtorch  
   ```shell
      # 下载 libtorch
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
    # 解压
    unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip
    # 安装
    cd libtorch
    sudo cp lib/* /usr/lib/
   ```
