# TensorSSA and For Loop Auto Parallel in long-tail
## Dependency
- LibTorch
- LibTorchVision
## Build From Source
### Linux
```bash
# install torch vision
git clone https://github.com/pytorch/vision.git
cd vision
mkdir build && cd build
cmake .. && make && make install
# build source
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)' ` ..
make -j{$proc}
```

## Example
```bash
# Generate pt file
python example/simple_loop.py
# Transform, Optimization and CodeGen
./build/example simple_loop.pt
```

