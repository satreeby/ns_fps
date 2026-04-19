# ns-fps Evaluation

This folder contains evaluation scripts for testing the latency performance of our ns-fps implementation on both CPU and GPU.

## Hardware Requirements

Tests can be completed on RTX 5090 GPU.

## Setup and Execution

To run the evaluation, please follow the instructions below:

### CPU Version

```bash
cd cpu/build
rm -rf *
cmake ..
make -j8
./compare_cpu_test_kitti -f ../../kitti_data/000000.bin -s 300000
```

### GPU Version

```bash
cd gpu/build
rm -rf *
cmake ..
make -j8
./compare_test_kitti -f ../../kitti_data/000000.bin -s 300000
```
