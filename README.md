# EasyTimer

EasyTimer是一个简单、易用的CUDA和C++计时器库。它包括了 CPU 计时器和 GPU 计时器。可以用于精确测量 CPU 和 GPU 任务的执行时间，以便更好地监控和优化你的代码性能。

## 主要功能

`CpuTimer`：使用 clock_gettime 函数，提供毫秒级的精确度，用于测量 CPU 任务的执行时间。
`GpuTimer`：使用 CUDA 事件（CUDA Events）来测量 GPU 任务的执行时间。

## 如何使用
1. 包含头文件 easy_timer.hpp。
```c++
#include "easy_timer.hpp"
```
2. 创建一个 CpuTimer 或 GpuTimer 对象。
```c++
easy_timer::CpuTimer cpuTimer;
easy_timer::GpuTimer gpuTimer;
```
3. 使用 start() 方法开始计时，使用 stop() 方法停止计时。
```c++
cpuTimer.start();
/*
 * CPU 任务代码
 */
cpuTimer.stop();

gpuTimer.start();
/*
 * GPU 任务代码
 */
gpuTimer.stop();
```
4. 使用 getElapsedTime() 方法获取经过的时间（毫秒）。
```c++
double cpuTime = cpuTimer.getElapsedTime();
double gpuTime = gpuTimer.getElapsedTime();
```
5. 使用 reset() 方法重置计时器。
```c++
cpuTimer.reset();
gpuTimer.reset();
```

## 注意事项
本库需要 CUDA 环境支持，因为 GPU 计时器使用了 CUDA 事件进行计时。

## 贡献

如果你发现任何问题或有改进建议，请随时提出issue或创建一个pull请求。你的贡献将对这个项目产生积极的影响！

## 许可证

这个项目基于MIT许可证进行许可。详细信息请参阅[LICENSE](LICENSE)文件。