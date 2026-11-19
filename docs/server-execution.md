# 服务器直接运行

分支 `cpu-gpu-kernelbench` 的 agent、模型 API、编译和评测全部在服务器执行。
不需要本地控制器，也不需要 `--executor ssh`。新增文件只写入服务器 `hzh`。

```bash
cd /mnt/data_nvme4n1/lab317_cnndev/hzh/Astra
bash run_server.sh --dataset akg --dataset-root benchmarks/akg \
  --task static_shape/elemwise/elemwise_add_001.py --target sve --prepare-only
```

LLM 运行需要当前项目 Python 中存在 `openai-agents`（导入名为 `agents`），
以及 PyTorch、NumPy、Ninja 和本机 C++ 编译器。服务器既有 `.venv` 的依赖不由
代码部署自动修改。只在项目环境安装依赖，不修改共享环境。

在服务器项目 `.env` 中配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。
文件已被 Git 忽略；代码部署不会复制本地密钥或覆盖服务器已有 `.env`。
已设置的进程环境变量优先于 `.env`。SDK tracing 默认关闭。

```bash
bash run_server.sh --dataset akg --dataset-root benchmarks/akg \
  --task static_shape/elemwise/elemwise_add_001.py --target sve \
  --model-api responses --max-iterations 1
```

使用 `--target neon` 切换 NEON；仅支持 Chat Completions 的接口使用
`--model-api chat_completions`。`--max-iterations 1` 是 v1 加一轮 v2，
`--max-iterations 5` 是 v1 到 v6。TLS 协商问题可尝试显式
`--tls-max-version 1.2`；该选项保留证书校验，不保证接口持续可用。
如果环境代理指向未运行的本机端口，应先修复代理，或仅在本次进程中取消相应
`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` 及小写变量；启动脚本不会擅自更改代理。

`--target neon` 保持 NEON 兼容性：编译参数为 `-march=armv8-a+simd`，即使
当前机器支持 SVE，也不允许生成 SVE 指令。对应优化建议为 `prompts/cpu_neon.md`。
`--target sve` 表示目标机器支持 SVE，编译参数为 `-march=armv8.2-a+sve`
和 `-msve-vector-bits=scalable`，允许纯 NEON、纯 SVE 或混合实现。对应优化建议为
`prompts/cpu_sve.md`，同时提供给 Coding Agent 和 Planning Agent。无需强制混用，
也无需强制出现 SVE 指令；应按 shape、stride 和实测性能选择。SVE 路径仍需使用
运行时向量长度，不得假定 SVE2、SME 或编译器未提供的桥接 intrinsics 可用。
每个目标只加载自己的 Markdown 文件；修改后在下一次启动时生效。

默认采用 compact 覆盖，TestingAgent 一次生成 8–15 个额外样例，先全部在参考模型上验证，再冻结为
`test_suite.json` 和 `reference.pt`。所有版本复用同一组原始及额外样例；正确性
不通过的版本不计时。编译失败不能进行正确性检查。可用
`--extra-test-cases` 调整额外数量下限（至少 2），`--max-extra-test-cases` 调整上限，
或 `--test-spec` 重放包含覆盖声明的已保存规格。只有 1 个原始样例时，每轮共测 9 个。
`--test-coverage full` 恢复完整覆盖，默认额外样例上限为 128；compact 的默认上限
为 max(15, extra-test-cases)。显式指定的上限始终生效；测试集冻结后不会追加样例。

所有版本仍检查并计时完整固定测试集，但最佳版本排名改为使用生成样例的
平均加速比，默认几何平均。迭代结束后只对选中的版本进行一次原始样例复测。
复测重新检查正确性，并交替测量 PyTorch/候选以得到新的基线；汇总主加速比
来自该次复测。复测失败不会回退到旧成绩，也不会改选其他版本。

## 选优指标和最终复测

对每个固定生成样例，记参考中位延迟为 B_i，候选中位延迟为 C_i，
加速比 s_i = B_i / C_i。两个指标都按样例等权计算：

- 算术平均：sum(s_i) / n。
- 几何平均：exp(sum(log(s_i)) / n)，默认选优指标。

不同样例的性能收益是比例量。几何平均对成比例的加速/减速对称；例如
4 倍和 0.25 倍的算术平均是 2.125 倍，几何平均为 1 倍。默认几何平均避免
一个极大的加速比掩盖严重退化。它仍允许样例之间有得有失，并非最差情况约束。
如果目标是整组任务的总执行时间，应另看总时间比，而非把它与平均加速比混同。

用 `--selection-metric arithmetic` 可显式切换算术平均；默认是
`--selection-metric geometric`。所有样例正确且全部计时成功的候选才参与排名，
分数越大越好。后续迭代从当前最佳有效候选继续；失败候选优先修复。

最终复测沿用原始输入快照、模型参数、线程配置、warmup 和 trials，但重新采样。
每个采样轮次交替 reference/candidate 的先后顺序，准备输入和恢复状态位于计时外，
输出分配及 wrapper 调用位于计时内。它减少基线时间间隔带来的影响，不能保证消除
频率或其他负载造成的波动。原始样例仍出现在迭代反馈中，因此这不是未见样例测试。

新增报告：

- `final_benchmark.json`：最后一次原始样例复测，包含新的参考与候选原始采样。
- `final_benchmark_details.csv`：最后一次复测的逐样例延迟及速度比。
- `benchmark_summary.csv`：增加选优用的算术/几何平均加速比列；既有原始样例
  时间列仍是迭代测量，不能当成最终复测值。
- `summary.json`：`reference_speedup` 使用最终复测的配置平均指标；
  `final_speedups` 同时列出两种平均，`final_total_time_speedup` 单独记录总时间比；
  `refinement_speedup` 表示生成样例上最佳分数相对 v1 的倍数，
  `refinement_speedup_scope` 明确该范围。

最终复测前，汇总状态为 pending，主加速比为空。没有有效候选时最终步骤标记
skipped；复测失败时 success=false，主加速比为空。迭代原始结果始终保留。

选优和最终复测的实现检查已覆盖：算术/几何平均产生不同排名、只按生成样例
选优、最后只调用一次复测、等待复测时不暴露旧主成绩、复测失败及无有效候选。
这些控制流程检查不调用模型。另在服务器使用已保存的 LLM SVE 候选实际执行最终
worker，仅测原始 AKG Add 的 1 个样例、重新交替采样参考/候选各 10 次并通过正确性。
结果目录为 `optimization_runs/final_validation_20261116_200133_438736/`。
本次没有重新生成 kernel，也不能用该次短采样证明稳定性能提升。

提示词修改位置：`prompts/cpu_neon.md`、`prompts/cpu_sve.md`。对应内容同时提供给
coding 和 planning agent。修改后启动新运行。CUDA 优化建议保留 Astra 原版。
Orchestrator 负责接收进度通知，Python 驱动测试、profiling、planning、coding 循环。
RAG 尚未实现。

## 边界与布局覆盖

Coding 和 Planning 都接收 `astra/prompts.py` 的 `BOUNDARY_CORRECTNESS`，要求
检查展开后超过一个向量的余数、连续/非连续路径、规约的 inactive lanes 和 GEMM
的 M/N/K 尾块。`prompts/cpu_sve.md` 也包含可编辑的 SVE cleanup 建议。
Testing Agent 在生成阶段补齐覆盖，在逐版验证阶段审查代码与覆盖报告；代码审查
意见与工具实测 pass/fail 分开记录，不会在迭代中追加样例。

`astra/coverage.py` 从目标和 dtype 计算边界指导：NEON 使用 16 字节，SVE 使用
实测当前线程 VL；读取不到 SVE VL 时使用明确标注的通用尺寸。CUDA 使用 warp
宽度 32 作为线程边界指导，并不声称它就是候选 tile。compact 尺寸为 1、V−1、V、
V+1、V+max(1, floor(V/2)) 和 4V+1；full 另外覆盖 2V 和 4V 两侧及整点。
始终不要求生成代码写死该 V。

测试规格顶层现在要求 `cases` 和 `coverage` 两个列表。每个原始顶层 Tensor 的
每个轴都需声明，例如：

```json
{"base_case": 0, "input": 0, "axis": 1, "kind": "variable", "reason": "参考源码允许改变规约长度"}
```

compact 中，可变轴的指定尺寸在合法布局之间合并覆盖，不再要求尺寸与布局的全组合。
但 V+max(1, floor(V/2)) 这一展开循环余数仍须出现在该轴的布局并集中，以保留对“只处理一个
剩余向量”错误的检查；每个合法布局至少出现一次。full 仍要求每个尺寸覆盖每种布局。
判断依据
为生成后张量的 `is_contiguous()` 和该轴实际 stride；transpose 后的轴才是声明
中的 axis。用小的其他维度控制成本，耦合维度仍须合法。一个样例可满足多个检查。
长度为 1 的轴没有步长遍历，不要求 strided 覆盖，避免要求 1x1 方阵等不可能的布局。
纯零分布不能用于满足边界覆盖，Testing 提示另要求合法时加入全 1 和混合符号。

受参考参数/语义固定的轴可用 `kind: "fixed"` 并说明源码依据，框架检查这些轴
确实没有改变；可变轴还可声明 minimum/maximum/multiple 或 layout_exemptions，
但必须解释参考模型的限制，不能为迎合候选代码而豁免。框架不会自动证明这些
语义声明；受固定维度和描述符能力限制、确实无法构造的布局也可说明后豁免。
这些理由需审阅；嵌套 Tensor 输入列为未评估，不宣称完整覆盖。

冻结前缺项会返回具体轴、布局及缺失尺寸，要求 Testing Agent 修复。通过后写入
`coverage.json`、`test_suite.json` 和 reference 元数据，并随 execution 上下文
提供给各 agent。所有版本仍复用同一组冻结数据，并按原有规则计时、选优和复测。
这会改变样例构成及选优分数，不能直接与旧套件分数比较；若预算不足应增加上限
或内存预算，不能静默删掉要求。旧 `--test-spec` 缺少 coverage 时需迁移，旧实验
产物不会被追溯改动。有限边界覆盖不是对任意展开因子、任意形状的正确性证明。

## 独立 kernel 生成约束

本实验要求生成独立的原生计算实现。`astra/prompts.py` 的
`INDEPENDENT_KERNEL_POLICY` 同时注入 Coding 和 Planning 的提示词，覆盖
CUDA、NEON、SVE、首次生成及每轮改进，`--generate-only` 也使用该规则。

禁止通过 Python PyTorch/NumPy 或 C++ ATen/LibTorch 数值算子完成计算，
包括 Tensor 方法和 dispatcher 等间接调用；禁止调用 BLAS、MKL、OpenBLAS、
oneDNN、cuBLAS、cuDNN 等库实现任务计算。所有形状、步长、大小及错误处理分支
都不能回退到 reference、原始算子或其他框架/库实现。允许自行编写的原生标量、
SIMD、OpenMP、CUDA 路径，以及这些路径之间的形状分派。

Tensor 类型、元数据查询、数据指针、空输出/工作区分配、只修改元数据的 view，
以及初始化时复制参考参数仍可用于接口封装；数值初始化、类型转换、packing 和
实际布局转换需自行实现。标准标量数学函数、指令 intrinsics、OpenMP 和 CUDA
运行时接口可在目标配置内使用。该限制只约束生成候选，PyTorch baseline 保持
参考实现，不受此限制。

修改后启动新实验即可生效，实际规则记录于 `v*/generation_instructions.txt`
和 `role_instructions.json`。这是提示词约束，尚无自动合规检测；正确性通过不等于
证明候选独立实现，旧实验产物也不会因此被追溯检查。

## CPU OpenMP 与公平线程配置

### CPU 硬件上下文

`astra/hardware.py` 只读采集 CPU 规格，`Backend.capabilities()` 将其放入
`reference.json` 的 `hardware.cpu`。`execution_context()` 整理共享缓存实例后，
经 `execution.hardware` 传给 Coding、Planning、Testing 和 Profiling Agent；
`--generate-only` 也使用相同整理逻辑。无需在优化 Markdown 中写死服务器规格。

采集内容包括 CPU implementer/part/MIDR、OS 报告的型号、整机在线物理/逻辑核心数、
所分配核心与 NUMA 拓扑、相关 L1/L2/L3 缓存容量及共享范围、缓存行大小、
所分配 CPU 的共同 ISA features、系统默认与当前线程实际 SVE 向量长度、
可读的频率上下限、主机和相关 NUMA 节点内存总量。每组信息保留来源。
缓存容量按共享实例记录，不能把整机总缓存或多核缓存相加后当作单核容量。
内存总量不代表作业可用量，OpenMP CPU 绑定也不等于 NUMA 内存绑定。

主机 features 与 `hardware.compilation_target.allowed_isa` 分开，agent 必须遵守
目标编译参数；主机有 SVE2 等扩展不表示本次允许使用。SVE 当前长度通过
`prctl(PR_SVE_GET_VL)` 只读查询，不改变进程设置，生成代码仍需使用运行时长度。
频率、型号等读取不到的项目为 `null`；不猜测峰值 FLOPS 或内存带宽，也不执行
额外性能基准。实测耗时依然由 Profiling Agent 对应工具提供。

若 OS 缺少商品型号，可参考 `cpu_hardware.example.json` 在项目根目录创建
`cpu_hardware.json`，填写确认的 `cpu_identity.model_name` 和 `source`。
该配置被 Git 忽略，只补充型号，不覆盖探测的缓存、ISA 或线程预算；系统报告与
用户补充的值分别保存，OS 报告值优先。配置文件可选；存在但格式错误会明确报错。
换服务器时应重新确认或移除此文件，不复用旧机器型号。服务器 `armv8_sve` 的
Kunpeng 920 型号来自用户于 2026-11-17 的确认，而不是根据 CPU part 推断。

`--threads` 默认仍为 1；设为 N（N > 1）时，CPU 编译和链接均启用 `-fopenmp`，
Coding/Planning Agent 获得对应线程预算，允许生成 OpenMP + NEON/SVE 实现。
小输入可走串行路径，设置多线程不代表所有输入都必须启动并行区域。
CUDA 的编译和执行方式保持原有配置。

```bash
bash run_server.sh --dataset akg --dataset-root benchmarks/akg \
  --task static_shape/elemwise/elemwise_add_001.py --target sve \
  --threads 4 --model-api responses --max-iterations 1
```

默认从进程原有允许的 CPU 中，选择同一 NUMA 节点内的 N 个不同物理核心，
避免自动把 SMT 兄弟线程算作不同物理核心。核心不足会明确报错。
`--cpu-core ID` 可选择自动分配的起点；也可显式指定恰好 N 个逻辑 CPU：

```bash
# CPU ID 应按实际服务器拓扑选择，此处仅为示例。
bash run_server.sh --dataset akg --dataset-root benchmarks/akg \
  --task static_shape/elemwise/elemwise_add_001.py --target sve \
  --threads 4 --cpu-cores 0,2,4,6 --prepare-only
```

显式列表也支持 `0-3`，但不会自动排除 SMT 或跨 NUMA；实际拓扑写入报告。
该功能设置线程和 CPU 亲和性，不负责 NUMA 内存绑定、独占核心或隔离其他用户负载。

配置在导入 PyTorch 前应用，并由所有评测子进程重放：PyTorch intra-op=N、
inter-op=1，OpenMP 线程上限=N、禁止动态团队和嵌套并行，MKL/OpenBLAS/BLIS
使用相同线程预算；OpenMP places 固定为选中的 CPU。已有运行目录保留自己的配置，
不会因为新的命令行参数改变原实验的线程数。

`reference.json`、`progress.json`、`summary.json` 和各评测结果中的
`cpu_execution` 记录分配的 CPU、NUMA/物理核心拓扑、运行时线程配置和 PyTorch
实际线程设置。`main_thread_affinity` 是主线程当时的绑定，可能因 OpenMP 绑定而只含
一个 CPU；完整资源预算看 `affinity`。配置线程数是上限，不等于每个 kernel 实际
启动的 OpenMP 团队大小，也不代表已实现多线程加速。

加载原生扩展前会恢复完整 CPU 集合，避免 PyTorch 的 OpenMP 主线程先绑定到
一个核心后，另一个 OpenMP 运行库仅识别该核心。2026-11-17 在服务器上通过框架的
编译及加载路径核查：OpenMP 团队大小为 4，线程分别运行在 CPU 0、2、4、6；
PyTorch intra-op=4、inter-op=1。NEON 单线程参考准备也通过。

另外使用真实 LLM 生成 SVE + OpenMP 广播乘法，v1、v2 均通过同一组 9 个固定样例
的正确性和性能检查，并完成原始样例最终复测。运行目录为
`optimization_runs/20261117_173142_796563/`，选中 v2。该次 10 次短采样参考中位延迟
2.8210 ms、候选 2.7870 ms，仅用于验证执行链路，不能证明稳定性能提升。

结果保存在 `optimization_runs/<运行时间>/`，重点查看 `summary.json`、
`benchmark_details.csv`、`v1/kernel.cpp` 和后续版本；提示词和策略建议也在运行目录。
`--prepare-only` 不调用模型，只检查原始参考样例；`--evaluate-only` 可复测本地保存
的候选，配合 `--test-spec` 重放固定测试集。

## 部署后核查（2026-11-16）

服务器代码已更新到本分支。`bash -n run_server.sh` 和原始 SVE Add 参考准备通过。
复用此前实际由 LLM 生成的 SVE v1、v2，各自完成编译、九组固定样例的正确性检查
和计时；两次测试集标识一致。样例覆盖广播、向量尾部、非连续输入及零/一值。

- v1 结果：`optimization_runs/20261116_183527_803759/evaluation.json`
- v2 结果：`optimization_runs/20261116_183555_064922/evaluation.json`

候选只放在服务器忽略的运行目录，未提交到 Git。本次没有发起新的模型请求，
也未安装 SDK 或复制本地密钥。部署时既有 `.venv` 缺少 `agents`；启动新的 LLM
工作流前仍需配置服务器模型 SDK 和 API。不要将网络错误计作 kernel 正确性错误。
