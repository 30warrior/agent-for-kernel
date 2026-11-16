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

默认 TestingAgent 一次生成 8 个额外样例，先全部在参考模型上验证，再冻结为
`test_suite.json` 和 `reference.pt`。所有版本复用同一组原始及额外样例；正确性
不通过的版本不计时。编译失败不能进行正确性检查。可用
`--extra-test-cases` 调整额外数量（至少 2），或 `--test-spec` 重放已保存规格。

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
