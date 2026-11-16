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

默认 TestingAgent 一次生成 8 个额外样例，先全部在参考模型上验证，再冻结为
`test_suite.json` 和 `reference.pt`。所有版本复用同一组原始及额外样例；正确性
不通过的版本不计时。编译失败不能进行正确性检查。可用
`--extra-test-cases` 调整额外数量（至少 2），或 `--test-spec` 重放已保存规格。

主加速比及最佳版本排名仍按原始数据集样例计算；额外样例必须通过正确性，
性能在 CSV 中单独记录，供策略 agent 分析。

提示词修改位置：`prompts/cpu_neon.md`、`prompts/cpu_sve.md`。对应内容同时提供给
coding 和 planning agent。修改后启动新运行。CUDA 优化建议保留 Astra 原版。
Orchestrator 负责接收进度通知，Python 驱动测试、profiling、planning、coding 循环。
RAG 尚未实现。

结果保存在 `optimization_runs/<运行时间>/`，重点查看 `summary.json`、
`benchmark_details.csv`、`v1/kernel.cpp` 和后续版本；提示词和策略建议也在运行目录。
`--prepare-only` 不调用模型，只检查原始参考样例；`--evaluate-only` 可复测本地保存
的候选，配合 `--test-spec` 重放固定测试集。

本次移植的多样例评测依赖已在另一分支的真实 Arm 运行中验证；本分支部署后会
单独核查。不要将网络错误计作 kernel 正确性错误，也不要把手写探针作为 LLM 结果。
