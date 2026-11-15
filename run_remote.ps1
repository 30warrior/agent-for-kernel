param(
    [ValidateSet('neon', 'sve', 'cuda')][string]$Target = 'neon',
    [string]$Python = 'python',
    [string]$SshHost = 'armv8_sve',
    [string]$RemoteRoot = '/mnt/data_nvme4n1/lab317_cnndev/hzh/Astra-remote',
    [string]$RemotePython = '/mnt/data_nvme4n1/lab317_cnndev/hzh/Astra/.venv/bin/python',
    [string]$Model = $env:OPENAI_MODEL,
    [ValidateSet('responses', 'chat_completions')][string]$ModelApi = 'responses',
    [string]$Task = 'static_shape/elemwise/elemwise_add_001.py',
    [int]$MaxIterations = 3,
    [switch]$PrepareOnly,
    [string]$EvaluateOnly
)

$commandArgs = @('-B', "$PSScriptRoot/kernel_optimizer.py", '--dataset', 'akg',
    '--dataset-root', "$PSScriptRoot/benchmarks/akg", '--task', $Task,
    '--target', $Target, '--executor', 'ssh', '--ssh-host', $SshHost,
    '--remote-root', $RemoteRoot, '--remote-python', $RemotePython,
    '--model-api', $ModelApi, '--max-iterations', $MaxIterations)
if ($Model) { $commandArgs += @('--model', $Model) }
if ($PrepareOnly) { $commandArgs += '--prepare-only' }
if ($EvaluateOnly) { $commandArgs += @('--evaluate-only', $EvaluateOnly) }
& $Python @commandArgs
exit $LASTEXITCODE
