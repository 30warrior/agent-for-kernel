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
    [int]$ExtraTestCases = 8,
    [double]$TestSuiteMaxMiB = 512,
    [string]$TestSpec,
    [ValidateSet('1.2')][string]$TlsMaxVersion,
    [switch]$PrepareOnly,
    [string]$EvaluateOnly
)

$commandArgs = @('-B', "$PSScriptRoot/kernel_optimizer.py", '--dataset', 'akg',
    '--dataset-root', "$PSScriptRoot/benchmarks/akg", '--task', $Task,
    '--target', $Target, '--executor', 'ssh', '--ssh-host', $SshHost,
    '--remote-root', $RemoteRoot, '--remote-python', $RemotePython,
    '--model-api', $ModelApi, '--max-iterations', $MaxIterations,
    '--extra-test-cases', $ExtraTestCases, '--test-suite-max-mib', $TestSuiteMaxMiB)
if ($Model) { $commandArgs += @('--model', $Model) }
if ($PrepareOnly) { $commandArgs += '--prepare-only' }
if ($TestSpec) { $commandArgs += @('--test-spec', $TestSpec) }
if ($TlsMaxVersion) { $commandArgs += @('--tls-max-version', $TlsMaxVersion) }
if ($EvaluateOnly) { $commandArgs += @('--evaluate-only', $EvaluateOnly) }
& $Python @commandArgs
exit $LASTEXITCODE
