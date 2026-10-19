<#
Run unattended on a Windows NVIDIA GPU or GitHub self-hosted runner.
Prerequisites once: Python 3.11, installed package, CUDA audio.cpp binary,
SAM Audio Q8 GGUF weight, NVIDIA GPU driver, applicable SAM model license.
No automatic license acceptance, GitHub secrets, paid cloud, or model redistribution.
#>
param(
  [string]$AudioCpp = $env:BARKFILTER_AUDIOCPP,
  [string]$Model = $env:BARKFILTER_MODEL,
  [string]$OutputDir = 'results/gpu',
  [switch]$RequireRtx5090
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Write-FailureRecord([string]$Reason) {
  $failure = @{ status = 'blocked'; reason = $Reason; real_sam_benchmark_completed = $false;
                timestamp_utc = [DateTime]::UtcNow.ToString('o') }
  $failure | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $OutputDir 'preflight.json') -Encoding utf8
  throw $Reason
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Write-FailureRecord 'NVIDIA driver/nvidia-smi unavailable. No GPU measurement.'
}
$gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $gpu) { Write-FailureRecord 'NVIDIA driver/nvidia-smi unavailable. No GPU measurement.' }
if ($RequireRtx5090 -and $gpu -notmatch 'RTX 5090') {
  Write-FailureRecord "Expected RTX 5090; observed '$gpu'."
}
if (-not $AudioCpp -or -not (Test-Path -LiteralPath $AudioCpp -PathType Leaf)) {
  Write-FailureRecord 'Set BARKFILTER_AUDIOCPP to an installed CUDA audiocpp_cli.exe.'
}
if (-not $Model -or -not (Test-Path -LiteralPath $Model -PathType Leaf)) {
  Write-FailureRecord 'Set BARKFILTER_MODEL to an installed SAM Audio Small Q8 GGUF model.'
}
$command = Get-Command python -ErrorAction SilentlyContinue
if (-not $command) { Write-FailureRecord 'Python is not installed/on PATH.' }
python -c 'import barkfilter, scipy, soundfile, psutil'
if ($LASTEXITCODE -ne 0) { Write-FailureRecord 'Install the package first: python -m pip install -e .[test]' }

$preflight = @{ status = 'ready'; gpu = $gpu; audio_cpp = $AudioCpp;
                model_name = (Split-Path -Leaf $Model);
                model_size_bytes = (Get-Item -LiteralPath $Model).Length;
                timestamp_utc = [DateTime]::UtcNow.ToString('o') }
$preflight | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $OutputDir 'preflight.json') -Encoding utf8
Write-Host "Running real SAM GGUF on $gpu. Output: $OutputDir"

python -m barkfilter.autonomous --backend sam-gguf --cli $AudioCpp --model $Model `
  --device cuda --mode continuous --mono --window 4 --overlap 1 --require-gpu `
  --cases 'speech_overlap,music_overlap,NO_BARK,overlapping_words_HARD' `
  --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { Write-FailureRecord 'SAM end-to-end suite failed; inspect report.json.' }

$benchInput = Join-Path $OutputDir 'synthetic_inputs/speech_overlap_input.wav'
python -m barkfilter.benchmark --input $benchInput --backend sam-gguf `
  --cli $AudioCpp --model $Model --device cuda --mono --mode continuous `
  --durations 4,8,16 --windows 4,8 --overlap 1 --repeats 1 `
  --output-json (Join-Path $OutputDir 'real_sam_gguf_reloading_benchmark.json')
if ($LASTEXITCODE -ne 0) { Write-FailureRecord 'GPU throughput benchmark failed.' }
Write-Host 'Actual SAM GGUF results created. Each chunk reloads the model; NOT persistent warm throughput.'
