param(
  [Parameter(Mandatory=$true)][string]$AudioCpp,
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$Source
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path '.\results' | Out-Null
Write-Host 'Benchmarking SAM GGUF (note: every audio.cpp CLI call reloads the model; this is NOT steady-state persistent model speed).'
python -m barkfilter.benchmark --input $Source --output-json '.\results\sam-gguf-mono.json' --backend sam-gguf --cli $AudioCpp --model $Model --device cuda --mono --mode continuous --durations 4,8,16,32 --windows 4,8,12 --overlap 2 --repeats 2
python -m barkfilter.benchmark --input $Source --output-json '.\results\sam-gguf-stereo.json' --backend sam-gguf --cli $AudioCpp --model $Model --device cuda --mode continuous --durations 8,16,32 --windows 4,8,12 --overlap 2 --repeats 2
Write-Host 'Saved JSON benchmark records. For sustained warm performance, additionally benchmark the official persistent Python SAM adapter.'
