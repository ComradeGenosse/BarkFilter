param(
  [Parameter(Mandatory=$true)][string]$Source,
  [string]$Model = 'facebook/sam-audio-small'
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path '.\results' | Out-Null
Write-Host 'Requires approved gated Hugging Face access and the upstream official SAM Python environment.'
python -m barkfilter.benchmark --input $Source --output-json '.\results\sam-official-mono.json' --backend sam-official --model $Model --device cuda --mono --mode continuous --durations 4,8,16,32 --windows 4,8,12 --overlap 2 --repeats 3
