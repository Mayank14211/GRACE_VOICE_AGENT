<#
.SYNOPSIS
  Download the local LLM (hash-checked) and start llama-server for GRACE's router.

.DESCRIPTION
  1. Downloads models/llm/google_gemma-4-E4B-it-Q4_K_M.gguf (5.4 GB) if it is missing and checks
     its sha256 against grace/models.py (scripts/fetch_models.py does the download and the
     check; a wrong hash is never kept).
  2. Starts llama-server with --jinja (tool calls), reasoning off, -t <inference.threads> from
     settings.yaml (4 on the i5-12450H, 6 on the Ryzen 5 5500U), one slot (prompt cache reuse),
     on router.local.base_url (default http://127.0.0.1:8080/v1).

  Then set router.local.enabled: true in settings.yaml (or demo_mode: true) and start GRACE.
  Ctrl+C stops the server.

  llama-server itself: winget install ggml.llamacpp   (or unzip a llama.cpp Windows release
  into tools\llama.cpp\, or set router.local.binary in settings.yaml).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_llama_server.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_llama_server.ps1 -Threads 6
#>
param(
    [int]$Threads = 0,          # 0 = inference.threads from settings.yaml
    [switch]$NoDownload,        # skip the download / hash step
    [switch]$PrintOnly          # print the llama-server command and exit
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install it: https://docs.astral.sh/uv/"
    exit 1
}

if (-not $NoDownload) {
    Write-Host "Checking the model file (first run downloads 5.4 GB)..."
    uv run python scripts/fetch_models.py --llm --only-llm
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Model download or hash check failed (see above)."
        exit 1
    }
}

$helperArgs = @('run', 'python', '-m', 'grace.intent.local_llm', '--print-command')
if ($Threads -gt 0) { $helperArgs += @('--threads', "$Threads") }
$json = & uv @helperArgs
if ($LASTEXITCODE -ne 0) { exit 1 }   # the helper printed why (binary or model missing)

$command = @($json | ConvertFrom-Json)
$exe = $command[0]
$serverArgs = @()
if ($command.Count -gt 1) { $serverArgs = $command[1..($command.Count - 1)] }

Write-Host ("llama-server: " + ($command -join ' '))
if ($PrintOnly) { exit 0 }
& $exe @serverArgs
exit $LASTEXITCODE
