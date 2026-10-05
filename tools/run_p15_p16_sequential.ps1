# 单进程串行执行 P16(width) -> P15-detsrc -> P15-dettransplant
# 任何时刻只有一个 python 训练进程（BelowNormal 优先级），保证交互流畅。
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_p15_p16_sequential.ps1
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'

function Invoke-Phase($name, $blocks, $log) {
  Write-Output "[seq] phase $name ($blocks) start $(Get-Date -Format HH:mm:ss)"
  $p = Start-Process -FilePath $py -PassThru -NoNewWindow -ArgumentList @(
    '-u', 'experiments/batch.py', '--run', '--blocks', $blocks,
    '--workers', '1', '--shard', '0/1', '--state-id', 'seq'
  ) -RedirectStandardOutput (Join-Path $root $log) -RedirectStandardError (Join-Path $root "$log.err")
  try { $p.PriorityClass = 'BelowNormal' } catch { }
  $p.WaitForExit()
  Write-Output "[seq] phase $name exit=$($p.ExitCode) end $(Get-Date -Format HH:mm:ss)"
}

Invoke-Phase 'detsrc' 'P15-detsrc' 'results/logs/seq_detsrc.log'
Invoke-Phase 'transplant' 'P15-dettransplant' 'results/logs/seq_transplant.log'
Invoke-Phase 'width' 'P16-width' 'results/logs/seq_width.log'

$n16 = (Get-ChildItem (Join-Path $root 'results\runs') -Directory -Filter 'P16_*' |
        Where-Object { Test-Path (Join-Path $_.FullName 'summary.json') }).Count
$nsrc = (Get-ChildItem (Join-Path $root 'results\runs') -Directory -Filter 'P15_detsrc_*' |
         Where-Object { Test-Path (Join-Path $_.FullName 'summary.json') }).Count
$ntr = (Get-ChildItem (Join-Path $root 'results\runs') -Directory -Filter 'P15_detHead*' |
        Where-Object { Test-Path (Join-Path $_.FullName 'summary.json') }).Count
Write-Output "[seq] DONE width=$n16/12 detsrc=$nsrc/6 transplant=$ntr/6"
