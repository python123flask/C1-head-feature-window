# 自动接力：等 P15-detsrc 的10 个 run 全部产出 summary.json，
# 再以6 分片启动 P15-dettransplant（移植 run 短但也是确定性核，必须并发跑）。
# 用法：pwsh -File tools/sequencer_wave2.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$deadline = (Get-Date).AddHours(8)

function Count-Detsrc {
  $n = 0
  Get-ChildItem (Join-Path $root 'results\runs') -Directory -Filter 'P15_detsrc_*' -ErrorAction SilentlyContinue |
    ForEach-Object { if (Test-Path (Join-Path $_.FullName 'summary.json')) { $n++ } }
  return $n
}
function Count-Transplant {
  $n = 0
  Get-ChildItem (Join-Path $root 'results\runs') -Directory -Filter 'P15_det*' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'P15_detHead*' } |
    ForEach-Object { if (Test-Path (Join-Path $_.FullName 'summary.json')) { $n++ } }
  return $n
}

Write-Output "[seq] waiting for P15-detsrc summaries (currently $((Count-Detsrc))/10)..."
while ((Count-Detsrc) -lt 10) {
  if ((Get-Date) -gt $deadline) { Write-Output "[seq] TIMEOUT waiting detsrc"; exit 1 }
  Start-Sleep -Seconds 30
}
Write-Output "[seq] detsrc complete -> launching P15-dettransplant (6 shards)"

$procs = @()
for ($k = 0; $k -lt 6; $k++) {
  $procs += Start-Process -FilePath $py -PassThru -NoNewWindow -ArgumentList @(
    '-u', 'experiments/batch.py', '--run', '--blocks', 'P15-dettransplant',
    '--workers', '1', '--shard', "$k/6", '--state-id', "t$k"
  ) -RedirectStandardOutput "results/logs/wave2_shard$k.log" -RedirectStandardError "results/logs/wave2_shard$k.err"
}
Write-Output "[seq] started $($procs.Count) shard processes; waiting for transplant summaries..."
while ((Count-Transplant) -lt 10) {
  if ((Get-Date) -gt $deadline) { Write-Output "[seq] TIMEOUT waiting transplant"; exit 1 }
  $alive = ($procs | Where-Object { -not $_.HasExited }).Count
  if ($alive -eq 0 -and (Count-Transplant) -lt 10) {
    Write-Output "[seq] all shard procs exited but only $((Count-Transplant))/10 done -> check wave2_shard*.err"
    exit 2
  }
  Start-Sleep -Seconds 30
}
Write-Output "[seq] DONE: transplant $((Count-Transplant))/10 complete"
