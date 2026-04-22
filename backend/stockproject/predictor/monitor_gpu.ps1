param(
    [int]$IntervalSec = 2
)

$nvCandidates = @(
    "nvidia-smi",
    "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "C:\Windows\System32\nvidia-smi.exe"
)

$nv = $null
foreach ($c in $nvCandidates) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) {
        $nv = $cmd.Source
        break
    }
    if (Test-Path $c) {
        $nv = $c
        break
    }
}

if (-not $nv) {
    Write-Error "nvidia-smi not found in PATH or default locations."
    exit 1
}

Write-Host "Using nvidia-smi: $nv"
Write-Host "Press Ctrl+C to stop monitoring."

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "`n[$timestamp]"
    & $nv --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    Start-Sleep -Seconds $IntervalSec
}
