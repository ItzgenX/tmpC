<#
Runs generate_scenarios.py in a loop, auto-resuming after any crash
(including the native core dumps that kill the Python process outright,
which generate_scenarios.py itself can never catch or retry around).
Every image is already persisted to disk as it's saved, so a crash loses
at most the one frame in flight - this script's only job is to keep
relaunching with --resume latest until the whole plan is actually done.

Usage (same args as a normal run, used only for the very first launch):
    .\run_until_done.ps1
    .\run_until_done.ps1 -Mode occlusion_only -Count 100
    .\run_until_done.ps1 -Weather clear -Count 50

After the first launch, every relaunch uses --resume latest automatically -
Mode/Count/Weather are ignored on restarts since --resume continues the
existing plan, which already has the split and scenario baked in.
#>

param(
    [string]$Mode = "",
    [string]$Count = "",
    [string]$Weather = "",
    [int]$MaxAttempts = 100,
    [int]$CooldownSeconds = 5,
    [int]$FastCrashSeconds = 8,
    [int]$MaxConsecutiveFastCrashes = 5
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$firstRunArgs = @()
if ($Mode -ne "") { $firstRunArgs += @("--mode", $Mode) }
if ($Count -ne "") { $firstRunArgs += @("--count", $Count) }
if ($Weather -ne "") { $firstRunArgs += @("--weather", $Weather) }

$attempt = 0
$consecutiveFastCrashes = 0

while ($attempt -lt $MaxAttempts) {
    $attempt++

    if ($attempt -eq 1) {
        $runArgs = $firstRunArgs
        Write-Host "`n=== attempt $attempt`: starting fresh run ===" -ForegroundColor Cyan
    } else {
        $runArgs = @("--resume", "latest")
        Write-Host "`n=== attempt $attempt`: resuming latest run ===" -ForegroundColor Cyan
    }

    $start = Get-Date
    & python generate_scenarios.py @runArgs
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    if ($exitCode -eq 0) {
        Write-Host "`n=== done: run completed successfully after $attempt attempt(s) ===" -ForegroundColor Green
        exit 0
    }

    Write-Host "`n--- attempt $attempt exited with code $exitCode after $([math]::Round($elapsed.TotalSeconds, 1))s ---" -ForegroundColor Yellow

    if ($elapsed.TotalSeconds -lt $FastCrashSeconds) {
        $consecutiveFastCrashes++
        if ($consecutiveFastCrashes -ge $MaxConsecutiveFastCrashes) {
            Write-Host "`n$consecutiveFastCrashes crashes in a row, each under ${FastCrashSeconds}s." -ForegroundColor Red
            Write-Host "This usually means the CARLA server itself needs a manual restart," -ForegroundColor Red
            Write-Host "not just the script. Restart CARLA, then re-run this script." -ForegroundColor Red
            exit 1
        }
    } else {
        $consecutiveFastCrashes = 0
    }

    Write-Host "retrying in ${CooldownSeconds}s (ctrl+C to stop)..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $CooldownSeconds
}

Write-Host "`n=== gave up after $MaxAttempts attempts without completing ===" -ForegroundColor Red
exit 1
