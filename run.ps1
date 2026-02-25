param(
    [switch]$Install
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Requirements = Join-Path $ProjectRoot 'requirements.txt'
$BotPath = Join-Path $ProjectRoot 'src\engine.py'

if (-not (Test-Path $VenvPython)) {
    Write-Host '[INFO] Tworze virtualenv (.venv)...'
    py -m venv (Join-Path $ProjectRoot '.venv')
}

if ($Install) {
    Write-Host '[INFO] Instaluje zaleznosci z requirements.txt...'
    & $VenvPython -m pip install -r $Requirements
}

Write-Host '[INFO] Uruchamiam bota (engine 2.0)...'
& $VenvPython $BotPath
