$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -u aawden.py serve --run runs/aawden-demo --port 8765

