[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

& $Python scripts/verify_reference_data.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python -m pytest tests/test_protocol_v7.py -q -p no:cacheprovider
exit $LASTEXITCODE
