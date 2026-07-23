[CmdletBinding()]
param(
    [string]$DownloadsDir = (Join-Path $env:USERPROFILE "Downloads"),
    [string]$WorkspaceRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$runName = "Ref.No.747_VietnamNationalUniversity-Hanoi"

$resolvedDownloads = (Resolve-Path -LiteralPath $DownloadsDir).Path
$resolvedWorkspace = (Resolve-Path -LiteralPath $WorkspaceRoot).Path
$destinationRoot = Join-Path $resolvedWorkspace "data\work\chandra2"
$destination = Join-Path $destinationRoot $runName
$destinationFullPath = [System.IO.Path]::GetFullPath($destination)

if (-not $destinationFullPath.StartsWith(
        $resolvedWorkspace,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Refusing to extract outside workspace: $destinationFullPath"
}

$archive = Get-ChildItem -LiteralPath $resolvedDownloads -File |
    Where-Object {
        $_.Extension -eq ".zip" -and $_.BaseName -like "$runName*"
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $archive) {
    throw "Không tìm thấy $runName*.zip trong $resolvedDownloads. Hãy chạy cell Examine để Colab tải archive trước."
}

New-Item -ItemType Directory -Path $destinationFullPath -Force | Out-Null
Expand-Archive -LiteralPath $archive.FullName -DestinationPath $destinationFullPath -Force

$localArchive = Join-Path $destinationRoot "$runName.zip"
Copy-Item -LiteralPath $archive.FullName -Destination $localArchive -Force

Write-Host "Imported: $($archive.FullName)"
Write-Host "Output:   $destinationFullPath"
Get-ChildItem -LiteralPath $destinationFullPath -File -Recurse |
    Select-Object FullName, Length
