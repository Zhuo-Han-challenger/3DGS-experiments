param(
    [string]$RepoUrl = "https://github.com/Zhuo-Han-challenger/3DGS-experiments.git",
    [string]$Branch = "main",
    [string]$CommitMessage = "Add LOD PLY RD experiment pipeline"
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Please install it and retry."
    }
}

Require-Command git

$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectName = Split-Path -Leaf $SourceDir
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("3dgs-experiments-push-" + [guid]::NewGuid().ToString("N"))
$CloneDir = Join-Path $TempRoot "repo"

Write-Host "Source: $SourceDir"
Write-Host "Repo:   $RepoUrl"
Write-Host "Branch: $Branch"
Write-Host "Temp:   $CloneDir"

New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

try {
    git clone --branch $Branch $RepoUrl $CloneDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Branch '$Branch' was not cloned directly. Trying default clone, then creating '$Branch'."
        if (Test-Path $CloneDir) {
            Remove-Item -LiteralPath $CloneDir -Recurse -Force
        }
        git clone $RepoUrl $CloneDir
        if ($LASTEXITCODE -ne 0) {
            throw "git clone failed."
        }
        Push-Location $CloneDir
        try {
            git checkout -B $Branch
            if ($LASTEXITCODE -ne 0) {
                throw "git checkout -B $Branch failed."
            }
        }
        finally {
            Pop-Location
        }
    }

    $DestDir = Join-Path $CloneDir $ProjectName
    if (Test-Path $DestDir) {
        Remove-Item -LiteralPath $DestDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

    Get-ChildItem -LiteralPath $SourceDir -Force |
        Where-Object { $_.Name -notin @(".git", "__pycache__", ".pytest_cache") -and $_.Name -ne "push_to_github.ps1" } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $DestDir -Recurse -Force
        }

    Copy-Item -LiteralPath (Join-Path $SourceDir "push_to_github.ps1") -Destination $DestDir -Force

    Push-Location $CloneDir
    try {
        git status --short
        git add $ProjectName
        if ($LASTEXITCODE -ne 0) {
            throw "git add failed."
        }

        $Changes = git status --porcelain
        if (-not $Changes) {
            Write-Host "No changes to commit."
            exit 0
        }

        git commit -m $CommitMessage
        if ($LASTEXITCODE -ne 0) {
            throw "git commit failed. Check git user.name/user.email or authentication."
        }

        git push -u origin $Branch
        if ($LASTEXITCODE -ne 0) {
            throw "git push failed. Check GitHub authentication/permissions."
        }

        Write-Host ""
        Write-Host "Push completed:"
        Write-Host "https://github.com/Zhuo-Han-challenger/3DGS-experiments/tree/$Branch/$ProjectName"
    }
    finally {
        Pop-Location
    }
}
finally {
    Write-Host "Temporary clone kept at: $CloneDir"
}
