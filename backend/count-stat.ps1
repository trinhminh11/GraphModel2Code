param(
    [string]$Root = "."
)

$Root = (Resolve-Path $Root).Path
$gitignorePath = Join-Path $Root ".gitignore"

$ignorePatterns = @()

if (Test-Path $gitignorePath) {
    $ignorePatterns = Get-Content $gitignorePath |
        ForEach-Object { $_.Trim() } |
        Where-Object {
            $_ -and
            -not $_.StartsWith("#")
        }
}

function Normalize-RelativePath {
    param([string]$FullPath)

    $relative = $FullPath.Substring($Root.Length).TrimStart('\')
    return $relative -replace '\\', '/'
}

function Matches-IgnorePattern {
    param(
        [string]$FullPath,
        [bool]$IsDirectory
    )

    $relativePath = Normalize-RelativePath $FullPath
    $name = Split-Path $FullPath -Leaf

    foreach ($pattern in $ignorePatterns) {
        $p = $pattern.Trim()

        if (-not $p) { continue }

        # Detect directory-only rule like "node_modules/"
        $dirOnly = $p.EndsWith("/")
        if ($dirOnly) {
            $p = $p.TrimEnd("/")
            if (-not $IsDirectory) { continue }
        }

        # Match exact or wildcard against name and relative path
        if ($name -like $p -or $relativePath -like $p -or $relativePath -like "$p/*") {
            return $true
        }

        # Also treat plain folder names as match anywhere in the path
        $parts = $relativePath -split '/'
        if ($parts -contains $p) {
            return $true
        }
    }

    return $false
}

$totalLines = 0
$totalChars = 0
$fileCount = 0
$skippedCount = 0

Get-ChildItem -Path $Root -Recurse -File -Force | ForEach-Object {
    $file = $_
    $relativePath = Normalize-RelativePath $file.FullName

    # Skip if file itself matches ignore
    if (Matches-IgnorePattern -FullPath $file.FullName -IsDirectory $false) {
        $skippedCount++
        return
    }

    # Skip if any parent directory matches ignore
    $parent = $file.DirectoryName
    $skip = $false

    while ($parent -and $parent.StartsWith($Root)) {
        if ($parent -eq $Root) { break }

        if (Matches-IgnorePattern -FullPath $parent -IsDirectory $true) {
            $skip = $true
            break
        }

        $parent = Split-Path $parent -Parent
    }

    if ($skip) {
        $skippedCount++
        return
    }

    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction Stop
        $chars = $content.Length

        $lines = 0
        if ($content.Length -gt 0) {
            $lines = ([regex]::Matches($content, "`n")).Count
            if (-not $content.EndsWith("`n")) {
                $lines += 1
            }
        }

        $totalChars += $chars
        $totalLines += $lines
        $fileCount += 1
    }
    catch {
        Write-Warning "Skipped unreadable file: $relativePath"
    }
}

Write-Host ""
Write-Host "========== TOTAL =========="
Write-Host "Files counted : $fileCount"
Write-Host "Files skipped : $skippedCount"
Write-Host "Lines         : $totalLines"
Write-Host "Characters    : $totalChars"
Write-Host "Ignore rules  : $($ignorePatterns -join ', ')"
