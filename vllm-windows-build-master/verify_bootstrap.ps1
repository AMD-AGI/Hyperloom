param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$ExpectedSha256,
    [Parameter(Mandatory = $true)][long]$ExpectedSize
)

$ErrorActionPreference = "Stop"

try {
    $item = Get-Item -LiteralPath $Path
    $stream = [System.IO.File]::OpenRead($item.FullName)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $actualSha256 = [System.BitConverter]::ToString(
            $sha256.ComputeHash($stream)
        ).Replace("-", "")
    }
    finally {
        $sha256.Clear()
        $stream.Dispose()
    }

    Write-Host ("Artifact: " + $item.FullName)
    Write-Host ("Size:     " + $item.Length + " bytes (expected " + $ExpectedSize + ")")
    Write-Host ("SHA256:   " + $actualSha256)

    if ($item.Length -ne $ExpectedSize) {
        throw "Artifact size does not match"
    }
    if ($actualSha256 -ne $ExpectedSha256.ToUpperInvariant()) {
        throw ("Expected SHA256 " + $ExpectedSha256.ToUpperInvariant())
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}

exit 0
