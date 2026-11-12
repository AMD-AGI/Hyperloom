param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$Destination
)

$ErrorActionPreference = "Stop"

try {
    if ($PSVersionTable.PSVersion.Major -lt 3) {
        throw "PowerShell 3 or newer is required for ZIP extraction"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $destinationFull = [System.IO.Path]::GetFullPath($Destination)
    $destinationPrefix = $destinationFull.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    [System.IO.Directory]::CreateDirectory($destinationFull) | Out-Null

    $zip = [System.IO.Compression.ZipFile]::OpenRead(
        [System.IO.Path]::GetFullPath($Archive)
    )
    try {
        foreach ($entry in $zip.Entries) {
            $target = [System.IO.Path]::GetFullPath(
                [System.IO.Path]::Combine($destinationFull, $entry.FullName)
            )
            if (-not $target.StartsWith(
                $destinationPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                throw ("ZIP entry escapes destination: " + $entry.FullName)
            }

            if ([string]::IsNullOrEmpty($entry.Name)) {
                [System.IO.Directory]::CreateDirectory($target) | Out-Null
                continue
            }

            $parent = [System.IO.Path]::GetDirectoryName($target)
            [System.IO.Directory]::CreateDirectory($parent) | Out-Null
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
                $entry,
                $target,
                $true
            )
        }
    }
    finally {
        $zip.Dispose()
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}

exit 0
