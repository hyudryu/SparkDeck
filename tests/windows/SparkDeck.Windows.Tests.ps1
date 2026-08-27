$modulePath = Join-Path $PSScriptRoot "..\..\scripts\windows\SparkDeck.Windows.psm1"
Import-Module $modulePath -Force

Describe "SparkDeck Windows launcher" {
    It "resolves the repository root independent of the current directory" {
        $paths = Get-SparkDeckPaths
        $paths.Root | Should Be ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..")))
        $paths.Server | Should Be (Join-Path $paths.Root "server.py")
    }

    It "adds a PATH entry once using case-insensitive comparison" {
        $actual = Add-SparkDeckPathEntry "C:\Windows;C:\Tools\SparkDeck\;c:\tools\sparkdeck" "C:\Tools\SparkDeck"
        $actual | Should Be "C:\Windows;C:\Tools\SparkDeck"
    }

    It "preserves unrelated PATH entries when removing SparkDeck" {
        $actual = Remove-SparkDeckPathEntry "C:\One;C:\SparkDeck\bin;C:\Two" "c:\sparkdeck\BIN\"
        $actual | Should Be "C:\One;C:\Two"
    }

    It "changes the frontend fingerprint when a source file changes" {
        $root = Join-Path $TestDrive "fingerprint repo"
        New-Item -ItemType Directory -Path (Join-Path $root "frontend\src") -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $root "frontend\package.json") -Value "{}"
        Set-Content -LiteralPath (Join-Path $root "frontend\src\app.ts") -Value "one"
        $paths = Get-SparkDeckPaths $root
        $first = Get-SparkDeckFrontendFingerprint $paths
        Set-Content -LiteralPath (Join-Path $root "frontend\src\app.ts") -Value "a longer value"
        $second = Get-SparkDeckFrontendFingerprint $paths
        $second | Should Not Be $first
    }

    It "ignores node_modules and dist when fingerprinting frontend sources" {
        $root = Join-Path $TestDrive "ignored frontend files"
        New-Item -ItemType Directory -Path (Join-Path $root "frontend\src") -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $root "frontend\node_modules\pkg") -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $root "frontend\dist") -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $root "frontend\package.json") -Value "{}"
        Set-Content -LiteralPath (Join-Path $root "frontend\src\app.ts") -Value "source"
        $paths = Get-SparkDeckPaths $root
        $first = Get-SparkDeckFrontendFingerprint $paths
        Set-Content -LiteralPath (Join-Path $root "frontend\node_modules\pkg\index.js") -Value "changed"
        Set-Content -LiteralPath (Join-Path $root "frontend\dist\index.html") -Value "changed"
        (Get-SparkDeckFrontendFingerprint $paths) | Should Be $first
    }

    It "accepts successful native commands that write progress to stderr" {
        { Invoke-SparkDeckCheckedCommand -FilePath "powershell.exe" -Arguments @(
            "-NoProfile", "-Command", "[Console]::Error.WriteLine('progress'); exit 0"
        ) } | Should Not Throw
    }

    It "prints help through the cmd shim from a different directory" {
        $root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
        Push-Location $env:TEMP
        try {
            $output = & cmd.exe /d /c ('"{0}" help' -f (Join-Path $root "sparkdeck.cmd")) 2>&1
            $LASTEXITCODE | Should Be 0
            ($output -join "`n") | Should Match "Usage: sparkdeck <command>"
        }
        finally { Pop-Location }
    }

    It "prints help when the checkout path contains spaces" {
        $sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
        $copyRoot = Join-Path $TestDrive "Spark Deck checkout"
        New-Item -ItemType Directory -Path (Join-Path $copyRoot "scripts\windows") -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $sourceRoot "sparkdeck.cmd") -Destination $copyRoot
        Copy-Item -LiteralPath (Join-Path $sourceRoot "scripts\windows\sparkdeck.ps1") -Destination (Join-Path $copyRoot "scripts\windows")
        Copy-Item -LiteralPath (Join-Path $sourceRoot "scripts\windows\SparkDeck.Windows.psm1") -Destination (Join-Path $copyRoot "scripts\windows")

        $output = & cmd.exe /d /c ('"{0}" help' -f (Join-Path $copyRoot "sparkdeck.cmd")) 2>&1
        $LASTEXITCODE | Should Be 0
        ($output -join "`n") | Should Match "Usage: sparkdeck <command>"
    }

    It "rejects a stale PID record without stopping that process" {
        $root = Join-Path $TestDrive "pid identity"
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        $paths = Get-SparkDeckPaths $root
        $current = Get-Process -Id $PID
        $record = [pscustomobject]@{
            pid = $PID
            started_at_utc = $current.StartTime.ToUniversalTime().ToString("o")
            executable = "C:\definitely-not-sparkdeck\python.exe"
            server = (Join-Path $root "server.py")
            repo_root = $root
        }
        (Test-SparkDeckProcessIdentity $paths $record) | Should Be $false
        (Get-Process -Id $PID -ErrorAction SilentlyContinue) | Should Not BeNullOrEmpty
    }

    It "returns an error for an unknown command" {
        $result = Invoke-SparkDeckCommand -Command "not-a-command"
        $result | Should Be 1
    }
}
