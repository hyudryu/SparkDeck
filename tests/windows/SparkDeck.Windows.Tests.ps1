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

    It "changes the frontend fingerprint when the embedded release version changes" {
        $root = Join-Path $TestDrive "version fingerprint repo"
        New-Item -ItemType Directory -Path (Join-Path $root "frontend\src") -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $root "frontend\package.json") -Value "{}"
        Set-Content -LiteralPath (Join-Path $root "frontend\src\app.ts") -Value "source"
        $paths = Get-SparkDeckPaths $root
        $previous = $env:SPARKDECK_VERSION
        try {
            $env:SPARKDECK_VERSION = "v1.2.3"
            $first = Get-SparkDeckFrontendFingerprint $paths
            $env:SPARKDECK_VERSION = "v1.2.4"
            (Get-SparkDeckFrontendFingerprint $paths) | Should Not Be $first
        }
        finally {
            $env:SPARKDECK_VERSION = $previous
        }
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

    InModuleScope SparkDeck.Windows {
        It "includes the current git revision in the frontend fingerprint" {
            $root = Join-Path $TestDrive "git fingerprint repo"
            New-Item -ItemType Directory -Path (Join-Path $root "frontend\src") -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $root "frontend\package.json") -Value "{}"
            Set-Content -LiteralPath (Join-Path $root "frontend\src\app.ts") -Value "source"
            $paths = Get-SparkDeckPaths $root
            $script:testGitRevision = "1111111111111111111111111111111111111111"

            Mock Get-Command { [pscustomobject]@{ Source = "C:\Git\git.exe" } } -ParameterFilter { $Name -eq "git.exe" }
            Mock Invoke-SparkDeckNativeCapture {
                if ($Arguments -contains "rev-parse") {
                    return [pscustomobject]@{ ExitCode = 0; Stdout = $script:testGitRevision; Stderr = "" }
                }
                return [pscustomobject]@{ ExitCode = 1; Stdout = ""; Stderr = "" }
            } -ParameterFilter { $FilePath -eq "C:\Git\git.exe" }

            $first = Get-SparkDeckFrontendFingerprint $paths
            $script:testGitRevision = "2222222222222222222222222222222222222222"
            (Get-SparkDeckFrontendFingerprint $paths) | Should Not Be $first
            Assert-MockCalled Invoke-SparkDeckNativeCapture -Times 6 -Exactly -ParameterFilter {
                $FilePath -eq "C:\Git\git.exe"
            }
        }

        It "accepts the default Python 3 launcher runtime when it is 3.11 or newer" {
            Mock Get-Command { [pscustomobject]@{ Source = "C:\Windows\py.exe" } } -ParameterFilter { $Name -eq "py.exe" }
            Mock Invoke-SparkDeckNativeCapture {
                [pscustomobject]@{ ExitCode = 0; Stdout = ""; Stderr = "" }
            } -ParameterFilter {
                $Arguments[0] -eq "-3" -and
                $Arguments[1] -eq "-c" -and
                $Arguments[2] -match "sys.version_info >= \(3, 11\)"
            }

            $result = Get-SparkDeckBootstrapPython

            $result.FilePath | Should Be "C:\Windows\py.exe"
            $result.PrefixArguments.Count | Should Be 1
            $result.PrefixArguments[0] | Should Be "-3"
            Assert-MockCalled Invoke-SparkDeckNativeCapture -Times 1 -Exactly -ParameterFilter {
                $Arguments[0] -eq "-3" -and
                $Arguments[1] -eq "-c"
            }
        }

        It "checks process liveness through the lightweight local endpoint" {
            Mock Invoke-WebRequest { [pscustomobject]@{ StatusCode = 204 } } -ParameterFilter {
                $Uri -eq "http://127.0.0.1:7878/healthz"
            }

            (Test-SparkDeckHealth) | Should Be $true
            Assert-MockCalled Invoke-WebRequest -Times 1 -Exactly -ParameterFilter {
                $Uri -eq "http://127.0.0.1:7878/healthz"
            }
        }

        It "replaces an unsupported virtual environment while preserving application data" {
            $root = Join-Path $TestDrive "old python repo"
            $oldScripts = Join-Path $root ".venv\Scripts"
            $data = Join-Path $root "data"
            New-Item -ItemType Directory -Path $oldScripts -Force | Out-Null
            New-Item -ItemType Directory -Path $data -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $oldScripts "python.exe") -Value "old"
            Set-Content -LiteralPath (Join-Path $root ".venv\old-marker.txt") -Value "old environment"
            Set-Content -LiteralPath (Join-Path $data "keep.txt") -Value "persistent data"
            $paths = Get-SparkDeckPaths $root
            $script:versionProbe = 0

            Mock Invoke-SparkDeckNativeCapture {
                $script:versionProbe++
                [pscustomobject]@{
                    ExitCode = if ($script:versionProbe -eq 1) { 1 } else { 0 }
                    Stdout = ""
                    Stderr = ""
                }
            } -ParameterFilter {
                $Arguments.Count -ge 2 -and $Arguments[0] -eq "-c"
            }
            Mock Get-SparkDeckBootstrapPython {
                [pscustomobject]@{ FilePath = "C:\Python312\python.exe"; PrefixArguments = @() }
            }
            Mock Invoke-SparkDeckCheckedCommand {
                New-Item -ItemType Directory -Path (Split-Path $paths.VenvPython -Parent) -Force | Out-Null
                Set-Content -LiteralPath $paths.VenvPython -Value "new"
                Set-Content -LiteralPath (Join-Path $paths.Venv "new-marker.txt") -Value "new environment"
            }

            Initialize-SparkDeckPythonEnvironment $paths

            (Test-Path -LiteralPath (Join-Path $paths.Venv "old-marker.txt")) | Should Be $false
            (Test-Path -LiteralPath (Join-Path $paths.Venv "new-marker.txt")) | Should Be $true
            (Get-Content -LiteralPath (Join-Path $data "keep.txt") -Raw).Trim() | Should Be "persistent data"
            @(Get-ChildItem -LiteralPath $root -Directory -Filter ".venv.replaced-*").Count | Should Be 0
            Assert-MockCalled Invoke-SparkDeckCheckedCommand -Times 1 -Exactly
        }

        It "leaves an unsupported virtual environment unchanged when no supported Python is installed" {
            $root = Join-Path $TestDrive "missing system python"
            New-Item -ItemType Directory -Path (Join-Path $root ".venv\Scripts") -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $root ".venv\Scripts\python.exe") -Value "old"
            Set-Content -LiteralPath (Join-Path $root ".venv\keep.txt") -Value "keep"
            $paths = Get-SparkDeckPaths $root

            Mock Test-SparkDeckPythonSupported { $false }
            Mock Get-SparkDeckBootstrapPython { throw "Python 3.11 or newer was not found" }

            { Initialize-SparkDeckPythonEnvironment $paths } | Should Throw "left unchanged"
            (Get-Content -LiteralPath (Join-Path $paths.Venv "keep.txt") -Raw).Trim() | Should Be "keep"
        }

        It "restores an unsupported virtual environment when replacement fails" {
            $root = Join-Path $TestDrive "failed venv replacement"
            New-Item -ItemType Directory -Path (Join-Path $root ".venv\Scripts") -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $root ".venv\Scripts\python.exe") -Value "old"
            Set-Content -LiteralPath (Join-Path $root ".venv\keep.txt") -Value "keep"
            $paths = Get-SparkDeckPaths $root

            Mock Test-SparkDeckPythonSupported { $false }
            Mock Get-SparkDeckBootstrapPython {
                [pscustomobject]@{ FilePath = "C:\Python312\python.exe"; PrefixArguments = @() }
            }
            Mock Invoke-SparkDeckCheckedCommand {
                New-Item -ItemType Directory -Path $paths.Venv -Force | Out-Null
                throw "venv creation failed"
            }

            { Initialize-SparkDeckPythonEnvironment $paths } | Should Throw "original environment was restored"
            (Get-Content -LiteralPath (Join-Path $paths.Venv "keep.txt") -Raw).Trim() | Should Be "keep"
            @(Get-ChildItem -LiteralPath $root -Directory -Filter ".venv.replaced-*").Count | Should Be 0
        }

        It "terminates and waits for the exact new process when the PID record cannot be written" {
            $root = Join-Path $TestDrive "pid write failure"
            $paths = Get-SparkDeckPaths $root
            $newProcess = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                "-NoProfile", "-Command", "Start-Sleep -Seconds 60"
            ) -WindowStyle Hidden -PassThru
            try {
                $script:pidFailurePaths = $paths
                Mock Get-SparkDeckPaths { $script:pidFailurePaths } -ParameterFilter {
                    [string]::IsNullOrEmpty($Root)
                }
                Mock Invoke-WithSparkDeckLock { & $ScriptBlock }
                Mock Get-SparkDeckPidRecord { $null }
                Mock Test-SparkDeckProcessIdentity { $false }
                Mock Test-SparkDeckPortOpen { $false }
                Mock Initialize-SparkDeckEnvironment {}
                Mock Write-SparkDeckDockerWarning {}
                Mock Rotate-SparkDeckLogs {}
                $script:pidFailureProcess = $newProcess
                Mock Start-Process { $script:pidFailureProcess } -ParameterFilter {
                    $FilePath -match '\\.venv\\Scripts\\python\.exe$'
                }
                Mock Write-SparkDeckPidRecord { throw "pid record failed" }

                { Start-SparkDeck } | Should Throw "pid record failed"
                $newProcess.Refresh()
                $newProcess.HasExited | Should Be $true
                Assert-MockCalled Start-Process -Times 1 -Exactly
                Assert-MockCalled Write-SparkDeckPidRecord -Times 1 -Exactly
            }
            finally {
                $newProcess.Refresh()
                if (-not $newProcess.HasExited) {
                    $newProcess.Kill()
                    $newProcess.WaitForExit()
                }
            }
        }

        It "terminates both a validated root process and its child process" {
            $childPidPath = Join-Path $TestDrive "child.pid"
            $escapedChildPidPath = $childPidPath.Replace("'", "''")
            $parentCommand = @"
`$child = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-Command", "Start-Sleep -Seconds 60") -WindowStyle Hidden -PassThru
[IO.File]::WriteAllText('$escapedChildPidPath', [string]`$child.Id)
Wait-Process -Id `$child.Id
"@
            $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($parentCommand))
            $parent = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                "-NoProfile", "-EncodedCommand", $encoded
            ) -WindowStyle Hidden -PassThru
            $childPid = $null
            try {
                $deadline = [DateTime]::UtcNow.AddSeconds(10)
                while (-not (Test-Path -LiteralPath $childPidPath) -and [DateTime]::UtcNow -lt $deadline) {
                    Start-Sleep -Milliseconds 100
                }
                (Test-Path -LiteralPath $childPidPath) | Should Be $true
                $childPid = [int](Get-Content -LiteralPath $childPidPath -Raw)
                (Get-Process -Id $childPid -ErrorAction SilentlyContinue) | Should Not BeNullOrEmpty

                Stop-SparkDeckProcessTree -Process $parent -ExpectedStartTimeUtc $parent.StartTime.ToUniversalTime()

                (Get-Process -Id $parent.Id -ErrorAction SilentlyContinue) | Should BeNullOrEmpty
                (Get-Process -Id $childPid -ErrorAction SilentlyContinue) | Should BeNullOrEmpty
            }
            finally {
                if ($null -ne $childPid) {
                    Stop-Process -Id $childPid -Force -ErrorAction SilentlyContinue
                }
                Stop-Process -Id $parent.Id -Force -ErrorAction SilentlyContinue
            }
        }

        It "refuses to terminate a process whose start time does not match" {
            $process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                "-NoProfile", "-Command", "Start-Sleep -Seconds 60"
            ) -WindowStyle Hidden -PassThru
            try {
                $wrongStart = $process.StartTime.ToUniversalTime().AddMinutes(-1)
                { Stop-SparkDeckProcessTree -Process $process -ExpectedStartTimeUtc $wrongStart } |
                    Should Throw "start time no longer matches"
                $process.Refresh()
                $process.HasExited | Should Be $false
            }
            finally {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }

        It "uses validated process-tree termination after a startup timeout" {
            $root = Join-Path $TestDrive "startup timeout repo"
            $paths = Get-SparkDeckPaths $root
            $process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                "-NoProfile", "-Command", "Start-Sleep -Seconds 60"
            ) -WindowStyle Hidden -PassThru
            $record = [pscustomobject]@{ pid = $process.Id }
            $script:pidReadCount = 0
            $script:startupTimeoutPaths = $paths
            $script:startupTimeoutProcess = $process
            $script:startupTimeoutRecord = $record
            $previousTimeout = $env:SPARKDECK_START_TIMEOUT_SECONDS
            try {
                $env:SPARKDECK_START_TIMEOUT_SECONDS = "1"
                Mock Get-SparkDeckPaths { $script:startupTimeoutPaths } -ParameterFilter {
                    [string]::IsNullOrEmpty($Root)
                }
                Mock Invoke-WithSparkDeckLock { & $ScriptBlock }
                Mock Get-SparkDeckPidRecord {
                    $script:pidReadCount++
                    if ($script:pidReadCount -eq 1) { return $null }
                    return $script:startupTimeoutRecord
                }
                Mock Test-SparkDeckProcessIdentity { $false }
                Mock Test-SparkDeckPortOpen { $false }
                Mock Initialize-SparkDeckEnvironment {}
                Mock Write-SparkDeckDockerWarning {}
                Mock Rotate-SparkDeckLogs {}
                Mock Start-Process { $script:startupTimeoutProcess } -ParameterFilter {
                    $FilePath -match '\\.venv\\Scripts\\python\.exe$'
                }
                Mock Write-SparkDeckPidRecord {}
                Mock Test-SparkDeckHealth { $false }
                Mock Start-Sleep {}
                Mock Stop-SparkDeckValidatedProcessTree { $true }
                Mock Remove-SparkDeckPidRecord {}

                { Start-SparkDeck } | Should Throw "did not become healthy"
                Assert-MockCalled Stop-SparkDeckValidatedProcessTree -Times 1 -Exactly -ParameterFilter {
                    $Paths.Root -eq $script:startupTimeoutPaths.Root -and
                    $Record.pid -eq $script:startupTimeoutProcess.Id
                }
            }
            finally {
                $env:SPARKDECK_START_TIMEOUT_SECONDS = $previousTimeout
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }

        It "uses validated process-tree termination after graceful stop times out" {
            $root = Join-Path $TestDrive "graceful timeout repo"
            $paths = Get-SparkDeckPaths $root
            New-Item -ItemType Directory -Path $paths.RuntimeDirectory -Force | Out-Null
            $record = [pscustomobject]@{ pid = 43210 }
            $script:gracefulTimeoutPaths = $paths
            $script:gracefulTimeoutRecord = $record

            Mock Get-SparkDeckPaths { $script:gracefulTimeoutPaths } -ParameterFilter {
                [string]::IsNullOrEmpty($Root)
            }
            Mock Invoke-WithSparkDeckLock { & $ScriptBlock }
            Mock Get-SparkDeckPidRecord { $script:gracefulTimeoutRecord }
            Mock Test-SparkDeckProcessIdentity { $true }
            Mock Wait-Process {}
            Mock Stop-SparkDeckValidatedProcessTree { $true }
            Mock Remove-SparkDeckPidRecord {}

            (Stop-SparkDeck -Quiet) | Should Be 0
            Assert-MockCalled Stop-SparkDeckValidatedProcessTree -Times 1 -Exactly -ParameterFilter {
                $Paths.Root -eq $script:gracefulTimeoutPaths.Root -and $Record.pid -eq 43210
            }
        }
    }
}
