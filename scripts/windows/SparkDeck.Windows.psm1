Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:SparkDeckRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$script:SparkDeckPort = 7878

function Get-SparkDeckPaths {
    param([string]$Root = $script:SparkDeckRoot)

    $resolvedRoot = [IO.Path]::GetFullPath($Root)
    $runtimeDirectory = Join-Path $resolvedRoot "data\run"
    $logDirectory = Join-Path $resolvedRoot "data\logs"
    [pscustomobject]@{
        Root = $resolvedRoot
        Server = Join-Path $resolvedRoot "server.py"
        Venv = Join-Path $resolvedRoot ".venv"
        VenvPython = Join-Path $resolvedRoot ".venv\Scripts\python.exe"
        Requirements = Join-Path $resolvedRoot "requirements.txt"
        RequirementsStamp = Join-Path $resolvedRoot ".venv\.sparkdeck-requirements.sha256"
        Frontend = Join-Path $resolvedRoot "frontend"
        FrontendIndex = Join-Path $resolvedRoot "frontend\dist\index.html"
        FrontendStamp = Join-Path $resolvedRoot "frontend\dist\.sparkdeck-source.stamp"
        RuntimeDirectory = $runtimeDirectory
        PidFile = Join-Path $runtimeDirectory "sparkdeck.pid.json"
        ShutdownFile = Join-Path $runtimeDirectory "shutdown.request"
        LockFile = Join-Path $runtimeDirectory "launcher.lock"
        LogDirectory = $logDirectory
        StdoutLog = Join-Path $logDirectory "sparkdeck.stdout.log"
        StderrLog = Join-Path $logDirectory "sparkdeck.stderr.log"
    }
}

function Initialize-SparkDeckDirectories {
    param($Paths)
    [IO.Directory]::CreateDirectory($Paths.RuntimeDirectory) | Out-Null
    [IO.Directory]::CreateDirectory($Paths.LogDirectory) | Out-Null
}

function Invoke-WithSparkDeckLock {
    param(
        $Paths,
        [scriptblock]$ScriptBlock,
        [int]$TimeoutSeconds = 10
    )

    Initialize-SparkDeckDirectories $Paths
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $stream = $null
    while ($null -eq $stream) {
        try {
            $stream = [IO.File]::Open(
                $Paths.LockFile,
                [IO.FileMode]::OpenOrCreate,
                [IO.FileAccess]::ReadWrite,
                [IO.FileShare]::None
            )
        }
        catch [IO.IOException] {
            if ([DateTime]::UtcNow -ge $deadline) {
                throw "another SparkDeck launcher command is still running"
            }
            Start-Sleep -Milliseconds 200
        }
    }

    try {
        & $ScriptBlock
    }
    finally {
        $stream.Dispose()
    }
}

function Get-SparkDeckPidRecord {
    param($Paths)
    if (-not [IO.File]::Exists($Paths.PidFile)) {
        return $null
    }
    try {
        return (Get-Content -LiteralPath $Paths.PidFile -Raw | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Write-SparkDeckPidRecord {
    param($Paths, [Diagnostics.Process]$Process)

    $record = [ordered]@{
        pid = $Process.Id
        started_at_utc = $Process.StartTime.ToUniversalTime().ToString("o")
        executable = [IO.Path]::GetFullPath($Paths.VenvPython)
        server = [IO.Path]::GetFullPath($Paths.Server)
        repo_root = [IO.Path]::GetFullPath($Paths.Root)
        stdout_log = [IO.Path]::GetFullPath($Paths.StdoutLog)
        stderr_log = [IO.Path]::GetFullPath($Paths.StderrLog)
    }
    $temporary = $Paths.PidFile + "." + [Guid]::NewGuid().ToString("N") + ".tmp"
    $json = $record | ConvertTo-Json
    [IO.File]::WriteAllText($temporary, $json, (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Paths.PidFile -Force
}

function Remove-SparkDeckPidRecord {
    param($Paths)
    if ([IO.File]::Exists($Paths.PidFile)) {
        Remove-Item -LiteralPath $Paths.PidFile -Force
    }
}

function Test-SparkDeckProcessIdentity {
    param($Paths, $Record)

    if ($null -eq $Record -or $null -eq $Record.pid) {
        return $false
    }
    try {
        $process = Get-Process -Id ([int]$Record.pid) -ErrorAction Stop
        $processPath = $process.Path
        if ([string]::IsNullOrWhiteSpace($processPath)) {
            $cim = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f [int]$Record.pid) -ErrorAction Stop
            $processPath = $cim.ExecutablePath
        }
        if (-not [string]::Equals(
                [IO.Path]::GetFullPath($processPath),
                [IO.Path]::GetFullPath([string]$Record.executable),
                [StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        if (-not [string]::Equals(
                [IO.Path]::GetFullPath([string]$Record.repo_root),
                [IO.Path]::GetFullPath($Paths.Root),
                [StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }

        $expectedStart = [DateTime]::Parse([string]$Record.started_at_utc).ToUniversalTime()
        if ([Math]::Abs(($process.StartTime.ToUniversalTime() - $expectedStart).TotalSeconds) -gt 5) {
            return $false
        }

        $cimProcess = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f [int]$Record.pid) -ErrorAction Stop
        $commandLine = [string]$cimProcess.CommandLine
        $serverPath = [IO.Path]::GetFullPath([string]$Record.server)
        return $commandLine.IndexOf($serverPath, [StringComparison]::OrdinalIgnoreCase) -ge 0
    }
    catch {
        return $false
    }
}

function Test-SparkDeckHealth {
    try {
        # This local-only liveness route never waits for Docker or remote nodes.
        $response = Invoke-WebRequest -Uri ("http://127.0.0.1:{0}/healthz" -f $script:SparkDeckPort) -UseBasicParsing -TimeoutSec 2
        return ([int]$response.StatusCode -eq 204)
    }
    catch {
        return $false
    }
}

function Test-SparkDeckPortOpen {
    param([int]$Port = $script:SparkDeckPort)
    $client = New-Object Net.Sockets.TcpClient
    try {
        $operation = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $operation.AsyncWaitHandle.WaitOne(400)) {
            return $false
        }
        $client.EndConnect($operation)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Invoke-SparkDeckNativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutSeconds = 5
    )

    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    # ProcessStartInfo.ArgumentList is unavailable in Windows PowerShell 5.1.
    $escaped = foreach ($argument in $Arguments) {
        if ($argument -notmatch '[\s"]') {
            $argument
        }
        else {
            '"' + ($argument -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
        }
    }
    $startInfo.Arguments = ($escaped -join " ")
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "could not start $FilePath"
    }
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill()
        $process.WaitForExit()
        return [pscustomobject]@{ ExitCode = 124; Stdout = ""; Stderr = "command timed out" }
    }
    [pscustomobject]@{
        ExitCode = $process.ExitCode
        Stdout = $process.StandardOutput.ReadToEnd()
        Stderr = $process.StandardError.ReadToEnd()
    }
}

function Get-SparkDeckBootstrapPython {
    $versionCheck = "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        foreach ($version in @("-3", "-3.12", "-3.11")) {
            $probe = Invoke-SparkDeckNativeCapture -FilePath $launcher.Source -Arguments @($version, "-c", $versionCheck)
            if ($probe.ExitCode -eq 0) {
                return [pscustomobject]@{ FilePath = $launcher.Source; PrefixArguments = @($version) }
            }
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $probe = Invoke-SparkDeckNativeCapture -FilePath $python.Source -Arguments @("-c", $versionCheck)
        if ($probe.ExitCode -eq 0) {
            return [pscustomobject]@{ FilePath = $python.Source; PrefixArguments = @() }
        }
    }
    throw "Python 3.11 or newer was not found. Install Python, including the py launcher, and try again."
}

function Test-SparkDeckPythonSupported {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    if (-not [IO.File]::Exists($PythonPath)) {
        return $false
    }
    $versionCheck = "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    try {
        $probe = Invoke-SparkDeckNativeCapture -FilePath $PythonPath -Arguments @("-c", $versionCheck)
        return ($probe.ExitCode -eq 0)
    }
    catch {
        return $false
    }
}

function Remove-SparkDeckManagedVenvDirectory {
    param($Paths, [Parameter(Mandatory = $true)][string]$Directory)

    $candidate = [IO.Path]::GetFullPath($Directory).TrimEnd('\', '/')
    $expectedVenv = [IO.Path]::GetFullPath($Paths.Venv).TrimEnd('\', '/')
    $candidateParent = [IO.Path]::GetFullPath([IO.Path]::GetDirectoryName($candidate)).TrimEnd('\', '/')
    $expectedParent = [IO.Path]::GetFullPath($Paths.Root).TrimEnd('\', '/')
    $candidateName = [IO.Path]::GetFileName($candidate)
    $isManagedBackup = (
        [string]::Equals($candidateParent, $expectedParent, [StringComparison]::OrdinalIgnoreCase) -and
        $candidateName.StartsWith(".venv.replaced-", [StringComparison]::OrdinalIgnoreCase)
    )
    if (-not [string]::Equals($candidate, $expectedVenv, [StringComparison]::OrdinalIgnoreCase) -and -not $isManagedBackup) {
        throw ("refusing to remove an unmanaged virtual environment directory: {0}" -f $candidate)
    }
    if ([IO.Directory]::Exists($candidate)) {
        [IO.Directory]::Delete($candidate, $true)
    }
}

function Initialize-SparkDeckPythonEnvironment {
    param($Paths)

    if (Test-SparkDeckPythonSupported -PythonPath $Paths.VenvPython) {
        return
    }

    $existingVenv = [IO.Directory]::Exists($Paths.Venv)
    try {
        # Resolve a supported system interpreter before moving the existing
        # environment, so a missing Python installation leaves it untouched.
        $bootstrap = Get-SparkDeckBootstrapPython
    }
    catch {
        if ($existingVenv) {
            throw ("SparkDeck's existing .venv does not use Python 3.11 or newer, and a supported system Python was not found. Install Python 3.11 or newer and try again. The existing .venv was left unchanged. {0}" -f $_.Exception.Message)
        }
        throw
    }

    $backup = $null
    if ($existingVenv) {
        $backup = Join-Path $Paths.Root (".venv.replaced-{0}" -f [Guid]::NewGuid().ToString("N"))
        Write-Host "Replacing SparkDeck's unsupported Python environment..."
        Move-Item -LiteralPath $Paths.Venv -Destination $backup
    }
    else {
        Write-Host "Creating SparkDeck Python environment..."
    }

    try {
        $arguments = @($bootstrap.PrefixArguments) + @("-m", "venv", $Paths.Venv)
        Invoke-SparkDeckCheckedCommand -FilePath $bootstrap.FilePath -Arguments $arguments -WorkingDirectory $Paths.Root
        if (-not (Test-SparkDeckPythonSupported -PythonPath $Paths.VenvPython)) {
            throw "the newly created virtual environment does not provide Python 3.11 or newer"
        }
    }
    catch {
        if ([IO.Directory]::Exists($Paths.Venv)) {
            Remove-SparkDeckManagedVenvDirectory -Paths $Paths -Directory $Paths.Venv
        }
        if ($null -ne $backup -and [IO.Directory]::Exists($backup)) {
            Move-Item -LiteralPath $backup -Destination $Paths.Venv
            throw ("Could not replace SparkDeck's unsupported .venv; the original environment was restored. {0}" -f $_.Exception.Message)
        }
        throw
    }

    if ($null -ne $backup) {
        Remove-SparkDeckManagedVenvDirectory -Paths $Paths -Directory $backup
    }
}

function Invoke-SparkDeckCheckedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = ""
    )
    # Windows PowerShell turns a native program's stderr records into
    # terminating NativeCommandError exceptions when ErrorActionPreference is
    # Stop, even when the program exits successfully (pip/npm both do this).
    $previousPreference = $ErrorActionPreference
    $nativeExitCode = 0
    try {
        $ErrorActionPreference = "Continue"
        if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
            & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        }
        else {
            Push-Location $WorkingDirectory
            try { & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ } }
            finally { Pop-Location }
        }
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($nativeExitCode -ne 0) {
        throw ("{0} exited with code {1}" -f $FilePath, $nativeExitCode)
    }
}

function Get-SparkDeckFrontendFingerprint {
    param($Paths)
    $files = New-Object Collections.Generic.List[IO.FileInfo]
    foreach ($name in @("package.json", "package-lock.json", "index.html", "vite.config.ts")) {
        $candidate = Join-Path $Paths.Frontend $name
        if ([IO.File]::Exists($candidate)) { $files.Add((Get-Item -LiteralPath $candidate)) }
    }
    Get-ChildItem -LiteralPath $Paths.Frontend -Filter "tsconfig*.json" -File -ErrorAction SilentlyContinue |
        ForEach-Object { $files.Add($_) }
    $sourceDirectory = Join-Path $Paths.Frontend "src"
    if ([IO.Directory]::Exists($sourceDirectory)) {
        Get-ChildItem -LiteralPath $sourceDirectory -File -Recurse | ForEach-Object { $files.Add($_) }
    }
    $lines = @($files | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($Paths.Frontend.Length).TrimStart('\', '/')
        $contentHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "{0}|{1}" -f $relative, $contentHash
    })
    # vite.config.ts embeds buildVersion() into the generated JavaScript. Keep
    # the launcher cache key aligned with every input that function reads so a
    # new backend revision (or an explicit release version) cannot keep serving
    # an older version string from frontend/dist.
    foreach ($name in @("SPARKDECK_VERSION", "GITHUB_REF_TYPE", "GITHUB_REF_NAME", "GITHUB_SHA")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        $lines += "environment:{0}|{1}" -f $name, [string]$value
    }
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -ne $git) {
        foreach ($probe in @(
            [pscustomobject]@{ Name = "revision"; Arguments = @("-C", $Paths.Root, "rev-parse", "HEAD") },
            [pscustomobject]@{ Name = "status"; Arguments = @("-C", $Paths.Root, "status", "--porcelain", "--untracked-files=no") },
            [pscustomobject]@{ Name = "exact-tag"; Arguments = @("-C", $Paths.Root, "describe", "--tags", "--exact-match", "HEAD") }
        )) {
            try {
                $result = Invoke-SparkDeckNativeCapture -FilePath $git.Source -Arguments $probe.Arguments
                $value = if ($result.ExitCode -eq 0) { $result.Stdout.Trim() } else { "" }
                $lines += "git:{0}|{1}" -f $probe.Name, $value
            }
            catch {
                $lines += "git:{0}|" -f $probe.Name
            }
        }
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Initialize-SparkDeckEnvironment {
    param($Paths)

    if (-not [IO.File]::Exists($Paths.Server)) {
        throw ("server.py was not found under {0}" -f $Paths.Root)
    }
    Initialize-SparkDeckPythonEnvironment $Paths

    $requirementsHash = (Get-FileHash -LiteralPath $Paths.Requirements -Algorithm SHA256).Hash.ToLowerInvariant()
    $installedHash = if ([IO.File]::Exists($Paths.RequirementsStamp)) {
        [IO.File]::ReadAllText($Paths.RequirementsStamp).Trim()
    } else { "" }
    if ($requirementsHash -ne $installedHash) {
        Write-Host "Installing SparkDeck Python dependencies..."
        Invoke-SparkDeckCheckedCommand -FilePath $Paths.VenvPython -Arguments @("-m", "pip", "install", "-r", $Paths.Requirements) -WorkingDirectory $Paths.Root
        [IO.File]::WriteAllText($Paths.RequirementsStamp, $requirementsHash)
    }

    $frontendFingerprint = Get-SparkDeckFrontendFingerprint $Paths
    $builtFingerprint = if ([IO.File]::Exists($Paths.FrontendStamp)) {
        [IO.File]::ReadAllText($Paths.FrontendStamp).Trim()
    } else { "" }
    if (-not [IO.File]::Exists($Paths.FrontendIndex) -or $frontendFingerprint -ne $builtFingerprint) {
        $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if ($null -eq $npm) {
            throw "SparkDeck's web app needs Node.js and npm for its first build. Install Node.js 20 or newer and try again."
        }
        Write-Host "Building the SparkDeck web app..."
        Invoke-SparkDeckCheckedCommand -FilePath $npm.Source -Arguments @("ci", "--no-audit", "--no-fund") -WorkingDirectory $Paths.Frontend
        Invoke-SparkDeckCheckedCommand -FilePath $npm.Source -Arguments @("run", "build") -WorkingDirectory $Paths.Frontend
        [IO.File]::WriteAllText($Paths.FrontendStamp, $frontendFingerprint)
    }
}

function Get-SparkDeckDockerState {
    $docker = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($null -eq $docker) {
        return [pscustomobject]@{ Available = $false; Message = "Docker CLI was not found" }
    }
    $probe = Invoke-SparkDeckNativeCapture -FilePath $docker.Source -Arguments @("info", "--format", "{{.ServerVersion}}|{{.OSType}}") -TimeoutSeconds 5
    if ($probe.ExitCode -ne 0) {
        return [pscustomobject]@{ Available = $false; Message = "Docker Desktop's engine is not running" }
    }
    $parts = $probe.Stdout.Trim().Split('|')
    if ($parts.Count -lt 2 -or $parts[1] -ne "linux") {
        return [pscustomobject]@{ Available = $false; Message = "Docker Desktop is using Windows containers; switch it to Linux containers" }
    }
    return [pscustomobject]@{ Available = $true; Message = ("Docker {0} (Linux engine)" -f $parts[0]) }
}

function Write-SparkDeckDockerWarning {
    $docker = Get-SparkDeckDockerState
    if (-not $docker.Available) {
        Write-Warning ("{0}. SparkDeck will start in controller-only mode; local Docker model actions remain unavailable until the Linux engine is running." -f $docker.Message)
    }
}

function Rotate-SparkDeckLogs {
    param($Paths, [int]$Keep = 5)
    foreach ($base in @($Paths.StdoutLog, $Paths.StderrLog)) {
        $oldest = "{0}.{1}" -f $base, $Keep
        if ([IO.File]::Exists($oldest)) { Remove-Item -LiteralPath $oldest -Force }
        for ($index = $Keep - 1; $index -ge 1; $index--) {
            $source = "{0}.{1}" -f $base, $index
            $destination = "{0}.{1}" -f $base, ($index + 1)
            if ([IO.File]::Exists($source)) { Move-Item -LiteralPath $source -Destination $destination -Force }
        }
        if ([IO.File]::Exists($base)) { Move-Item -LiteralPath $base -Destination ($base + ".1") -Force }
    }
}

function Quote-SparkDeckProcessArgument {
    param([string]$Value)
    # Windows paths cannot contain a literal quote. Doubling trailing slashes
    # keeps CommandLineToArgvW from escaping our closing quote.
    return '"' + (($Value -replace '(\\+)$', '$1$1') -replace '"', '\"') + '"'
}

function Stop-SparkDeckProcessTree {
    param(
        [Parameter(Mandatory = $true)][Diagnostics.Process]$Process,
        $ExpectedStartTimeUtc = $null,
        [int]$TimeoutSeconds = 5
    )

    $Process.Refresh()
    if ($Process.HasExited) { return }
    if ($null -ne $ExpectedStartTimeUtc) {
        $expected = ([DateTime]$ExpectedStartTimeUtc).ToUniversalTime()
        if ([Math]::Abs(($Process.StartTime.ToUniversalTime() - $expected).TotalSeconds) -gt 5) {
            throw ("refusing to terminate PID {0} because its start time no longer matches" -f $Process.Id)
        }
    }

    # A Windows venv python.exe is a redirector that can leave the real base
    # interpreter running as its child. taskkill /T terminates that complete
    # tree; the exact Process/start time check above prevents acting on a stale
    # PID. Invoke-SparkDeckNativeCapture also bounds a hung taskkill command.
    if ($env:OS -eq "Windows_NT") {
        $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
        if (-not [IO.File]::Exists($taskkill)) {
            throw "Windows taskkill.exe was not found"
        }
        $result = Invoke-SparkDeckNativeCapture -FilePath $taskkill `
            -Arguments @("/PID", [string]$Process.Id, "/T", "/F") `
            -TimeoutSeconds $TimeoutSeconds
        $Process.Refresh()
        if ($result.ExitCode -ne 0 -and -not $Process.HasExited) {
            throw ("could not terminate SparkDeck process tree for PID {0}: {1}" -f $Process.Id, $result.Stderr.Trim())
        }
    }
    else {
        $Process.Kill()
    }

    if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
        throw ("timed out waiting for SparkDeck process tree rooted at PID {0} to exit" -f $Process.Id)
    }
}

function Stop-SparkDeckValidatedProcessTree {
    param($Paths, $Record, [int]$TimeoutSeconds = 5)

    if (-not (Test-SparkDeckProcessIdentity $Paths $Record)) { return $false }
    try {
        $process = Get-Process -Id ([int]$Record.pid) -ErrorAction Stop
    }
    catch {
        return $false
    }
    # Revalidate after acquiring the Process object, then pass both that exact
    # object and its recorded start time to the tree terminator.
    if (-not (Test-SparkDeckProcessIdentity $Paths $Record)) { return $false }
    $expectedStart = [DateTime]::Parse([string]$Record.started_at_utc).ToUniversalTime()
    Stop-SparkDeckProcessTree -Process $process -ExpectedStartTimeUtc $expectedStart -TimeoutSeconds $TimeoutSeconds
    return $true
}

function Start-SparkDeck {
    $paths = Get-SparkDeckPaths
    $result = Invoke-WithSparkDeckLock $paths {
        $record = Get-SparkDeckPidRecord $paths
        if (Test-SparkDeckProcessIdentity $paths $record) {
            if (Test-SparkDeckHealth) {
                Write-Host ("SparkDeck is already running at http://localhost:{0} (PID {1})." -f $script:SparkDeckPort, $record.pid)
                return 0
            }
            Write-Host ("SparkDeck process {0} is running but is not ready yet. Use 'sparkdeck logs'." -f $record.pid) -ForegroundColor Yellow
            return 2
        }
        if ($null -ne $record) {
            Write-Warning "Removing a stale SparkDeck PID record; no process was stopped."
            Remove-SparkDeckPidRecord $paths
        }
        if (Test-SparkDeckPortOpen) {
            throw ("port {0} is already in use by another process" -f $script:SparkDeckPort)
        }

        Initialize-SparkDeckEnvironment $paths
        Write-SparkDeckDockerWarning
        Rotate-SparkDeckLogs $paths
        if ([IO.File]::Exists($paths.ShutdownFile)) {
            Remove-Item -LiteralPath $paths.ShutdownFile -Force
        }

        $process = Start-Process -FilePath $paths.VenvPython `
            -ArgumentList @("-u", (Quote-SparkDeckProcessArgument $paths.Server)) `
            -WorkingDirectory $paths.Root `
            -RedirectStandardOutput $paths.StdoutLog `
            -RedirectStandardError $paths.StderrLog `
            -WindowStyle Hidden `
            -PassThru
        $processStartTimeUtc = $process.StartTime.ToUniversalTime()
        try {
            Write-SparkDeckPidRecord $paths $process
        }
        catch {
            $pidRecordError = $_.Exception
            # This is the exact Process instance returned above. Do not use a
            # PID lookup here because the PID record was never persisted and a
            # later lookup could target a reused PID.
            try {
                Stop-SparkDeckProcessTree -Process $process -ExpectedStartTimeUtc $processStartTimeUtc
            }
            catch {
                throw ("SparkDeck could not write its PID record ({0}) and could not terminate the new process tree safely ({1})" -f $pidRecordError.Message, $_.Exception.Message)
            }
            throw $pidRecordError
        }

        $timeout = 60
        if (-not [string]::IsNullOrWhiteSpace($env:SPARKDECK_START_TIMEOUT_SECONDS)) {
            $parsed = 0
            if ([int]::TryParse($env:SPARKDECK_START_TIMEOUT_SECONDS, [ref]$parsed) -and $parsed -gt 0) { $timeout = $parsed }
        }
        $deadline = [DateTime]::UtcNow.AddSeconds($timeout)
        while ([DateTime]::UtcNow -lt $deadline) {
            $process.Refresh()
            if ($process.HasExited) {
                Remove-SparkDeckPidRecord $paths
                $tail = if ([IO.File]::Exists($paths.StderrLog)) {
                    (Get-Content -LiteralPath $paths.StderrLog -Tail 25) -join "`n"
                } else { "No error log was written." }
                throw ("SparkDeck exited during startup.`n{0}" -f $tail)
            }
            if (Test-SparkDeckHealth) {
                Write-Host ("SparkDeck started in the background at http://localhost:{0} (PID {1})." -f $script:SparkDeckPort, $process.Id) -ForegroundColor Green
                Write-Host "Use 'sparkdeck logs' to view its logs."
                return 0
            }
            Start-Sleep -Milliseconds 500
        }

        $current = Get-SparkDeckPidRecord $paths
        Stop-SparkDeckValidatedProcessTree -Paths $paths -Record $current | Out-Null
        Remove-SparkDeckPidRecord $paths
        throw ("SparkDeck did not become healthy within {0} seconds. See {1}" -f $timeout, $paths.StderrLog)
    }
    return [int]$result
}

function Stop-SparkDeck {
    param([switch]$Quiet)
    $paths = Get-SparkDeckPaths
    $result = Invoke-WithSparkDeckLock $paths {
        $record = Get-SparkDeckPidRecord $paths
        if ($null -eq $record) {
            if (-not $Quiet) { Write-Host "SparkDeck is not running." }
            return 0
        }
        if (-not (Test-SparkDeckProcessIdentity $paths $record)) {
            Remove-SparkDeckPidRecord $paths
            if (-not $Quiet) { Write-Warning "Removed a stale SparkDeck PID record; no process was stopped." }
            return 0
        }
        # Ask uvicorn to exit so FastAPI lifespan cleanup can flush state and
        # close clients. Force termination is a bounded last resort only after
        # revalidating that the PID still belongs to this exact checkout.
        [IO.File]::WriteAllText($paths.ShutdownFile, ([string]$record.pid))
        try { Wait-Process -Id ([int]$record.pid) -Timeout 15 -ErrorAction SilentlyContinue } catch { }
        $current = Get-SparkDeckPidRecord $paths
        if (Test-SparkDeckProcessIdentity $paths $current) {
            Write-Warning "SparkDeck did not stop gracefully; forcing its validated process to exit."
            Stop-SparkDeckValidatedProcessTree -Paths $paths -Record $current | Out-Null
        }
        if ([IO.File]::Exists($paths.ShutdownFile)) {
            Remove-Item -LiteralPath $paths.ShutdownFile -Force
        }
        Remove-SparkDeckPidRecord $paths
        if (-not $Quiet) { Write-Host "SparkDeck stopped." -ForegroundColor Green }
        return 0
    }
    return [int]$result
}

function Get-SparkDeckStatus {
    $paths = Get-SparkDeckPaths
    $record = Get-SparkDeckPidRecord $paths
    if ($null -eq $record) {
        Write-Host "SparkDeck is stopped."
        return 1
    }
    if (-not (Test-SparkDeckProcessIdentity $paths $record)) {
        Write-Host "SparkDeck is stopped (the PID record is stale)." -ForegroundColor Yellow
        return 1
    }
    if (Test-SparkDeckHealth) {
        Write-Host ("SparkDeck is healthy at http://localhost:{0} (PID {1})." -f $script:SparkDeckPort, $record.pid) -ForegroundColor Green
        return 0
    }
    Write-Host ("SparkDeck process {0} is running but the API is not healthy." -f $record.pid) -ForegroundColor Yellow
    return 2
}

function Get-SparkDeckProcessStatus {
    $paths = Get-SparkDeckPaths
    $record = Get-SparkDeckPidRecord $paths
    if ($null -eq $record) {
        Write-Host "SparkDeck is stopped."
        return 1
    }
    if (-not (Test-SparkDeckProcessIdentity $paths $record)) {
        Write-Host "SparkDeck is stopped (the PID record is stale)." -ForegroundColor Yellow
        return 1
    }
    Write-Host ("SparkDeck process {0} is owned by this launcher." -f $record.pid) -ForegroundColor Green
    return 0
}

function Show-SparkDeckLogs {
    param([string[]]$Arguments)
    $Arguments = @($Arguments)
    $tail = 100
    $follow = $false
    for ($index = 0; $index -lt $Arguments.Count; $index++) {
        switch ($Arguments[$index]) {
            { $_ -in @("-f", "--follow") } { $follow = $true; continue }
            { $_ -in @("-n", "--tail") } {
                if ($index + 1 -ge $Arguments.Count -or -not [int]::TryParse($Arguments[$index + 1], [ref]$tail) -or $tail -lt 1) {
                    throw "logs --tail requires a positive number"
                }
                $index++
                continue
            }
            default { throw ("unknown logs option: {0}" -f $Arguments[$index]) }
        }
    }
    $paths = Get-SparkDeckPaths
    $logs = @(@($paths.StderrLog, $paths.StdoutLog) | Where-Object { [IO.File]::Exists($_) })
    if ($logs.Count -eq 0) {
        Write-Host "SparkDeck has not written any logs yet."
        return 0
    }
    foreach ($log in $logs) {
        Write-Host ("--- {0} ---" -f $log)
        Get-Content -LiteralPath $log -Tail $tail | ForEach-Object { Write-Host $_ }
    }
    if ($follow) {
        Write-Host "Following logs; press Ctrl+C to stop following."
        Get-Content -LiteralPath $logs -Tail 0 -Wait | ForEach-Object { Write-Host $_ }
    }
    return 0
}

function Start-SparkDeckForeground {
    $paths = Get-SparkDeckPaths
    if (Test-SparkDeckPortOpen) {
        throw ("port {0} is already in use" -f $script:SparkDeckPort)
    }
    Initialize-SparkDeckEnvironment $paths
    Write-SparkDeckDockerWarning
    Write-Host ("Starting SparkDeck at http://localhost:{0}. Press Ctrl+C to stop." -f $script:SparkDeckPort)
    Push-Location $paths.Root
    try { & $paths.VenvPython "-u" $paths.Server 2>&1 | ForEach-Object { Write-Host $_ } }
    finally { Pop-Location }
    return [int]$LASTEXITCODE
}

function Get-NormalizedPathEntries {
    param([string]$PathValue)
    if ([string]::IsNullOrWhiteSpace($PathValue)) { return @() }
    return @($PathValue.Split(';') | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Add-SparkDeckPathEntry {
    param([string]$PathValue, [string]$Entry)
    $normalizedEntry = $Entry.Trim().TrimEnd('\')
    $entries = New-Object Collections.Generic.List[string]
    $found = $false
    foreach ($item in (Get-NormalizedPathEntries $PathValue)) {
        if ([string]::Equals($item.TrimEnd('\'), $normalizedEntry, [StringComparison]::OrdinalIgnoreCase)) {
            if (-not $found) { $entries.Add($normalizedEntry); $found = $true }
        }
        else { $entries.Add($item) }
    }
    if (-not $found) { $entries.Add($normalizedEntry) }
    return ($entries -join ';')
}

function Remove-SparkDeckPathEntry {
    param([string]$PathValue, [string]$Entry)
    $normalizedEntry = $Entry.Trim().TrimEnd('\')
    return ((Get-NormalizedPathEntries $PathValue | Where-Object {
        -not [string]::Equals($_.TrimEnd('\'), $normalizedEntry, [StringComparison]::OrdinalIgnoreCase)
    }) -join ';')
}

function Install-SparkDeckCommand {
    $paths = Get-SparkDeckPaths
    $bin = Join-Path $env:LOCALAPPDATA "SparkDeck\bin"
    [IO.Directory]::CreateDirectory($bin) | Out-Null
    $shim = Join-Path $bin "sparkdeck.cmd"
    $escapedRootCommand = (Join-Path $paths.Root "sparkdeck.cmd").Replace('%', '%%')
    $content = "@echo off`r`ncall `"$escapedRootCommand`" %*`r`nexit /b %ERRORLEVEL%`r`n"
    [IO.File]::WriteAllText($shim, $content, [Text.Encoding]::Default)

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $updated = Add-SparkDeckPathEntry $userPath $bin
    if ($updated -ne $userPath) {
        [Environment]::SetEnvironmentVariable("Path", $updated, "User")
    }
    Write-Host ("Installed the SparkDeck command at {0}." -f $shim) -ForegroundColor Green
    Write-Host "Open a new PowerShell window, then run 'sparkdeck start'."
    return 0
}

function Uninstall-SparkDeckCommand {
    $bin = Join-Path $env:LOCALAPPDATA "SparkDeck\bin"
    $shim = Join-Path $bin "sparkdeck.cmd"
    if ([IO.File]::Exists($shim)) { Remove-Item -LiteralPath $shim -Force }
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $updated = Remove-SparkDeckPathEntry $userPath $bin
    if ($updated -ne $userPath) {
        [Environment]::SetEnvironmentVariable("Path", $updated, "User")
    }
    Write-Host "Uninstalled the SparkDeck command. The application files and data were not removed." -ForegroundColor Green
    return 0
}

function Show-SparkDeckHelp {
    Write-Host @"
SparkDeck for Windows

Usage: sparkdeck <command>

  install             Add the sparkdeck command to your user PATH
  uninstall           Remove the PATH command (keeps application data)
  start               Prepare and start SparkDeck in the background
  stop                Stop the background SparkDeck process
  restart             Stop and start the background process
  status              Show process and API health
  logs [-f] [-n N]    Show logs; -f follows new output
  run                 Run in the foreground (Ctrl+C stops it)
  help                Show this help

Docker Desktop is optional for the controller UI, but local model actions
require Docker Desktop's Linux engine and compatible NVIDIA GPU support.
"@
    return 0
}

function Invoke-SparkDeckCommand {
    param([string]$Command, [string[]]$Arguments = @())
    $Arguments = @($Arguments)
    switch ($Command.ToLowerInvariant()) {
        "install" { return (Install-SparkDeckCommand) }
        "uninstall" { return (Uninstall-SparkDeckCommand) }
        "start" { return (Start-SparkDeck) }
        "stop" { return (Stop-SparkDeck) }
        "restart" {
            $stopped = Stop-SparkDeck -Quiet
            if ($stopped -ne 0) { return $stopped }
            return (Start-SparkDeck)
        }
        "status" { return (Get-SparkDeckStatus) }
        # Internal updater preflight: the API endpoint handling this command is
        # already healthy, so probing it again would deadlock that request.
        "process-status" { return (Get-SparkDeckProcessStatus) }
        "logs" { return (Show-SparkDeckLogs -Arguments $Arguments) }
        "run" { return (Start-SparkDeckForeground) }
        "help" { return (Show-SparkDeckHelp) }
        "--help" { return (Show-SparkDeckHelp) }
        "-h" { return (Show-SparkDeckHelp) }
        default {
            Write-Host ("Unknown command: {0}" -f $Command) -ForegroundColor Red
            Show-SparkDeckHelp | Out-Null
            return 1
        }
    }
}

Export-ModuleMember -Function @(
    "Get-SparkDeckPaths",
    "Get-SparkDeckPidRecord",
    "Write-SparkDeckPidRecord",
    "Remove-SparkDeckPidRecord",
    "Test-SparkDeckProcessIdentity",
    "Test-SparkDeckHealth",
    "Test-SparkDeckPortOpen",
    "Invoke-SparkDeckCheckedCommand",
    "Get-SparkDeckFrontendFingerprint",
    "Stop-SparkDeckProcessTree",
    "Add-SparkDeckPathEntry",
    "Remove-SparkDeckPathEntry",
    "Invoke-SparkDeckCommand"
)
