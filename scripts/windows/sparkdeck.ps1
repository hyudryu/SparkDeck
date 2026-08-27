[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$CommandArguments = @()
)

$ErrorActionPreference = "Stop"

Import-Module (Join-Path $PSScriptRoot "SparkDeck.Windows.psm1") -Force

try {
    $exitCode = Invoke-SparkDeckCommand -Command $Command -Arguments $CommandArguments
    exit $exitCode
}
catch {
    Write-Host ("SparkDeck: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
