# Check if Python is installed
try {
    $null = python --version
} catch {
    Write-Host "[ERROR] Python is not installed or not in your system's PATH." -ForegroundColor Red
    Write-Host "Please install Python (and check 'Add Python to PATH' during installation)."
    Write-Host "You can download it from https://www.python.org/"
    Read-Host "Press Enter to exit"
    exit 1
}

# Resolve script path
$ScriptPath = Join-Path $PSScriptRoot "rover_navigation_simulation.py"

# Run Python script passing all arguments
python $ScriptPath $args

if ($args.Count -eq 0) {
    Write-Host ""
    Read-Host "Command completed. Press Enter to exit"
}
