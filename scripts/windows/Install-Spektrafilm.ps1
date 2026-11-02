$ErrorActionPreference = 'Stop'

$PackageRoot = $PSScriptRoot
$InstallRoot = Join-Path $env:LOCALAPPDATA 'Programs\Spektrafilm'
$SourceRoot = Join-Path $InstallRoot 'Source'
$EnvironmentRoot = Join-Path $InstallRoot '.venv'
$PythonLauncher = Get-Command 'py.exe' -ErrorAction SilentlyContinue

if ($null -eq $PythonLauncher) {
    throw 'Python 3.13 (64-bit) is required. Install it from https://www.python.org/downloads/windows/ and run this installer again.'
}

& $PythonLauncher.Source -3.13 --version
if ($LASTEXITCODE -ne 0) {
    throw 'Python 3.13 (64-bit) was not found. Install it from https://www.python.org/downloads/windows/ and run this installer again.'
}
$PythonDetails = & $PythonLauncher.Source -3.13 -c "import struct,sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor)+' '+str(struct.calcsize('P')*8))"
if ($LASTEXITCODE -ne 0 -or $PythonDetails -notmatch '^3\.13 64$') {
    throw 'Python 3.13 (64-bit) is required. Install the 64-bit version and run this installer again.'
}

New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null
New-Item -ItemType Directory -Path $SourceRoot -Force | Out-Null
Copy-Item -Path (Join-Path $PackageRoot 'Source\*') -Destination $SourceRoot -Recurse -Force
Copy-Item -Path (Join-Path $PackageRoot 'Launch-Spektrafilm.bat') -Destination $InstallRoot -Force
Copy-Item -Path (Join-Path $PackageRoot 'Spektrafilm.ico') -Destination $InstallRoot -Force

if (-not (Test-Path (Join-Path $EnvironmentRoot 'Scripts\python.exe'))) {
    & $PythonLauncher.Source -3.13 -m venv $EnvironmentRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the Spektrafilm Python environment.'
    }
}

$Python = Join-Path $EnvironmentRoot 'Scripts\python.exe'
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Could not update pip.'
}

& $Python -m pip install $SourceRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Spektrafilm dependencies could not be installed. Check your internet connection and Python version, then run this installer again.'
}

$ShortcutPaths = @(
    (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Spektrafilm.lnk'),
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Spektrafilm.lnk')
)
$Shell = New-Object -ComObject WScript.Shell
foreach ($ShortcutPath in $ShortcutPaths) {
    $ShortcutDirectory = Split-Path -Parent $ShortcutPath
    New-Item -ItemType Directory -Path $ShortcutDirectory -Force | Out-Null
    $Shortcut = $Shell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = Join-Path $EnvironmentRoot 'Scripts\pythonw.exe'
    $Shortcut.Arguments = '-m spektrafilm_gui.app'
    $Shortcut.WorkingDirectory = $SourceRoot
    $Shortcut.IconLocation = (Join-Path $InstallRoot 'Spektrafilm.ico')
    $Shortcut.Save()
}

Write-Host ''
Write-Host 'Spektrafilm is installed. Launch it from the Start menu or desktop shortcut.'
Write-Host "Install location: $InstallRoot"
