# Check if two arguments are provided
param (
    [Parameter(Mandatory=$true)]
    [string]$FolderName,

    [Parameter(Mandatory=$true)]
    [string]$Hostname
)

# Set the server path based on the hostname
if ($Hostname -eq "vast") {
    $ServerPath = "~/kaggle1stReimp/checkpoints"
} else {
    $ServerPath = "/scratch/medfm/vesuv/kaggle1stReimp/checkpoints"
}

# Full path on the server
$FullServerPath = "$ServerPath/$FolderName"
Write-Output ${FullServerPath}

# Define the local path where you want to save the folder
$LocalPath = Join-Path -Path (Get-Location) -ChildPath "checkpoints\${FolderName}"

# Using scp to copy the directory. SCP needs to be installed on Windows, typically via OpenSSH
$ScpCommand = "scp -r ${Hostname}:${FullServerPath} ${LocalPath}"
Write-Output ${ScpCommand}

Invoke-Expression $ScpCommand

# Check if scp succeeded
if ($LastExitCode -eq 0) {
    Write-Host "Folder copied successfully."
} else {
    Write-Host "Error in copying folder."
}
