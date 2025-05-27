# Enhanced Deployment script for Supabase Vector Database (Windows PowerShell)
# Automatically loads .env from vector_database directory
# Usage: .\scripts\deploy.ps1 [options]

param(
    [switch]$DryRun,
    [switch]$Verbose,
    [switch]$SkipTests,
    [switch]$Help
)

# Enable strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Cyan"
    NC = "White"
}

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $ProjectDir ".env"

# Show help if requested
if ($Help) {
    Write-Host "Usage: .\deploy.ps1 [options]"
    Write-Host "Options:"
    Write-Host "  -DryRun       Show what would be done without executing"
    Write-Host "  -Verbose      Enable verbose output"
    Write-Host "  -SkipTests    Skip connection and validation tests"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "Environment:"
    Write-Host "  Loads configuration from: $EnvFile"
    exit 0
}

Write-Host "🚀 Starting Supabase Vector Database Deployment" -ForegroundColor $Colors.Green

# Function to log messages
function Write-Log {
    param(
        [string]$Level,
        [string]$Message
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    switch ($Level) {
        "INFO" {
            Write-Host "[$timestamp] INFO: $Message" -ForegroundColor $Colors.Blue
        }
        "SUCCESS" {
            Write-Host "[$timestamp] SUCCESS: $Message" -ForegroundColor $Colors.Green
        }
        "WARNING" {
            Write-Host "[$timestamp] WARNING: $Message" -ForegroundColor $Colors.Yellow
        }
        "ERROR" {
            Write-Host "[$timestamp] ERROR: $Message" -ForegroundColor $Colors.Red
        }
    }
    
    if ($Verbose) {
        $logFile = Join-Path $ProjectDir "deployment.log"
        Add-Content -Path $logFile -Value "[$timestamp] $Level`: $Message"
    }
}

# Function to load .env file
function Import-EnvFile {
    param(
        [string]$Path
    )
    
    if (Test-Path $Path) {
        Write-Log "INFO" "Loading environment from: $Path"
        
        $envCount = 0
        Get-Content $Path | ForEach-Object {
            $line = $_.Trim()
            
            # Skip empty lines and comments
            if ($line -and !$line.StartsWith('#')) {
                if ($line -match '^([^=]+)=(.*)$') {
                    $key = $matches[1].Trim()
                    $value = $matches[2].Trim()
                    
                    # Remove quotes if present
                    if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") {
                        $value = $matches[1]
                    }
                    
                    # Set environment variable
                    [Environment]::SetEnvironmentVariable($key, $value, [EnvironmentVariableTarget]::Process)
                    $envCount++
                    
                    if ($Verbose) {
                        Write-Log "INFO" "Loaded: $key"
                    }
                }
            }
        }
        
        Write-Log "SUCCESS" "Loaded $envCount environment variables from .env file"
    }
    else {
        Write-Log "WARNING" "No .env file found at: $Path"
    }
}

# Function to create example .env file
function New-EnvExample {
    $exampleContent = @"
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here

# Optional: PostgreSQL connection settings
PGSSLMODE=require
"@
    
    $examplePath = Join-Path $ProjectDir ".env.example"
    Set-Content -Path $examplePath -Value $exampleContent
    Write-Log "INFO" "Created example environment file at: $examplePath"
}

# Load environment file
Import-EnvFile -Path $EnvFile

# Check required environment variables
function Test-Environment {
    Write-Log "INFO" "Checking environment variables..."
    
    $missingVars = @()
    
    if (-not $env:SUPABASE_URL) {
        $missingVars += "SUPABASE_URL"
    }
    
    if (-not $env:SUPABASE_SERVICE_KEY) {
        $missingVars += "SUPABASE_SERVICE_KEY"
    }
    
    if ($missingVars.Count -gt 0) {
        Write-Log "ERROR" "Missing required environment variables:"
        foreach ($var in $missingVars) {
            Write-Host "  - $var" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "Please create a .env file at:" -ForegroundColor Yellow
        Write-Host "  $EnvFile" -ForegroundColor White
        Write-Host ""
        Write-Host "Example .env content:" -ForegroundColor Yellow
        Write-Host "SUPABASE_URL=https://your-project.supabase.co" -ForegroundColor Gray
        Write-Host "SUPABASE_SERVICE_KEY=your-service-key-here" -ForegroundColor Gray
        Write-Host ""
        
        # Offer to create example file
        $createExample = Read-Host "Create .env.example file? (y/n)"
        if ($createExample -eq 'y') {
            New-EnvExample
        }
        
        exit 1
    }
    
    Write-Log "SUCCESS" "Environment variables are set"
}

# Check if psql is available (lightweight check)
function Test-PsqlAvailable {
    Write-Log "INFO" "Checking for PostgreSQL client (psql)..."
    
    $psqlCommand = Get-Command psql -ErrorAction SilentlyContinue
    
    if (-not $psqlCommand) {
        Write-Log "ERROR" "psql command not found!"
        Write-Host ""
        Write-Host "You need to install PostgreSQL client tools (psql) to deploy to Supabase." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Installation options:" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Option 1: PostgreSQL Client Tools Only (Recommended)" -ForegroundColor Green
        Write-Host "  Download from: https://www.postgresql.org/download/windows/"
        Write-Host "  - Choose 'Download the installer'"
        Write-Host "  - During installation, you can select ONLY 'Command Line Tools'"
        Write-Host "  - This installs just psql without the PostgreSQL server"
        Write-Host ""
        Write-Host "Option 2: Using Chocolatey" -ForegroundColor Green
        Write-Host "  choco install postgresql --params '/Client'"
        Write-Host ""
        Write-Host "Option 3: Using Scoop" -ForegroundColor Green
        Write-Host "  scoop install postgresql"
        Write-Host ""
        Write-Host "After installation, make sure PostgreSQL bin directory is in your PATH" -ForegroundColor Yellow
        Write-Host "Default location: C:\Program Files\PostgreSQL\16\bin" -ForegroundColor Gray
        exit 1
    }
    
    Write-Log "SUCCESS" "Found psql at: $($psqlCommand.Source)"
}

# Extract database connection details
function Get-DatabaseDetails {
    Write-Log "INFO" "Extracting database connection details..."
    
    # Extract host from URL
    $script:DbHost = $env:SUPABASE_URL -replace 'https://', '' -replace '\.supabase\.co.*', '.supabase.co'
    $script:DbName = "postgres"
    $script:DbUser = "postgres"
    
    if ($Verbose) {
        Write-Log "INFO" "Database host: db.$DbHost"
        Write-Log "INFO" "Database name: $DbName"
        Write-Log "INFO" "Database user: $DbUser"
    }
}

# Execute SQL file
function Invoke-SqlFile {
    param(
        [string]$SqlFile,
        [string]$Description
    )
    
    Write-Log "INFO" "Executing: $Description"
    
    $fullPath = Join-Path $ProjectDir $SqlFile
    
    if (-not (Test-Path $fullPath)) {
        Write-Log "ERROR" "SQL file not found: $fullPath"
        exit 1
    }
    
    if ($DryRun) {
        Write-Log "INFO" "[DRY RUN] Would execute: $SqlFile"
        return
    }
    
    # Set PGPASSWORD environment variable temporarily
    $env:PGPASSWORD = $env:SUPABASE_SERVICE_KEY
    
    try {
        $result = & psql `
            -h "db.$DbHost" `
            -U "$DbUser" `
            -d "$DbName" `
            -f "$fullPath" `
            -v ON_ERROR_STOP=1 `
            --quiet 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "SUCCESS" "Successfully executed: $Description"
        }
        else {
            Write-Log "ERROR" "Failed to execute: $Description"
            if ($Verbose -or $result) {
                Write-Log "ERROR" "Error details:"
                Write-Host $result -ForegroundColor Red
            }
            exit 1
        }
    }
    catch {
        Write-Log "ERROR" "Exception while executing: $Description"
        Write-Log "ERROR" $_.Exception.Message
        exit 1
    }
    finally {
        # Clear the password from environment
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Test database connection
function Test-DatabaseConnection {
    Write-Log "INFO" "Testing database connection..."
    
    if ($DryRun) {
        Write-Log "INFO" "[DRY RUN] Would test database connection"
        return
    }
    
    $env:PGPASSWORD = $env:SUPABASE_SERVICE_KEY
    
    try {
        $result = & psql `
            -h "db.$DbHost" `
            -U "$DbUser" `
            -d "$DbName" `
            -c "SELECT version();" `
            --quiet `
            --tuples-only 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "SUCCESS" "Database connection successful"
            if ($Verbose) {
                Write-Log "INFO" "PostgreSQL version: $($result.Trim())"
            }
        }
        else {
            Write-Log "ERROR" "Database connection failed"
            Write-Host ""
            Write-Host "Troubleshooting:" -ForegroundColor Yellow
            Write-Host "1. Check your SUPABASE_URL: $env:SUPABASE_URL"
            Write-Host "2. Verify your SUPABASE_SERVICE_KEY is correct"
            Write-Host "3. Ensure your Supabase project is active"
            Write-Host "4. Check if you're behind a firewall/proxy"
            exit 1
        }
    }
    catch {
        Write-Log "ERROR" "Exception during connection test: $_"
        exit 1
    }
    finally {
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Check pgvector extension
function Test-VectorExtension {
    Write-Log "INFO" "Checking pgvector extension..."
    
    if ($DryRun) {
        Write-Log "INFO" "[DRY RUN] Would check pgvector extension"
        return
    }
    
    $env:PGPASSWORD = $env:SUPABASE_SERVICE_KEY
    
    try {
        # First, try to create the extension (will succeed if not exists, fail silently if exists)
        & psql `
            -h "db.$DbHost" `
            -U "$DbUser" `
            -d "$DbName" `
            -c "CREATE EXTENSION IF NOT EXISTS vector;" `
            --quiet 2>&1 | Out-Null
        
        # Now check if it exists
        $result = & psql `
            -h "db.$DbHost" `
            -U "$DbUser" `
            -d "$DbName" `
            -c "SELECT extname FROM pg_extension WHERE extname = 'vector';" `
            --quiet `
            --tuples-only 2>&1
        
        if ($LASTEXITCODE -eq 0 -and $result -match "vector") {
            Write-Log "SUCCESS" "pgvector extension is enabled"
        }
        else {
            Write-Log "ERROR" "pgvector extension is not available"
            Write-Host ""
            Write-Host "Please enable pgvector in your Supabase project:" -ForegroundColor Yellow
            Write-Host "1. Go to your Supabase dashboard"
            Write-Host "2. Navigate to Database > Extensions"
            Write-Host "3. Search for 'vector' and enable it"
            Write-Host "4. Run this script again"
            exit 1
        }
    }
    finally {
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
}

# Main deployment process
try {
    Write-Log "INFO" "Starting deployment process..."
    Write-Log "INFO" "Project directory: $ProjectDir"
    
    # Pre-flight checks
    Test-Environment
    Test-PsqlAvailable
    Get-DatabaseDetails
    
    if (-not $SkipTests) {
        Test-DatabaseConnection
        Test-VectorExtension
    }
    
    # Deploy schema files
    Invoke-SqlFile -SqlFile "setup.sql" -Description "Main database schema"
    Invoke-SqlFile -SqlFile "functions.sql" -Description "Search functions"
    
    # Optional security file
    if (Test-Path (Join-Path $ProjectDir "security.sql")) {
        Invoke-SqlFile -SqlFile "security.sql" -Description "Security policies"
    }
    
    if (-not $DryRun) {
        Write-Log "SUCCESS" "Deployment completed successfully!"
        Write-Host ""
        Write-Host "🎉 Database deployed to Supabase cloud!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "1. Your database is ready in the cloud"
        Write-Host "2. Start the ingestion service to process documents"
        Write-Host "3. Use the RAG agent to query your data"
    }
    else {
        Write-Log "INFO" "Dry run completed - no changes were made"
    }
}
catch {
    Write-Log "ERROR" "Deployment failed: $_"
    exit 1
}