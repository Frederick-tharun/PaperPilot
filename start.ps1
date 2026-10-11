# Starts Ollama, the PaperPilot API and the Streamlit UI.
# Double-click start.bat (or run this file) to launch everything.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$streamlit = Join-Path $root ".venv\Scripts\streamlit.exe"
$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"

function Test-Port($port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-Port($port, $name, $seconds = 60) {
    for ($i = 0; $i -lt $seconds; $i++) {
        if (Test-Port $port) { return }
        Start-Sleep -Seconds 1
    }
    throw "$name did not start on port $port within $seconds seconds."
}

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Create it first: python -m venv .venv"
}

# 1. Ollama (the local LLM server on port 11434)
if (Test-Port 11434) {
    Write-Host "Ollama already running."
} elseif (Test-Path $ollama) {
    Write-Host "Starting Ollama..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Minimized
    Wait-Port 11434 "Ollama"
} else {
    throw "Ollama not found at $ollama. Install it from https://ollama.com"
}

# 2. FastAPI backend (port 8000)
if (Test-Port 8000) {
    Write-Host "API already running on port 8000."
} else {
    Write-Host "Starting API on http://localhost:8000 ..."
    Start-Process -FilePath $python -WorkingDirectory $root `
        -ArgumentList "-m", "uvicorn", "paperpilot.api:app", "--host", "127.0.0.1", "--port", "8000" `
        -WindowStyle Minimized
    Wait-Port 8000 "API"
}

# 3. Streamlit UI (port 8501)
if (Test-Port 8501) {
    Write-Host "Streamlit already running on port 8501."
} else {
    Write-Host "Starting Streamlit on http://localhost:8501 ..."
    Start-Process -FilePath $streamlit -WorkingDirectory $root `
        -ArgumentList "run", "app/streamlit_app.py", "--server.headless", "true" `
        -WindowStyle Minimized
    Wait-Port 8501 "Streamlit"
}

Start-Process "http://localhost:8501"
Write-Host ""
Write-Host "PaperPilot is running:"
Write-Host "  UI   http://localhost:8501"
Write-Host "  API  http://localhost:8000/docs"
Write-Host "Close the minimized API and Streamlit windows to shut them down."
