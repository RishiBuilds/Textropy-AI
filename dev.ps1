param(
    [Parameter(Position=0)]
    [ValidateSet("start", "api", "all", "install", "docker-build", "docker-up", "docker-down")]
    [string]$Command = "start"
)

switch ($Command) {
    "install" {
        Write-Host "Installing dependencies..." -ForegroundColor Cyan
        pip install -r requirements.txt
    }
    "start" {
        Write-Host "Starting Textropy AI Dashboard..." -ForegroundColor Green
        streamlit run app.py
    }
    "api" {
        Write-Host "Starting Textropy AI API..." -ForegroundColor Green
        python -m uvicorn api.main:app --reload --port 8000
    }
    "all" {
        Write-Host "Starting API + Dashboard..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-Command", "python -m uvicorn api.main:app --reload --port 8000"
        Start-Sleep -Seconds 2
        streamlit run app.py
    }
    "docker-build" {
        Write-Host "Building Docker images..." -ForegroundColor Cyan
        docker compose build
    }
    "docker-up" {
        Write-Host "Starting Docker containers..." -ForegroundColor Green
        docker compose up -d
    }
    "docker-down" {
        Write-Host "Stopping Docker containers..." -ForegroundColor Yellow
        docker compose down
    }
}
