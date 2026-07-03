@echo off
echo Building Docker image for PlaceMux API (Task 13)...
docker build -t placemux-api:latest .

echo.
echo Starting PlaceMux API container on port 8000...
docker run -d -p 8000:8000 --name placemux-api-container placemux-api:latest

echo.
echo API is running! 
echo Health check: http://localhost:8000/v1/health
echo Swagger UI: http://localhost:8000/docs
echo.
echo To view logs, run: docker logs -f placemux-api-container
echo To stop, run: docker stop placemux-api-container ^&^& docker rm placemux-api-container
pause
