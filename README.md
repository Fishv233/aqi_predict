# AQI Prediction

## Project Overview
This project focuses on Air Quality Index (AQI) prediction and consists of two main modules:

### 1. Data Processing Module
The data processing module is divided into three main pipelines:
- Feature Pipeline
- Training Pipeline
- Batch Prediction Pipeline

All pipeline workflows are orchestrated using Apache Airflow and deployed using Docker containers.

### 2. Web System Module
The web interface is built using:
- FastAPI for backend services
- Streamlit for frontend visualization

## System Architecture
[Diagram placeholder]

## Getting Started
[Installation instructions placeholder]

## Prerequisites
- Docker
- Python 3.x
- Apache Airflow
- FastAPI
- Streamlit

## Installation

### Prerequisites
- Docker
- Python 3.x
- Apache Airflow
- FastAPI
- Streamlit
- Apache2-utils
- Poetry
- Passlib

### Environment Setup

1. **Configure Environment Variables**
   - All sensitive data configurations are stored in `airflow/dags/.env.default`
   - Airflow initialization parameters should follow this format:
   ```json
   {
     "ts_col": "",
     "datetime_format": "",
     "project_name": "",
     "feature_group_description": {
       "name": "",
       "version": 1,
       "description": ".",
       "primary_key": [],
       "event_time": "",
       "online_enabled": false
     }
   }
   ```

2. **Set Up PyPI Server Authentication**
   ```bash
   # Install required dependencies
   sudo apt install -y apache2-utils
   pip install passlib

   # Create credentials directory and file
   mkdir ~/.htpasswd
   htpasswd -sc ~/.htpasswd/htpasswd.txt energy-forecasting
   ```

3. **Configure Poetry with PyPI Settings**
   ```bash
   # Initialize PyPI settings in Poetry
   poetry config repositories.my-pypi http://localhost
   poetry config http-basic.my-pypi energy-forecasting <password>
   ```

4. **Initialize and Start Docker Services**
   ```bash
   # Navigate to Airflow directory
   cd ./airflow

   # Initialize Airflow database
   docker compose up airflow-init

   # Start all services
   docker compose --env-file .env up --build -d
   ```

5. **Deploy Pipeline**
   - Execute `ml-pipeline.sh` in the deploy directory to monitor pipeline processes in the Airflow DAGs

### Important Notes
- The data pipeline requires a private PyPI server (configured in `airflow/docker-compose.yaml`)
- Ensure PyPI server credentials are properly set up before starting Docker services
- Monitor pipeline progress through the Airflow DAGs interface

## Usage
[Usage instructions placeholder]

## Pipeline Details
[Pipeline documentation placeholder]

## Web Interface
[Web interface documentation placeholder]

## Contributing
[Contributing guidelines placeholder]

## License
[License information placeholder]


