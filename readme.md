# Smart Stock Predictor

A comprehensive stock prediction system built with machine learning, featuring a Django backend for data processing and API services, and a React frontend for user interaction.

## Features

- Stock price prediction using advanced ML models
- Asset-aware prediction algorithms
- Interactive web interface
- RESTful API for data access

## Project Structure

- `backend/`: Django application for backend services
- `frontend/`: React application for frontend interface
- `models/`: Trained machine learning models
- `cache/`: Cached data and computations
- `documents/`: Project documentation and reports

## Setup

### Backend

1. Navigate to `backend/stockproject/`
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run migrations: `python manage.py migrate`
6. Start server: `python manage.py runserver`

### Frontend

1. Navigate to `frontend/`
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`

## Usage

- Access the frontend at `http://localhost:5173` (default Vite port)
- Backend API at `http://localhost:8000`

