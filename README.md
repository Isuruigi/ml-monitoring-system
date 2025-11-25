# ML Monitoring System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.0.4-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Production-ready ML model monitoring system with automated drift detection, retraining, and real-time dashboard**

![ML Monitoring System](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🚀 Features

- **Real-time Monitoring**: Track model performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Drift Detection**: Automated data drift monitoring with Evidently AI
- **Auto-Retraining**: Smart triggers based on drift, accuracy, and schedule
- **REST API**: 15+ FastAPI endpoints with Swagger documentation
- **Modern Dashboard**: Next.js frontend with glassmorphism UI
- **MLflow Integration**: Experiment tracking and model registry
- **Database Persistence**: PostgreSQL/SQLite with SQLAlchemy
- **Docker Deployment**: Complete stack with docker-compose
- **CI/CD Pipeline**: GitHub Actions for testing and deployment
- **Comprehensive Testing**: 50+ unit and integration tests

## 📊 Architecture

```
Backend (FastAPI) → Database (PostgreSQL) → MLflow
     ↓                    ↓
Predictions API     Model Registry
     ↓                    ↓
Monitoring API    ← Drift Detection
     ↓
Frontend Dashboard (Next.js)
```

## 🏃 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ml-monitoring-system.git
cd ml-monitoring-system

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Access services:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001
```

### Option 2: Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📖 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Detailed setup instructions
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI
- **[Project Summary](FINAL_SUMMARY.md)** - Complete feature overview

## 🛠️ Tech Stack

**Backend:**
- FastAPI 0.104.1
- XGBoost 2.0.3
- Scikit-learn 1.3.2
- Evidently 0.4 (Drift Detection)
- MLflow 2.9.2
- SQLAlchemy 2.0.23
- Prometheus Client

**Frontend:**
- Next.js 14
- TypeScript
- Tailwind CSS
- Recharts
- Lucide Icons

**Infrastructure:**
- Docker & Docker Compose
- PostgreSQL 15
- Redis 7
- Prometheus
- Grafana

## 📁 Project Structure

```
ml-monitoring-system/
├── backend/           # FastAPI application
│   ├── api/          # REST API routes
│   ├── ml/           # ML models & training
│   ├── data/         # Data loading & database
│   └── monitoring/   # Metrics & drift detection
├── frontend/         # Next.js dashboard
│   ├── app/         # Pages & layouts
│   └── components/  # React components
├── tests/           # Test suite
├── docker/          # Dockerfiles
└── .github/         # CI/CD workflows
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=backend

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage report
pytest tests/ --cov=backend --cov-report=html
```

## 🚢 Deployment

The system includes:
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Docker Support**: Multi-stage builds for optimized images
- **Health Checks**: All services monitored
- **Database Migrations**: Automatic on deployment

See [deployment guide](docs/DEPLOYMENT.md) for details.

## 📊 Metrics Tracked

- **Model Performance**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Drift Metrics**: PSI per feature, overall drift score
- **System Metrics**: API latency, prediction volume, uptime
- **Business Metrics**: Prediction accuracy over time

## 🔒 Security

- API key authentication
- Admin key for model management
- Environment-based secrets
- Rate limiting
- Input validation

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🎯 Roadmap

- [ ] Advanced model comparison
- [ ] Email/Slack alerts
- [ ] Multi-model support
- [ ] A/B testing framework
- [ ] Cloud deployment guides (AWS/GCP/Azure)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using modern MLOps practices**

⭐ Star this repo if you find it useful!
