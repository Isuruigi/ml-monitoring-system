# Deployment Guide

This guide covers deploying the ML Monitoring System to various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployments](#cloud-deployments)
  - [AWS](#aws-deployment)
  - [Google Cloud Platform](#gcp-deployment)
  - [Azure](#azure-deployment)
  - [Heroku](#heroku-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring & Maintenance](#monitoring--maintenance)

## Prerequisites

### Required
- Docker 20.10+ and Docker Compose 2.0+
- Git
- Domain name (for production)
- SSL certificate (recommended)

### Recommended
- CI/CD pipeline (GitHub Actions, GitLab CI, etc.)
- Monitoring tools (Prometheus, Grafana)
- Log aggregation (ELK Stack, Loki)

## Docker Deployment

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/ml-monitoring-system.git
cd ml-monitoring-system
```

2. **Configure environment variables**:
```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

Update these critical variables:
```bash
# Security
API_KEY=<generate-strong-random-key>
ADMIN_API_KEY=<generate-strong-random-key>
DB_PASSWORD=<secure-database-password>

# Database
DATABASE_URL=postgresql://mluser:${DB_PASSWORD}@postgres:5432/ml_monitoring

# Grafana (optional)
GRAFANA_PASSWORD=<secure-grafana-password>
```

3. **Start all services**:
```bash
docker-compose up -d
```

4. **Verify deployment**:
```bash
# Check service health
docker-compose ps

# Check logs
docker-compose logs -f backend

# Test API
curl http://localhost:8000/health
```

### Production Docker Deployment

For production, use the production compose file:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

Key differences in production:
- Multi-stage builds for smaller images
- Non-root user execution
- Health checks configured
- Resource limits set
- Production-ready configurations

## Cloud Deployments

### AWS Deployment

#### Option 1: EC2 with Docker

1. **Launch EC2 instance**:
   - AMI: Amazon Linux 2 or Ubuntu 22.04
   - Instance Type: t3.medium or larger
   - Storage: 30GB minimum
   - Security Group: Allow ports 22, 80, 443, 8000, 3000

2. **Install Docker**:
```bash
# Amazon Linux 2
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. **Deploy**:
```bash
git clone https://github.com/your-username/ml-monitoring-system.git
cd ml-monitoring-system
cp .env.example .env
# Edit .env with production values
docker-compose up -d
```

4. **Setup Nginx reverse proxy** (optional):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Option 2: ECS (Elastic Container Service)

1. **Create ECR repositories**:
```bash
aws ecr create-repository --repository-name ml-monitoring-backend
aws ecr create-repository --repository-name ml-monitoring-frontend
```

2. **Build and push images**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push backend
docker build -f docker/Dockerfile.backend -t ml-monitoring-backend .
docker tag ml-monitoring-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-monitoring-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-monitoring-backend:latest

# Build and push frontend
docker build -f docker/Dockerfile.frontend -t ml-monitoring-frontend .
docker tag ml-monitoring-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-monitoring-frontend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-monitoring-frontend:latest
```

3. **Create ECS Task Definition** and deploy via AWS Console or CLI

#### Option 3: EKS (Kubernetes)

See `kubernetes/` directory for deployment manifests.

### GCP Deployment

#### Cloud Run

1. **Enable required APIs**:
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

2. **Build and push to Container Registry**:
```bash
# Backend
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-monitoring-backend

# Frontend
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-monitoring-frontend
```

3. **Deploy to Cloud Run**:
```bash
# Backend
gcloud run deploy ml-monitoring-backend \
  --image gcr.io/PROJECT_ID/ml-monitoring-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "DATABASE_URL=$DATABASE_URL,API_KEY=$API_KEY"

# Frontend
gcloud run deploy ml-monitoring-frontend \
  --image gcr.io/PROJECT_ID/ml-monitoring-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "NEXT_PUBLIC_API_URL=$API_URL"
```

#### GKE (Google Kubernetes Engine)

1. **Create GKE cluster**:
```bash
gcloud container clusters create ml-monitoring-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --region us-central1
```

2. **Deploy using kubectl**:
```bash
kubectl apply -f kubernetes/
```

### Azure Deployment

#### Azure Container Instances

1. **Create resource group**:
```bash
az group create --name ml-monitoring-rg --location eastus
```

2. **Create container registry**:
```bash
az acr create --resource-group ml-monitoring-rg \
  --name mlmonitoringregistry --sku Basic
```

3. **Build and push images**:
```bash
az acr build --registry mlmonitoringregistry \
  --image ml-monitoring-backend:latest \
  -f docker/Dockerfile.backend .
```

4. **Deploy container**:
```bash
az container create \
  --resource-group ml-monitoring-rg \
  --name ml-monitoring-backend \
  --image mlmonitoringregistry.azurecr.io/ml-monitoring-backend:latest \
  --dns-name-label ml-monitoring \
  --ports 8000 \
  --environment-variables API_KEY=$API_KEY DATABASE_URL=$DATABASE_URL
```

#### AKS (Azure Kubernetes Service)

Similar to EKS/GKE - use Kubernetes manifests in `kubernetes/` directory.

### Heroku Deployment

1. **Install Heroku CLI** and login:
```bash
heroku login
```

2. **Create Heroku app**:
```bash
heroku create ml-monitoring-app
```

3. **Add PostgreSQL**:
```bash
heroku addons:create heroku-postgresql:standard-0
```

4. **Set environment variables**:
```bash
heroku config:set API_KEY=your-key
heroku config:set ADMIN_API_KEY=your-admin-key
```

5. **Deploy**:
```bash
git push heroku main
```

6. **Scale dynos**:
```bash
heroku ps:scale web=2
```

## Production Considerations

### Security

1. **API Keys**:
   - Use strong, randomly generated keys
   - Rotate keys regularly
   - Never commit keys to version control
   - Use secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)

2. **HTTPS/SSL**:
   - Always use HTTPS in production
   - Use Let's Encrypt for free SSL certificates
   - Configure SSL termination at load balancer or reverse proxy

3. **Database Security**:
   - Use strong passwords
   - Enable SSL connections
   - Restrict network access
   - Regular backups

4. **Network Security**:
   - Use VPC/Private networks
   - Configure security groups/firewall rules
   - Implement rate limiting
   - Use WAF for additional protection

### Performance

1. **Scaling**:
   - Horizontal scaling: Add more backend instances
   - Vertical scaling: Increase instance resources
   - Use load balancer for distribution
   - Auto-scaling based on metrics

2. **Caching**:
   - Redis for prediction caching
   - CDN for frontend assets
   - Database query caching

3. **Database Optimization**:
   - Add indexes on frequently queried columns
   - Connection pooling
   - Read replicas for read-heavy workloads
   - Regular VACUUM/ANALYZE

4. **Resource Limits**:
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Monitoring

1. **Application Monitoring**:
   - Prometheus for metrics collection
   - Grafana for visualization
   - Alert rules for critical issues
   - Uptime monitoring (UptimeRobot, Pingdom)

2. **Log Management**:
   - Centralized logging (ELK, Loki)
   - Log rotation
   - Error tracking (Sentry)
   - Audit logs

3. **Infrastructure Monitoring**:
   - Server metrics (CPU, memory, disk)
   - Network metrics
   - Database performance
   - Container health

### Backup & Recovery

1. **Database Backups**:
   - Automated daily backups
   - Off-site backup storage
   - Test restoration regularly
   - Point-in-time recovery

2. **Configuration Backups**:
   - Version control for all configs
   - Environment variable backup
   - Infrastructure as Code (Terraform)

3. **Disaster Recovery**:
   - Document recovery procedures
   - Regular DR drills
   - Multi-region deployment (optional)
   - Backup monitoring and alerting

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          # Your deployment script
```

## Monitoring & Maintenance

### Health Checks

Monitor these endpoints:
- `GET /health` - Basic health check
- `GET /metrics` - Prometheus metrics
- `GET /api/v1/monitoring/health` - Detailed system health

### Key Metrics to Monitor

1. **Application Metrics**:
   - Request rate
   - Error rate
   - Response time (p50, p95, p99)
   - Prediction accuracy

2. **Model Metrics**:
   - Drift score
   - Prediction latency
   - Model version
   - Retraining frequency

3. **Infrastructure Metrics**:
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network traffic

### Maintenance Tasks

1. **Daily**:
   - Check system health
   - Review error logs
   - Monitor drift scores

2. **Weekly**:
   - Review performance metrics
   - Check backup status
   - Update dependencies (if needed)

3. **Monthly**:
   - Security updates
   - Cost optimization review
   - Capacity planning
   - DR drill

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
```bash
# Check database is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify connection string
echo $DATABASE_URL
```

2. **High Memory Usage**:
```bash
# Check container stats
docker stats

# Increase memory limits in docker-compose.yml
```

3. **Slow Predictions**:
```bash
# Check model loading
# Consider model optimization
# Enable caching
# Scale horizontally
```

## Support

For deployment issues:
- Check logs: `docker-compose logs -f`
- Review [Troubleshooting Guide](TROUBLESHOOTING.md)
- Open an issue on GitHub

---

**Note**: Always test deployments in a staging environment before production!
