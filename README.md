# 🩺 Diabetes Prediction Model - MLOps Project (FastAPI + Docker + Kubernetes)

> 🎯 End-to-End MLOps Project: Model Training → API → Docker → Kubernetes Deployment

This project demonstrates how to build and deploy a Machine Learning model using a real-world use case: predicting whether a person is diabetic based on health metrics.

We cover the complete MLOps workflow:

- ✅ Model Training
- ✅ API Development with FastAPI
- ✅ Docker Containerization
- ✅ Kubernetes Deployment (Docker Desktop)
- ✅ Multi-Replica Setup
- ✅ LoadBalancer Service Exposure

---

# 📊 Problem Statement

Predict whether a person is diabetic based on:

- Pregnancies  
- Glucose  
- Blood Pressure  
- BMI  
- Age  

We use a **Random Forest Classifier** trained on the **Pima Indians Diabetes Dataset**.

---

# 🏗 Project Architecture

Browser  
   ↓  
Kubernetes LoadBalancer Service  
   ↓  
Deployment (2 Replicas)  
   ↓  
Pods  
   ↓  
Docker Containers  
   ↓  
FastAPI  
   ↓  
ML Model  

---

# 🚀 Quick Start (Local Development)

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Praveen7477/mlops_project.git
cd mlops_project
```

---

## 2️⃣ Create Virtual Environment

### Windows (PowerShell)

```bash
python -m venv .mlops
.mlops\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .mlops
source .mlops/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Train the Model

```bash
python train.py
```

This generates:

```
diabetes_model.pkl
```

---

## 5️⃣ Run the API Locally

```bash
uvicorn main:app --reload
```

Open:

```
http://localhost:8000/docs
```

---

# 🧪 Sample Input for `/predict`

```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

---

# 🐳 Dockerization

## Build Docker Image

```bash
docker build -t mlops-app .
```

## Run Container

```bash
docker run -p 8000:8000 mlops-app
```

Access:

```
http://localhost:8000
```

---

# ☸️ Kubernetes Deployment (Docker Desktop)

⚠️ Make sure Kubernetes is enabled in Docker Desktop.

---

## 1️⃣ Apply Deployment & Service

```bash
kubectl apply -f k8s-deploy.yml
```

---

## 2️⃣ Verify Pods

```bash
kubectl get pods
```

You should see:

```
2/2 Running
```

(If replicas are set to 2)

---

## 3️⃣ Verify Service

```bash
kubectl get svc
```

Since service type is `LoadBalancer` and using Docker Desktop:

Access the API at:

```
http://localhost
```

or

```
http://localhost/docs
```

---

# 🔄 Scaling the Application

Increase replicas:

```bash
kubectl scale deployment diabetes-api --replicas=5
```

Verify:

```bash
kubectl get pods
```

---

# 🧠 Key MLOps Concepts Covered

- Docker Image vs Container
- Kubernetes Deployment
- Pods & Replicas
- Service & LoadBalancer
- imagePullPolicy configuration
- ErrImagePull debugging
- YAML indentation troubleshooting
- Self-healing behavior in Kubernetes

---

# 🧹 Cleanup

Stop everything:

```bash
kubectl delete -f k8s-deploy.yml
```

Optional Docker cleanup:

```bash
docker stop $(docker ps -q)
```

---

# 🎯 Future Improvements

- CI/CD Integration (GitHub Actions)
- Cloud Deployment (AWS EKS)
- Horizontal Pod Autoscaling (HPA)
- Monitoring with Prometheus & Grafana
- Model Versioning

---


---

⭐ If you found this helpful, feel free to star the repository!
