# Kubernetes Deployment

These manifests provide a production-style Kubernetes deployment layout for the Enterprise RAG Support Automation Platform.

## Resources

- Namespace
- ConfigMap
- Secret template
- FastAPI deployment and service
- Streamlit deployment and service
- Persistent volume claims for logs and vectorstore
- Ingress template

## Usage

Build and tag the image:

```bash
docker build -t enterprise-rag-support-platform:latest .
```

Create a real secret file from the template:

```bash
cp k8s/secret.example.yaml k8s/secret.yaml
```

Edit `k8s/secret.yaml`, then apply:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/persistent-volume-claims.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/streamlit-service.yaml
kubectl apply -f k8s/ingress.yaml
```

For local clusters such as Minikube or Kind, you may need to load the Docker image into the cluster before applying the manifests.
