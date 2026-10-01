# Run on Kubernetes (kind or minikube)

```bash
docker build -t agentic-research:local .
kind create cluster --name research
kind load docker-image agentic-research:local --name research   # minikube: minikube image load agentic-research:local

# API keys from your .env (see .env.example); ACCESS_TOKEN is optional
kubectl create secret generic agentic-research-env --from-env-file=.env

kubectl apply -f deploy/k8s/deployment.yaml
kubectl rollout status deployment/agentic-research
kubectl port-forward svc/agentic-research 8080:80     # open http://localhost:8080
```

The pod requests 2.5 GiB because the baked-in Laya model needs it, so give Docker Desktop at least 5 GiB.
The SQLite store lives in the container and is lost on restart; mount a PersistentVolumeClaim at `/srv/data` to keep runs and share links.
