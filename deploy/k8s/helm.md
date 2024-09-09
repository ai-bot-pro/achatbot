### namespace
```shell
kubectl create namespace achatbot
```

### secret
```shell
kubectl apply -f fastapi-daily-chat-bot/secret.yaml
```



### k8s kind
```shell
#minikube image load weedge/fastapi-daily-chat-bot:latest  

kubectl apply -f fastapi-daily-chat-bot/configmap.yaml 
#kubectl delete -f fastapi-daily-chat-bot/configmap.yaml 
kubectl apply -f fastapi-daily-chat-bot/deployment.yaml
#kubectl delete -f fastapi-daily-chat-bot/deployment.yaml

kubectl get all --namespace achatbot
kubectl get pods -o wide -w --namespace achatbot
kubectl get svc -w -o wide --namespace achatbot

#kubectl logs -n achatbot -f -c fastapi-daily-chat-bot-container fastapi-daily-chat-bot-deployment-77d9d85dd-wl4rx
#kubectl logs -n achatbot -f -c fastapi-daily-chat-bot-container fastapi-daily-chat-bot-deployment-77d9d85dd-rxgjf
# SLS or E(C)LK
```


### minikube service
```shell
minikube service list
minikube service -n achatbot fastapi-daily-chat-bot-service
```

### scaling
```
kubectl scale -n achatbot --replicas 2 deployment/fastapi-daily-chat-bot-deployment
```

### helm uninstall/delete Chart
```shell
helm delete --namespace achatbot fastapi-daily-chat-bot
```


### balancing

### reference
1. https://cloud.google.com/blog/products/containers-kubernetes/your-guide-kubernetes-best-practices
2. https://learnk8s.io/kubernetes-long-lived-connections
3. https://octopus.com/blog/local-images-minikube