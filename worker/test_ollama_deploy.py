from manager import WorkerManager
import os

# mock environment
os.environ["HOST_DATA_DIR"] = "/tmp/fake_host_data"

def test():
    mgr = WorkerManager()
    
    req = {
        "deploy_id": "test_deploy",
        "replica_id": "test_rep",
        "name": "test_ollama",
        "is_embedding": False,
        "model": "llama3",
        "engine": "ollama",
        "gpus": [0],
        "tp": 1,
        "max_len": 4096,
        "gpu_util": 0.9,
    }
    
    print("Testing Deploy...")
    dep = mgr.deploy_model(req)
    print("Deployment:", dep)
    
    print("Testing logs...")
    logs = mgr.get_logs("test_deploy")
    print("Logs:")
    print(logs)
    
if __name__ == "__main__":
    test()
