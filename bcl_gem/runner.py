import argparse
import sys
import numpy as np
import torch
import json
import logging
from benchmarks import set_seed, get_split_mnist, get_split_cifar10, get_permuted_mnist
from model import MLP, SmallCNN
from gem import GEM
from metrics import compute_accuracy, compute_bwt, compute_avg_acc, compute_drift_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_and_eval(benchmark_name, seed=0, epochs=1):
    logging.info(f"Starting benchmark: {benchmark_name} with seed {seed}")
    set_seed(seed)
    
    # 1. Get loaders
    if benchmark_name == "split_mnist":
        tasks_train, tasks_test, num_classes_per_task = get_split_mnist(batch_size=10)
        model = MLP(28*28, 256, num_classes_per_task, num_tasks=len(tasks_train))
        n_tasks = 5
    elif benchmark_name == "split_cifar10":
        tasks_train, tasks_test, num_classes_per_task = get_split_cifar10(batch_size=10)
        model = SmallCNN(num_classes_per_task, num_tasks=len(tasks_train))
        n_tasks = 5
    elif benchmark_name == "permuted_mnist":
        tasks_train, tasks_test, num_classes_per_task = get_permuted_mnist(batch_size=10, n_tasks=20, seed=seed)
        model = MLP(28*28, 256, num_classes_per_task)
        n_tasks = 20
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialization for GEM
    # memory size 256 per task as required. 
    gem = GEM(model, n_tasks=n_tasks, memory_size=256, lr=0.1, margin=0.5)
    
    metrics = {
        "acc_matrix": np.zeros((n_tasks, n_tasks)),
        "kl_drift": [],
        "js_drift": []
    }
    
    batches_seen = 0
    
    for t in range(n_tasks):
        logging.info(f"=== Training Task {t+1}/{n_tasks} ===")
        loader_train = tasks_train[t]
        
        for ep in range(epochs):
            for idx, (x, y) in enumerate(loader_train):
                x = x.to(device)
                y = y.to(device)
                
                gem.observe(x, y, t)
                batches_seen += 1
                
                # Distribution drift logging every 100 batches
                if batches_seen % 100 == 0:
                    kl, js = compute_drift_metrics(gem, y, num_classes_per_task)
                    metrics["kl_drift"].append({"batch": batches_seen, "task": t, "kl": float(kl)})
                    metrics["js_drift"].append({"batch": batches_seen, "task": t, "js": float(js)})
                    
        # Populate episodic memory for the completed task
        gem.end_task(loader_train, t)
        
        # Evaluate backward across ALL tasks seen so far
        logging.info(f"Evaluating after task {t+1}...")
        for eval_t in range(t + 1):
            acc = compute_accuracy(model, tasks_test[eval_t], eval_t)
            metrics["acc_matrix"][t, eval_t] = acc
            logging.info(f"  Accuracy on task {eval_t+1}: {acc*100:.2f}%")
            
    # Final metrics
    bwt = compute_bwt(metrics["acc_matrix"])
    avg_acc = compute_avg_acc(metrics["acc_matrix"])
    
    logging.info(f"=== Final Metrics ===")
    logging.info(f"BWT: {bwt*100:.2f}%")
    logging.info(f"Average Accuracy: {avg_acc*100:.2f}%")
    
    # KILL CONDITION checks
    if benchmark_name == "split_mnist":
        # GEM typically retains perfect memory on split_mnist (BWT approx 0)
        # Assuming typical naive fine tuning loses much, GEM retains it.
        # Literature typically gives near 0% BWT for split MNIST GEM.
        if abs(bwt*100) > 5.0:
            logging.error(f"KILL CONDITION MET: Split-MNIST BWT deviates >5pp. Got {bwt*100:.2f}%. Stopping pipeline.")
            sys.exit(1)
            
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, choices=["split_mnist", "split_cifar10", "permuted_mnist", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    if args.benchmark == "all":
        benchmarks = ["split_mnist", "split_cifar10", "permuted_mnist"]
    else:
        benchmarks = [args.benchmark]
        
    all_results = {}
    for bench in benchmarks:
        metrics = train_and_eval(bench, seed=args.seed, epochs=args.epochs)
        
        # Convert acc matrix to list for json serialization
        metrics["acc_matrix"] = metrics["acc_matrix"].tolist()
        all_results[bench] = metrics
        
        # Save structural checkpoint
        with open(f"bcl_gem_results_{bench}.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
    with open("bcl_gem_results_all.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
