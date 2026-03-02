import numpy as np
import torch
import torch.nn.functional as F

def compute_accuracy(model, loader, task_id):
    """Computes accuracy of the model on the data in loader for a specific task."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(next(model.parameters()).device)
            y = y.to(next(model.parameters()).device)
            
            out = model(x, task_id=task_id) if hasattr(model, 'num_tasks') and model.num_tasks is not None else model(x)
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += x.size(0)
    model.train()
    return correct / total

def compute_bwt(acc_matrix):
    """
    Computes Backward Transfer (BWT) from the TxT accuracy matrix.
    R_{i,j} is the test classification accuracy of the model on task j after
    observing the last sample from task i.
    BWT = 1/(T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    bwt = 0.0
    for i in range(T - 1):
        bwt += (acc_matrix[T - 1, i] - acc_matrix[i, i])
    return bwt / (T - 1)

def compute_avg_acc(acc_matrix):
    """
    Computes Average Accuracy from the TxT accuracy matrix after all tasks.
    AvgAcc = 1/T * sum_{i=1}^{T} R_{T,i}
    """
    T = acc_matrix.shape[0]
    return np.mean(acc_matrix[T - 1, :])

def js_divergence(p, q):
    """
    Jensen-Shannon Divergence.
    p, q: numpy arrays of probabilities
    """
    m = 0.5 * (p + q)
    # Using scipy.stats.entropy which computes KL divergence
    from scipy.stats import entropy
    return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

def kl_divergence(p, q):
    """
    KL Divergence KL(p || q).
    p, q: numpy arrays of probabilities
    """
    from scipy.stats import entropy
    return entropy(p, q, base=2)

def compute_drift_metrics(gem_model, current_batch_y, num_classes):
    """
    Computes KL and JS divergence between the empirical class distribution 
    of the current batch and the episodic memory buffer.
    """
    # 1. Stream distribution (current batch)
    stream_counts = torch.bincount(current_batch_y, minlength=num_classes).float()
    p_stream = (stream_counts / stream_counts.sum()).cpu().numpy()
    
    # Add a small epsilon to avoid div by zero or log zero
    eps = 1e-8
    p_stream = p_stream + eps
    p_stream = p_stream / p_stream.sum()
    
    # 2. Buffer distribution
    if not gem_model.memory_y:
        # Buffer is empty, return 0 divergence
        return 0.0, 0.0
        
    # Concatenate all labels in memory
    all_y = torch.cat(gem_model.memory_y)
    buffer_counts = torch.bincount(all_y, minlength=num_classes).float()
    p_buffer = (buffer_counts / buffer_counts.sum()).cpu().numpy()
    
    p_buffer = p_buffer + eps
    p_buffer = p_buffer / p_buffer.sum()
    
    # Calculate KL(Buffer || Stream) according to requirement: "KL(buffer || stream)"
    kl = kl_divergence(p_buffer, p_stream)
    js = js_divergence(p_buffer, p_stream)
    
    return kl, js
