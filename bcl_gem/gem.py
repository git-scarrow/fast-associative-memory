import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import quadprog

def store_grad(pp, grads, grad_dims, tid):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    tid: task id
    """
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
    pp: parameters
    newgrad: corrected gradient
    grad_dims: list storing number of parameters per layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
    Solves the GEM dual QP:
    min_{v} 1/2 v^T G G^T v + g^T G^T v
    s.t. v >= 0
    
    where G is `memories` and g is `gradient`.
    gradient: 1D tensor of gradients from current task
    memories: 2D tensor of past task gradients (params x tasks)
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    
    t = memories_np.shape[0]
    
    # P = G G^T
    P = np.dot(memories_np, memories_np.transpose())
    
    # make P positive definite by adding small eps
    P = P + margin * np.eye(t) 
    
    # q = G g
    q = np.dot(memories_np, gradient_np) * -1
    
    # G in quadprog problem is A^T in scipy notation
    # v >= 0 -> I v >= 0
    G = np.eye(t)
    h = np.zeros(t)
    
    try:
        # quadprog signature: solve_qp(G, a, C, b, meq=0)
        # min 1/2 x^T G x - a^T x s.t. C^T x >= b
        # so Our P = G, Our q = -a -> a = -q, Our G = C^T -> C = G^T = G, Our h = b
        a = q # Because quadprog min 1/2 x^T G x - a^T x
        C = G # C is transposed inside quadprog vs our formulation: C^T x >= b
        b = h
        v = quadprog.solve_qp(P, a, C, b)[0]
    except ValueError as e:
        print("ValueError in quadprog:", e)
        v = np.zeros(t)

    x = np.dot(v, memories_np) + gradient_np
    
    return torch.Tensor(x).view(-1)


class GEM:
    def __init__(self, model, n_tasks, memory_size=256, lr=0.1, margin=0.5):
        self.model = model
        self.n_tasks = n_tasks
        self.memory_size = memory_size
        self.margin = margin
        self.device = next(model.parameters()).device
        
        self.opt = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Buffer
        self.memory_x = []
        self.memory_y = []
        
        # Allocate matrices for gradients
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
            
        self.grads = torch.zeros(sum(self.grad_dims), self.n_tasks).to(self.device).float()
        
    def observe(self, x, y, task_id):
        self.opt.zero_grad()
        
        # 1. Compute gradients of current batch
        out = self.model(x, task_id=task_id) if hasattr(self.model, 'num_tasks') and self.model.num_tasks is not None else self.model(x)
        loss = self.criterion(out, y)
        loss.backward()
        
        # 2. If past tasks exist, compute their gradients and check for violation
        if task_id > 0:
            # Store current gradient temporarily
            store_grad(self.model.parameters, self.grads, self.grad_dims, task_id)
            current_grad = self.grads[:, task_id].clone()
            
            # Compute past task gradients
            for t in range(task_id):
                self.opt.zero_grad()
                
                # Forward past data
                ptx = self.memory_x[t]
                pty = self.memory_y[t]
                pout = self.model(ptx, task_id=t) if hasattr(self.model, 'num_tasks') and self.model.num_tasks is not None else self.model(ptx)
                ploss = self.criterion(pout, pty)
                ploss.backward()
                
                # Store past task gradient
                store_grad(self.model.parameters, self.grads, self.grad_dims, t)
                
            # Check violation: dot product of current gradient with past gradients
            # G: (num_params, task_id)
            G = self.grads[:, :task_id]
            # g: (num_params)
            g = current_grad
            
            dotp = (G.t() @ g)
            if (dotp < 0).any():
                # Violation: project gradient
                projected_g = project2cone2(g, G, margin=self.margin)
                # Overwrite gradient with projected gradient
                overwrite_grad(self.model.parameters, projected_g.to(self.device), self.grad_dims)
            else:
                # No violation: restore current gradient
                overwrite_grad(self.model.parameters, current_grad, self.grad_dims)
        
        # 3. Take optimization step
        self.opt.step()
        
        # 4. Update memory (reservoir sampling simple version since memory_size matches initial batch size logic, 
        #    but standard GEM typically selects randomly or takes last N. We take random N elements uniformly).
        # We will do this at the end of the epoch, or we just maintain a running buffer.
        # It's typical for GEM to just take a fixed set at the end of the task, 
        # or randomly sample throughout.
        # We will implement an `end_task` method to populate the buffer properly.
    
    def end_task(self, loader, task_id):
        """Populate the memory buffer for task_id using the first memory_size examples from the loader."""
        samples_x = []
        samples_y = []
        n_samples = 0
        
        for x, y in loader:
            samples_x.append(x)
            samples_y.append(y)
            n_samples += x.size(0)
            if n_samples >= self.memory_size:
                break
                
        all_x = torch.cat(samples_x)[:self.memory_size].to(self.device)
        all_y = torch.cat(samples_y)[:self.memory_size].to(self.device)
        
        if len(self.memory_x) <= task_id:
            self.memory_x.append(all_x)
            self.memory_y.append(all_y)
        else:
            self.memory_x[task_id] = all_x
            self.memory_y[task_id] = all_y
