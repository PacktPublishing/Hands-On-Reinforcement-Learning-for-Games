import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

batch_size, inputs, hidden, outputs = 64, 1000, 100, 10

x = torch.randn(batch_size, inputs, device=device, dtype=dtype)
y = torch.randn(batch_size, outputs, device=device, dtype=dtype)

layer1 = torch.randn(inputs, hidden, device=device, dtype=dtype)
layer2 = torch.randn(hidden, outputs, device=device, dtype=dtype)


for t in range(500):    
    h = x.mm(layer1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(layer2)
    
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_layer2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(layer2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_layer1 = x.t().mm(grad_h)
    
    layer1 -= learning_rate * grad_layer1
    layer2 -= learning_rate * grad_layer2
