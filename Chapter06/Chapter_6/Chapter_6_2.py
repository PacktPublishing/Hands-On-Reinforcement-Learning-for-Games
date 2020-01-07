import torch

batch_size, inputs, hidden, outputs = 64, 1000, 100, 10

x = torch.randn(batch_size, inputs)
y = torch.randn(batch_size, outputs)

model = torch.nn.Sequential(
    torch.nn.Linear(inputs, hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, outputs),
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):    
    y_pred = model(x)
    
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    
    model.zero_grad()
    
    loss.backward()
   
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad