import torch

def train_vae(vae, loader, optimizer, n_epochs, out_dim, verbose=False):
    for epoch in range(n_epochs):
        for i, x in enumerate(loader):
            x = x.view(-1, out_dim)
            optimizer.zero_grad()
            x_hat, mu, log_var = vae(x)
            loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss.backward()
            optimizer.step()
            if verbose and (i+1) % 100 == 0:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')

def train_transformer(transformer, loader, optimizer, criterion, n_epochs, verbose=False):
    for epoch in range(n_epochs):
        for i, (src, trg) in enumerate(loader):
            optimizer.zero_grad()
            out = transformer(src, trg)
            out_seq = torch.round(torch.clamp(out, 1, 20))
            # print(out_seq.shape)
            out_seq = out_seq[:,:,-1:]
            out_seq = out_seq.view(out_seq.shape[0], out_seq.shape[1])
            # print(out_seq.shape, trg.shape)
            out_seq = out_seq.type(torch.FloatTensor)
            trg = trg.type(torch.FloatTensor)
            loss = criterion(out_seq, trg)
            loss.backward()
            optimizer.step()
            if verbose and (i+1) % 100 == 0:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')
                print(out_seq)

def train_graph_vae(graph_vae, data_list, optimizer, n_epochs, verbose=False):
    for epoch in range(n_epochs):
        for data in data_list:
            optimizer.zero_grad()
            a_in, e_in, f_in, a_tilde, e_tilde, f_tilde, X, mu, log_var = graph_vae(data)
            likelihood = graph_vae.likelihood(a_in, e_in, f_in, a_tilde, e_tilde, f_tilde, X)
            loss = graph_vae.loss(likelihood, mu, log_var)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'epoch {epoch+1}/{n_epochs}, loss = {loss.item():.4}')