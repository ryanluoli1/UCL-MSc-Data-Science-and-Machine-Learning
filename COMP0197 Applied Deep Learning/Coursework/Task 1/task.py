import time

import torch
from torch import nn, optim
from torch.utils import data

def polynomial_fun(w, x):
    """
    Map the scalar input into its polynomial form.
    Input:
        w   | a vector of polynomial weights, shape = (M+1,)
        x   | input value(s), shape = (n,)
    Output:
        y   | mapped value(s), shape = (n,)
    """
    x_pow = torch.vander(x, w.shape[0], increasing=True)
    y = x_pow @ w
    return y.squeeze()

def fit_polynomial_ls(x, t, M):
    """
    Computes the optimal weights in least square sense.
    Input:
        x       | input values, shape = (N,)
        t       | target values, shape = (N,)
        M       | polynomial degree, int
    Output:
        w_hat   | optimal weights, shape = (M+1,)
    """
    x_pow = torch.vander(x, M+1, increasing=True)
    w_hat = torch.linalg.lstsq(x_pow, t.unsqueeze(1)).solution
    return w_hat

def fit_polynomial_sgd(x, t, M, learning_rate, batch_size, report_loss=True):
    """
    Computes the optimal weights with stochastic gradient descent.
    Input:
        x               | input values, shape = (N,)
        t               | target values, shape = (N,)
        M               | polynomial degree, int
        learning_rate   | learning rate of the algorithm, float
        batch_size      | batch size of the minibatches, int
    Output:
        w_hat           | optimal weights, shape = (M+1,)
    """
    x_pow = torch.vander(x, M+1, increasing=True)
    training_data = data.TensorDataset(x_pow, t)
    train_iter = data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    model = nn.Sequential(nn.Linear(M+1, 1, bias=False))
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(1000):   # setting num_epochs = 1000
        loss_i = 0
        for x_batch, t_batch in train_iter:
            optimizer.zero_grad()
            preds = model(x_batch)
            t_batch = torch.reshape(t_batch, (-1, 1))
            l = loss(preds, t_batch)
            loss_i += l
            l.backward()
            optimizer.step()
        loss_i = loss_i / x.shape[0]
        if (epoch+1) % 100 == 0 and report_loss:
            print('Epoch {}, Training Loss {}'.format(epoch+1, loss_i))
    w_hat = model[0].weight.data
    w_hat = torch.reshape(w_hat, (-1, 1))
    return w_hat

def RMSE(y_pred, y_true):
    rmse = torch.sqrt(torch.mean(torch.square(y_pred-y_true)))
    return rmse


if __name__ == "__main__":

    torch.manual_seed(0)

    # generate (normalized) training and test for least square
    w = torch.tensor([1,2,3,4,5], dtype=torch.float).reshape(-1,1)
    x_train = torch.empty((100,)).uniform_(-20, 20)
    x_train = (x_train + 20) / 40
    y_train = polynomial_fun(w, x_train)
    t_train = y_train + torch.normal(0, 0.2, size=(100,))
    x_test = torch.empty((50,)).uniform_(-20, 20)
    x_test = (x_test + 20) / 40
    y_test = polynomial_fun(w, x_test)
    t_test = y_test + torch.normal(0, 0.2, size=(50,))

    # implement least square
    time_start = time.time()
    w_hat_ls = fit_polynomial_ls(x_train, t_train, 5)
    time_end = time.time()
    time_ls = time_end - time_start
    y_hat_ls_train = polynomial_fun(w_hat_ls, x_train)
    y_hat_ls_test = polynomial_fun(w_hat_ls, x_test)
    print("Observed training data vs True polynomial curve: {:.4f} ± {:.4f}".format(torch.mean(t_train-y_train), torch.std(t_train-y_train)))
    print("LS-predicted training values vs True polynomial curve: {:.4f} ± {:.4f}".format(torch.mean(y_hat_ls_train-y_train), torch.std(y_hat_ls_train-y_train)))

    # implement stochastic gradient descent with M = 5
    time_start = time.time()
    w_hat_sgd = fit_polynomial_sgd(x_train, t_train, 5, 0.001, 20)
    time_end = time.time()
    time_sgd = time_end - time_start
    y_hat_sgd_train = polynomial_fun(w_hat_sgd, x_train)
    y_hat_sgd_test = polynomial_fun(w_hat_sgd, x_test)
    print("SGD-predicted training values vs True polynomial curve: {:.4f} ± {:.4f}".format(torch.mean(y_hat_sgd_train-y_train), torch.std(y_hat_sgd_train-y_train)))

    # compare accuracy using RMSE
    w = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.float).reshape(-1, 1)
    print("Least Square: RMSE_w = {:.4f} and RMSE_y = {:.4f}".format(RMSE(y_hat_ls_test,y_test), RMSE(w_hat_ls,w)))
    print("Stochastic Gradient Descent: RMSE_w = {:.4f} and RMSE_y = {:.4f}".format(RMSE(y_hat_sgd_test, y_test), RMSE(w_hat_sgd, w)))

    # report training time in seconds
    print('Training with Least Square took: {:.4f}s.'.format(time_ls))
    print('Training with Stochastic Gradient Descent took: {:.4f}s.'.format(time_sgd))




