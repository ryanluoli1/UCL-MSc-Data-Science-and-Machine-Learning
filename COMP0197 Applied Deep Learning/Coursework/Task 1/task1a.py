from task import *

def generalized_fit_polynomial_sgd(x, t, max_M, learning_rate, batch_size):
    """
    Computes the optimal weights with stochastic gradient descent.
    Input:
        x               | input values, shape = (N,)
        t               | target values, shape = (N,)
        max_M           | maximum polynomial degree, int
        learning_rate   | learning rate of the algorithm, float
        batch_size      | batch size of the minibatches, int
    Output:
        M               | optimal polynomial degree, int
        w_hat           | optimal weights, shape = (M+1,)
    """
    best_M, best_w = 0, None
    best_rmse = float('inf')
    for M in range(1, max_M+1):
        w_hat = fit_polynomial_sgd(x, t, M, learning_rate, batch_size, report_loss=False)
        y_hat = polynomial_fun(w_hat, x)
        rmse = RMSE(y_hat, t)
        print("M: {}, RMSE: {:.4f}".format(M, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_M = M
            best_w = w_hat
    return best_M, best_w


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

    # implement generalized model
    best_M, best_w = generalized_fit_polynomial_sgd(x_train, t_train, 10, 0.001, 30)
    y_hat_train = polynomial_fun(best_w, x_train)
    y_hat_test = polynomial_fun(best_w, x_test)
    print("Optimal M: ", best_M)
    print("Predicted training values vs True polynomial curve: {:.4f} ± {:.4f}".format(torch.mean(y_hat_train-y_train), torch.std(y_hat_train-y_train)))
    print("Predicted test values vs True polynomial curve: {:.4f} ± {:.4f}".format(torch.mean(y_hat_test-y_test),torch.std(y_hat_test-y_test)))