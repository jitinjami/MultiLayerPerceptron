import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Interface definitions
class Layer:
    var: Dict[str, np.ndarray] = {}

    @dataclass
    class BackwardResult:
        variable_grads: Dict[str, np.ndarray]
        input_grads: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, error: np.ndarray) -> BackwardResult:
        raise NotImplementedError()


class Loss:
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError()

    def backward(self) -> np.ndarray:
        raise NotImplementedError()

# Implementation starts


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        ## Implement

        result = np.tanh(x) #Calculating the hyperbolic tan of all elements of x

        ## End
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        tanh_x = self.saved_variables["result"]

        ## Implement
        #Backpropogation of tanh activation layer = input_gradient *(1 - tanh(x)^2)
        d_x = grad_in * (1 - np.square(tanh_x))
        
        ## End
        assert d_x.shape == tanh_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, tanh_x.shape)

        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)


class Softmax(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        ## Implement
        #Softmax layer of one element = exp(x_i)/ sum of i = 1 to n exp(x_i)

        exp_sum = np.sum(np.exp(x), axis = -1, keepdims=True)
        result = np.exp(x)/exp_sum

        ## End
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        softmax = self.saved_variables["result"]

        ## Implement
        interim_grad = (grad_in * softmax).sum(axis = -1, keepdims=True)

        d_x = softmax*(grad_in - interim_grad)

        ## End
        assert d_x.shape == softmax.shape, "Input: grad shape differs: %s %s" % (d_x.shape, softmax.shape)

        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.var = {
            "W": np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size)),
            "b": np.zeros((output_size), dtype=np.float32)
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        W = self.var['W']
        b = self.var['b']

        ## Implement
        ## Save your variables needed in backward pass to self.saved_variables.

        y = np.dot(x,W) + b

        self.saved_variables = {
            "input": x
        }

        ## End
        return y

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        ## Implement
        x = self.saved_variables["input"]
        W = self.var["W"]
        x_T = np.transpose(x)
        W_T = np.transpose(W)

        dW = np.dot(x_T,grad_in)

        db = np.mean(grad_in, axis=0)

        d_inputs = np.dot(grad_in, W_T)

        ## End
        assert d_inputs.shape == x.shape, "Input: grad shape differs: %s %s" % (d_inputs.shape, x.shape)
        assert dW.shape == self.var["W"].shape, "W: grad shape differs: %s %s" % (dW.shape, self.var["W"].shape)
        assert db.shape == self.var["b"].shape, "b: grad shape differs: %s %s" % (db.shape, self.var["b"].shape)

        self.saved_variables = None
        updates = {"W": dW,
                   "b": db}
        return Layer.BackwardResult(updates, d_inputs)


class Sequential(Layer):
    class RefDict(dict):
        def __setitem__(self, k, v):
            assert k in self, "Trying to set a non-existing variable %s" % k
            ref = super().__getitem__(k)
            ref[0][ref[1]] = v

        def __getitem__(self, k):
            ref = super().__getitem__(k)
            return ref[0][ref[1]]

        def items(self) -> Tuple[str, np.ndarray]:
            for k in self.keys():
                yield k, self[k]

    def __init__(self, list_of_modules: List[Layer]):
        self.modules = list_of_modules

        refs = {}
        for i, m in enumerate(self.modules):
            refs.update({"mod_%d.%s" % (i,k): (m.var, k) for k in m.var.keys()})

        self.var = self.RefDict(refs)

    def forward(self, input: np.ndarray) -> np.ndarray:
        ## Implement
        x = input
        for layer in self.modules:
            x = layer.forward(x)
        ## End
        return x

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        variable_grads = {}

        for module_index in reversed(range(len(self.modules))):
            module = self.modules[module_index]

            ## Implement - check
            
            grads = module.backward(grad_in)

            ## End
            grad_in = grads.input_grads
            variable_grads.update({"mod_%d.%s" % (module_index, k): v for k, v in grads.variable_grads.items()})

        return Layer.BackwardResult(variable_grads, grad_in)


class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        Y = prediction
        T = target
        n = prediction.size

        ## Implement
        ## The loss function has to return a single scalar, so we have to take the mean over the batch dimension.
        ## Don't forget to save your variables needed for backward to self.saved_variables.

        mean_ce = - np.mean(T * np.log(Y))

        self.saved_variables = {"Y": Y,
                   "T": T,
                   "n": n}

        ## End
        return mean_ce

    def backward(self) -> np.ndarray:
        ## Implement
        y = self.saved_variables["Y"]
        T = self.saved_variables["T"]
        n = self.saved_variables["n"]
        
        d_prediction = -(T/(y*n))

        ## End
        assert d_prediction.shape == y.shape, "Error shape doesn't match prediction: %d %d" % \
                                              (d_prediction.shape, y.shape)

        self.saved_variables = None
        return d_prediction


def train_one_step(model: Layer, loss: Loss, learning_rate: float, input: np.ndarray, target: np.ndarray) -> float:
    ## Implement
    y = model.forward(input)

    loss_value = loss.forward(y,target)
    
    variable_gradients = model.backward(loss.backward()).variable_grads
    
    for key, value in model.var.items():
        model.var[key] = value - learning_rate * variable_gradients[key]

    ## End
    return loss_value


def create_network() -> Layer:
    ## Implement
    network = Sequential([Linear(2, 50), Tanh(), Linear(50, 30), Tanh(), Linear(30, 2), Softmax()])
    ## End
    return network


def gradient_check():
    X, T = twospirals(n_points=10)
    NN = create_network()
    eps = 0.0001

    loss = CrossEntropy()
    loss.forward(NN.forward(X), T)
    variable_gradients = NN.backward(loss.backward()).variable_grads

    all_succeeded = True

    # Check all variables. Variables will be flattened (reshape(-1)), in order to be able to generate a single index.
    for key, value in NN.var.items():
        variable = NN.var[key].reshape(-1)
        variable_gradient = variable_gradients[key].reshape(-1)
        success = True

        if NN.var[key].shape != variable_gradients[key].shape:
            print("[FAIL]: %s: Shape differs: %s %s" % (key, NN.var[key].shape, variable_gradients[key].shape))
            success = False
            break

        # Check all elements in the variable
        for index in range(variable.shape[0]):
            var_backup = variable[index]

            ## Implement

            analytic_grad = variable_gradient[index]
            # loss + eps
            variable[index] = var_backup + eps
            val_pos = loss.forward(NN.forward(X), T)
            # loss  - eps
            variable[index] = var_backup - eps
            val_neg = loss.forward(NN.forward(X), T)

            numeric_grad = (val_pos - val_neg) / (2 * eps) 

            ## End

            variable[index] = var_backup
            if abs(numeric_grad - analytic_grad) > 0.00001:
                print("[FAIL]: %s: Grad differs: numerical: %f, analytical %f" % (key, numeric_grad, analytic_grad))
                success = False
                break

        if success:
            print("[OK]: %s" % key)

        all_succeeded = all_succeeded and success

    return all_succeeded

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    np.random.seed(12345)

    plt.ion()


    def twospirals(n_points=120, noise=1.6, twist=420):
        """
         Returns a two spirals dataset.
        """
        np.random.seed(0)
        n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))
        T = np.reshape(T, (T.shape[0], 1))
        T = np.concatenate([T, 1-T], axis=1)
        return X, T


    fig, ax = plt.subplots()


    def plot_data(X, T):
        ax.scatter(X[:, 0], X[:, 1], s=40, c=T[:, 0], cmap=plt.cm.Spectral)


    def plot_boundary(model, X, targets, threshold=0.0):
        ax.clear()
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        y = model.forward(X_grid)[:, 0]
        ax.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
        plot_data(X, targets)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
        plt.show()
        plt.draw()
        plt.pause(0.001)


    def main():
        print("Checking the network")
        if not gradient_check():
            print("Failed. Not training, because your gradients are not good.")
            return
        print("Done. Training...")

        X, T = twospirals(n_points=200, noise=1.6, twist=600)
        NN = create_network()
        loss = CrossEntropy()

        learning_rate = 0.02

        for i in range(20000):
            curr_error = train_one_step(NN, loss, learning_rate, X, T)
            if i % 200 == 0:
                print("step: ", i, " cost: ", curr_error)
                plot_boundary(NN, X, T, 0.5)

        plot_boundary(NN, X, T, 0.5)
        print("Done. Close window to quit.")
        plt.ioff()
        plt.show()



    main()