import numpy as np
import torch
from tqdm import *
from .utils import estimate_koopman_matrix, map_data
from ..processing.dataprocessing import Postprocessing_vamp

class VAMPNet_Estimator:

    def __init__(self, epsilon, mode, symmetrized):

        self._score = None
        self._score_list = []

        self._epsilon = epsilon
        self._mode = mode
        self._symmetrized = symmetrized
        self._is_fitted = False

    @property
    def loss(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        else:
            return -self._score

    @property
    def score(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        else:
            return self._score

    def fit(self, data):

        assert len(data) == 2

        koopman = estimate_koopman_matrix(data[0], data[1], epsilon=self._epsilon, mode=self._mode, symmetrized=self._symmetrized)
        self._score = torch.pow(torch.norm(koopman, p='fro'), 2) + 1
        self._is_fitted = True

        return self

    def save(self):

        with torch.no_grad():

            #self._score_list.append(self.score.cpu().numpy())
            self._score_list.append(self.score)

        return self

    def clear(self):

        self._score_list = []

        return self

    def output_mean_score(self):

        #mean_score = np.mean(np.stack(self._score_list), axis=0)
        mean_score = torch.mean(torch.stack(self._score_list))

        return mean_score

class VAMPNet_Model:
    """ The VAMPNet model from VAMPNet estimator.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network model which maps the input data to the basis functions.
    lobe_lagged : torch.nn.Module, optional, default = None
        Neural network model for timelagged data, in case of None the lobes are shared (structure and weights).
    device : torch device, default = None
        The device on which the torch modules are executed.
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, lobe_lagged=None, device=None, dtype=np.float32):

        self._lobe = lobe
        if dtype == np.float32:
            self._lobe = self._lobe.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()
        if lobe_lagged is not None:
            self._lobe_lagged = lobe_lagged
            if dtype == np.float32:
                self._lobe_lagged = self._lobe_lagged.float()
            elif dtype == np.float64:
                self._lobe_lagged = self._lobe_lagged.double()

        self._dtype = dtype
        self._device = device

    @property
    def lobe(self):
        return self._lobe

    @property
    def lobe_lagged(self):
        if self._lobe_lagged is None:
            raise ValueError('There is only one neural network for both time-instant and time-lagged data')
        return self._lobe_lagged

    def transform(self, data, instantaneous=True, return_cv=False, lag_time=None):
        """ Transform the data through the trained networks.

        Parameters
        ----------
        data : list or tuple or ndarray
            The data to be transformed.
        instantaneous : boolean, default = True
            Whether to use the instantaneous lobe or the time-lagged lobe for transformation.
            Note that only VAMPNet method requires two lobes
            
        Returns
        -------
        output : array_like
            List of numpy array or numpy array containing transformed data.
        """

        if instantaneous or self._lobe_lagged is None:
            self._lobe.eval()
            net = self._lobe
        else:
            self._lobe_lagged.eval()
            net = self._lobe_lagged

        output = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            output.append(net(data_tensor).cpu().numpy())

        if not return_cv:
            return output if len(output) > 1 else output[0]
        else:
            if lag_time is None:
                raise ValueError('Please input the lag time for transformation to CVs')
            else:
                post = Postprocessing_vamp(lag_time=lag_time, dtype=self._dtype)
                output_cv = post.fit_transform(output, instantanuous=instantaneous)
            return output_cv if len(output_cv) > 1 else output_cv[0]
    
class VAMPNet:
    """ The method used to train the VAMPnets.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network model which maps the input data to the basis functions.
    lobe_lagged : torch.nn.Module, optional, default = None
        Neural network model for timelagged data, in case of None the lobes are shared (structure and weights).
    optimizer : str, default = 'Adam'
        The type of optimizer used for training.
    device : torch.device, default = None
        The device on which the torch modules are executed.
    learning_rate : float, default = 5e-4
        The learning rate of the optimizer.
    epsilon : float, default = 1e-6
        The strength of the regularization/truncation under which matrices are inverted.
    method : str, default = 'vamp-2'
        The methods to be applied for training.
        'vamp-2': VAMP-2 score.
    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, lobe_lagged=None, optimizer='Adam', device=None, learning_rate=5e-4,
                 epsilon=1e-6, mode='regularize', symmetrized=False, dtype=np.float32, save_model_interval = None):

        self._lobe = lobe
        self._lobe_lagged = lobe_lagged
        self._device = device
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._mode = mode
        self._symmetrized = symmetrized
        self._dtype = dtype
        self._save_model_interval = save_model_interval

        if self._dtype == np.float32:
            self._lobe = self._lobe.float()
            if self._lobe_lagged is not None:
                self._lobe_lagged = self._lobe_lagged.float()
        elif self._dtype == np.float64:
            self._lobe = self._lobe.double()
            if self._lobe_lagged is not None:
                self._lobe_lagged = self._lobe_lagged.double()
        
        self._step = 0
        self._save_models = []
        self.optimizer_types = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
        if optimizer not in self.optimizer_types.keys():
            raise ValueError(f"Unknown optimizer type, supported types are {self.optimizer_types.keys()}")
        else:
            if self._lobe_lagged is None:
                self._optimizer = self.optimizer_types[optimizer](self._lobe.parameters(), lr=learning_rate)
            else:
                self._optimizer = self.optimizer_types[optimizer](
                    list(self._lobe.parameters()) + list(self._lobe_lagged.parameters()), lr=learning_rate)

        self._training_scores = []
        self._validation_scores = []

        self._estimator = VAMPNet_Estimator(epsilon=self._epsilon, mode=self._mode, symmetrized=self._symmetrized)

    @property
    def training_scores(self):
        return np.array(self._training_scores)

    @property
    def validation_scores(self):
        return np.array(self._validation_scores)

    def partial_fit(self, data):
        """ Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.

        Returns
        -------
        self : VAMPNet
        """

        batch_0, batch_1 = data[0], data[1]

        self._lobe.train()
        if self._lobe_lagged is not None:
            self._lobe_lagged.train()

        self._optimizer.zero_grad()
        x_0 = self._lobe(batch_0)
        if self._lobe_lagged is None:
            x_1 = self._lobe(batch_1)
        else:
            x_1 = self._lobe_lagged(batch_1)

        loss = self._estimator.fit([x_0, x_1]).loss

        loss.backward()
        self._optimizer.step()

        self._training_scores.append((-loss).item())
        self._step += 1

        return self

    def validate(self, val_data):
        """ Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        val_data : tuple or list of length 2, containing instantaneous and time-lagged validation data.
        
        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """

        val_batch_0, val_batch_1 = val_data[0], val_data[1]

        self._lobe.eval()
        if self._lobe_lagged is not None:
            self._lobe_lagged.eval()

        with torch.no_grad():
            val_output_0 = self._lobe(val_batch_0)
            if self._lobe_lagged is None:
                val_output_1 = self._lobe(val_batch_1)
            else:
                val_output_1 = self._lobe_lagged(val_batch_1)

            score = self._estimator.fit([val_output_0, val_output_1]).score
            self._estimator.save()

        return score

    def fit(self, train_loader, n_epochs=1, validation_loader=None, progress=tqdm):
        """ Performs fit on data.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Yield a tuple of batches representing instantaneous and time-lagged samples for training.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
             Yield a tuple of batches representing instantaneous and time-lagged samples for validation.
        progress : context manager, default=tqdm

        Returns
        -------
        self : VAMPNet
        """

        self._step = 0

        for epoch in progress(range(n_epochs), desc="epoch", total=n_epochs, leave=False):

            for batch_0, batch_1 in train_loader:

                self.partial_fit((batch_0.to(device=self._device), batch_1.to(device=self._device)))

            if validation_loader is not None:
                with torch.no_grad():
                    for val_batch_0, val_batch_1 in validation_loader:
                        self.validate((val_batch_0.to(device=self._device), val_batch_1.to(device=self._device)))

                    mean_score = self._estimator.output_mean_score()
                    self._validation_scores.append(mean_score.item())
                    self._estimator.clear()

                    print(epoch, mean_score.item())

                    if self._save_model_interval is not None:       
                        if (epoch + 1) % self._save_model_interval == 0:
                            m = self.fetch_model()
                            self._save_models.append((epoch, m))

        return self

    def transform(self, data, instantaneous=True, return_cv=False, lag_time=None):
        """ Transform the data through the trained networks.

        Parameters
        ----------
        data : list or tuple or ndarray
            The data to be transformed.
        instantaneous : boolean, default = True
            Whether to use the instantaneous lobe or the time-lagged lobe for transformation.
            Note that only VAMPNet method requires two lobes

        Returns
        -------
        output : array_like
            List of numpy array containing transformed data.
        """

        model = self.fetch_model()
        return model.transform(data, instantaneous=instantaneous, return_cv=return_cv, lag_time=lag_time)

    def fetch_model(self) -> VAMPNet_Model:
        """ Yields the current model.

        Returns
        -------
        VAMPNet_Model :
            The VAMPNet model from VAMPNet estimator.
        """

        from copy import deepcopy
        lobe = deepcopy(self._lobe)
        lobe_lagged = deepcopy(self._lobe_lagged)
        return VAMPNet_Model(lobe, lobe_lagged, device=self._device, dtype=self._dtype)