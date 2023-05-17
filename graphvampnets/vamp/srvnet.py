import numpy as np
import torch
from tqdm import *
from .utils import estimate_c_tilde_matrix, map_data
from ..processing.dataprocessing import Postprocessing_vac

class SRVNet_Estimator:

    def __init__(self):

        self._score = None
        self._eigenvalues = None

        self._score_list = []
        self._eigenvalues_list = []

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

    @property
    def eigenvalues(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        else:
            return self._eigenvalues.flip(dims=[0])

    def fit(self, data):

        assert len(data) == 2

        c_tilde = estimate_c_tilde_matrix(data[0], data[1])
        self._eigenvalues, _ = torch.linalg.eigh(c_tilde)
        self._score = torch.sum(self._eigenvalues**2) + 1
        self._is_fitted = True

        return self

    def save(self):

        with torch.no_grad():

            #self._score_list.append(self.score.cpu().numpy())
            #self._eigenvalues_list.append(self.eigenvalues.cpu().numpy())
            self._score_list.append(self.score)
            self._eigenvalues_list.append(self.eigenvalues)

        return self

    def clear(self):

        self._score_list = []
        self._eigenvalues_list = []

        return self

    def output_mean_score(self):

        #mean_score = np.mean(np.stack(self._score_list), axis=0)
        mean_score = torch.mean(torch.stack(self._score_list))

        return mean_score

    def output_mean_eigenvalues(self):

        #mean_eigenvalues = np.mean(np.stack(self._eigenvalues_list), axis=0)
        mean_eigenvalues = torch.mean(torch.stack(self._eigenvalues_list),axis=0)

        return mean_eigenvalues
        

class SRVNet_Model:
    """ The SRVNet model from SRVNet estimator.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network model which maps the input data to the basis functions.
    device : torch device, default = None
        The device on which the torch modules are executed.
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, device=None, dtype=np.float32):

        self._lobe = lobe
        if dtype == np.float32:
            self._lobe = self._lobe.float()
        elif dtype == np.float64:
            self._lobe = self._lobe.double()

        self._dtype = dtype
        self._device = device

    @property
    def lobe(self):
        return self._lobe

    def transform(self, data, return_cv=False, lag_time=None):
        """ Transform the data through the trained networks.

        Parameters
        ----------
        data : list or tuple or ndarray
            The data to be transformed.

        return_cv : boolean, default = False
            If true, return the transformed collective variables.
            If false, return the outputs of neural networks.

        lag_time : int, default = None
            If return_cv is true, lag_time is required.
            
        Returns
        -------
        output : array_like
            List of numpy array or numpy array containing transformed data.
        """

        self._lobe.eval()
        net = self._lobe

        output = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            output.append(net(data_tensor).cpu().numpy())
        
        if not return_cv:
            return output if len(output) > 1 else output[0]
        else:
            if lag_time is None:
                raise ValueError('Please input the lag time for transformation to CVs')
            else:
                post = Postprocessing_vac(lag_time=lag_time, dtype=self._dtype)
                output_cv = post.fit_transform(output)
            return output_cv if len(output_cv) > 1 else output_cv[0]
    
class SRVNet:
    """ The method used to train the SRVnets.

    Parameters
    ----------
    lobe : torch.nn.Module
        A neural network model which maps the input data to the basis functions.
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
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, optimizer='Adam', device=None, learning_rate=5e-4,
                 epsilon=1e-6, dtype=np.float32, save_model_interval = None):

        self._lobe = lobe
        self._device = device
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._dtype = dtype
        self._save_model_interval = save_model_interval

        if self._dtype == np.float32:
            self._lobe = self._lobe.float()
        elif self._dtype == np.float64:
            self._lobe = self._lobe.double()
        
        self._step = 0
        self._save_models = []
        self.optimizer_types = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
        if optimizer not in self.optimizer_types.keys():
            raise ValueError(f"Unknown optimizer type, supported types are {self.optimizer_types.keys()}")
        else:
            self._optimizer = self.optimizer_types[optimizer](self._lobe.parameters(), lr=learning_rate)

        self._training_scores = []
        self._validation_scores = []
        self._training_eigenvalues = []
        self._validation_eigenvalues = []
        
        self._estimator = SRVNet_Estimator()

    @property
    def training_scores(self):
        return np.array(self._training_scores)

    @property
    def validation_scores(self):
        return np.array(self._validation_scores)
    
    @property
    def training_eigenvalues(self):
        return np.array(self._training_eigenvalues)
    
    @property
    def validation_eigenvalues(self):
        return np.array(self._validation_eigenvalues)

    def partial_fit(self, data):
        """ Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.

        Returns
        -------
        self : SRVNet
        """

        batch_0, batch_1 = data[0], data[1]

        self._lobe.train()

        self._optimizer.zero_grad()
        x_0 = self._lobe(batch_0)
        x_1 = self._lobe(batch_1)

        loss = self._estimator.fit([x_0, x_1]).loss

        loss.backward()
        self._optimizer.step()

        self._training_scores.append((-loss).item())
        self._training_eigenvalues.append((self._estimator.eigenvalues.detach().cpu().numpy()))
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

        with torch.no_grad():
            val_output_0 = self._lobe(val_batch_0)
            val_output_1 = self._lobe(val_batch_1)

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
        self : SRVNet
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
                    mean_eigenvalues = self._estimator.output_mean_eigenvalues()

                    self._validation_scores.append(mean_score.item())
                    self._validation_eigenvalues.append(mean_eigenvalues.cpu().numpy())

                    self._estimator.clear()

                    print(epoch, mean_score.item(), mean_eigenvalues.cpu().numpy())

                    if self._save_model_interval is not None:       
                        if (epoch + 1) % self._save_model_interval == 0:
                            m = self.fetch_model()
                            self._save_models.append((epoch, m))

        return self

    def transform(self, data, return_cv=False, lag_time=None):
        """ Transform the data through the trained networks.

        Parameters
        ----------
        data : list or tuple or ndarray
            The data to be transformed.

        return_cv : boolean, default = False
            If true, return the transformed collective variables.
            If false, return the outputs of neural networks.

        lag_time : int, default = None
            If return_cv is true, lag_time is required.

        Returns
        -------
        output : array_like
            List of numpy array containing transformed data.
        """

        model = self.fetch_model()
        return model.transform(data, return_cv=return_cv, lag_time=lag_time)

    def fetch_model(self) -> SRVNet_Model:
        """ Yields the current model.

        Returns
        -------
        SRVNet_Model :
            The SRVNet model from SRVNet estimator.
        """

        from copy import deepcopy
        lobe = deepcopy(self._lobe)
        return SRVNet_Model(lobe, device=self._device, dtype=self._dtype)