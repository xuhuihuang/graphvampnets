import functools
import numpy as np
from ..utils import rao_blackwell_ledoit_wolf

class Preprocessing:
    """ Preprocess the original trajectories to create datasets for training.

    Parameters
    ----------
    dtype : dtype, default = np.float32
    """

    def __init__(self, dtype=np.float32):

        self._dtype = dtype

    def _seq_trajs(self, data):

        data = data.copy()
        if not isinstance(data, list):
            data = [data]
        for i in range(len(data)):
            data[i] = data[i].astype(self._dtype)
        
        return data

    def transform2pw(self, data):
        """ Transform xyz coordinates data to pairwise distances data.

        Parameters
        ----------
        data : list or ndarray
            xyz coordinates data, shape of each traj [num_frames,num_atoms,3].

        Returns
        -------
        pw_data : list or ndarray
            Pairwise distances data.
        """

        data = self._seq_trajs(data)

        if not (len(data[0].shape) == 3 and data[0].shape[-1] == 3):
            raise ValueError('Please make sure the shape of each traj is [num_frames,num_atoms,3]')
        
        num_trajs = len(data)
        num_atoms = data[0].shape[1]

        pw_data = []
        for traj in range(num_trajs):
            tmp = []
            for i in range(num_atoms-1):
                for j in range(i+1, num_atoms):
                    dist = np.sqrt(np.sum((data[traj][:,i,:] - data[traj][:,j,:])**2, axis=-1))
                    tmp.append(dist)
            pw_data.append(np.stack(tmp,axis=1))
        
        return pw_data if num_trajs > 1 else pw_data[0]
    
    def create_dataset(self, data, lag_time):
        """ Create the dataset as the input to VAMPnets.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        lag_time : int
            The lag_time used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has two elements: one is the instantaneous data frame, the other is the corresponding time-lagged data frame.
        """

        data = self._seq_trajs(data)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time
            for i in range(L_re):
                dataset.append((data[k][i,:], data[k][i+lag_time,:]))

        return dataset    
    
    def _default_graph_fn(self, pw_data_frame, num_atoms, num_nbrs=1, pw_indices=None):
        """ This method is a default function to create the neighbor list of a graph.
            The neighbor list is created by considering a certain number of neighbors for each particle.

        Parameters
        ----------
        pw_data_frame : ndarray
            shape : [num_pw_distances,]
        
        num_atoms : int
            Number of particles in the system.

        num_nbrs : int

        pw_indices : ndarray
            shape : [num_pw_distances,2]

        Returns
        -------
        graph : ndarray
            shape : [num_edges,3]
            The first two columns are edge indices, and the last comlumn is distance.
        """

        if pw_indices is None:
            pw_indices = []
            for i in range(num_atoms-1):
                for j in range(i+1, num_atoms):
                    pw_indices.append(np.array([i,j]))
            pw_indices = np.array(pw_indices)

        assert len(pw_indices) == len(pw_data_frame)
        assert int(np.min(pw_indices)) == 0
        assert int(np.max(pw_indices) + 1) == num_atoms
        assert int(num_atoms*(num_atoms-1)/2) == len(pw_indices)

        pw_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(len(pw_data_frame)):
            pw_matrix[pw_indices[i,0],pw_indices[i,1]] = pw_data_frame[i]
        pw_matrix = pw_matrix + pw_matrix.T

        tmp_dist = []
        tmp_list = []

        for i in range(num_atoms):
            idx = np.argsort(pw_matrix[i,:],axis=-1)
            tmp_dist.extend(pw_matrix[i,idx[1:num_nbrs+1]].tolist()) 
            tmp_list.extend([[i, ni] for ni in idx[1:num_nbrs+1]])

        tmp_dist = np.array(tmp_dist).reshape((-1,1)).astype(self._dtype)
        tmp_list = np.array(tmp_list).reshape((-1,2)).astype(self._dtype)

        graph = np.concatenate([tmp_list,tmp_dist],axis=-1)

        return graph
    
    def transform2graph(self, data, num_atoms, graph_fn=None, **kwargs):
        """ This method is used to create graph data.

        Parameters
        ----------
        data : ndarray
            shape : [num_pw_distances,]

        pw_indices : ndarray
            shape : [num_pw_distances,2]

        num_nbrs : int

        Returns
        -------
        graph : ndarray
            shape : [num_edges,3]
            The first two columns are edge indices, and the last comlumn is distance.
        """

        data = self._seq_trajs(data)
        num_trajs = len(data)

        if not graph_fn is None:
            graph_fn = functools.partial(graph_fn)
        else:
            graph_fn = functools.partial(self._default_graph_fn, num_atoms=num_atoms)

        graph_data_for_training = []
        graph_data_for_projection = []

        for traj in range(num_trajs):
            graph_traj_for_training = []
            graph_traj_for_projection = []
            for frame in range(data[traj].shape[0]):
                graph = graph_fn(data[traj][frame], **kwargs)
                graph_traj_for_training.append(graph)
                graph_traj_for_projection.append(graph+np.array([num_atoms*frame,num_atoms*frame,0]).reshape(1,-1).repeat(len(graph),axis=0))
            graph_data_for_training.append(np.array(graph_traj_for_training))
            graph_traj_for_projection = np.vstack((np.concatenate(graph_traj_for_projection), np.array([-1,-1,data[traj].shape[0]*num_atoms]).reshape(1,-1)))
            graph_data_for_projection.append(graph_traj_for_projection)
        
        return graph_data_for_training if num_trajs > 1 else graph_data_for_training[0], graph_data_for_projection if num_trajs > 1 else graph_data_for_projection[0]
    
class Postprocessing_vac(Preprocessing):
    """ Transform the outputs from neural networks to slow CVs.
        Note that this method force the detailed balance constraint,
        which can be used to process the simulation data with sufficient sampling.

    Parameters
    ----------
    lag_time : int
        The lag time used for transformation.

    dtype : dtype, default = np.float32

    shrinkage : boolean, default = True
        To tell whether to do the shrinkaged estimation of covariance matrix. 
    
    n_dims : int, default = None
        The number of slow collective variables to obtain.
    """
    
    def __init__(self, lag_time=1, dtype=np.float32, shrinkage=True, n_dims=None):
        super().__init__(dtype)
        self._n_dims = n_dims
        self._lag_time = lag_time
        self._dtype = dtype
        self._shrinkage = shrinkage

        self._is_fitted = False
        self._mean = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._time_scales = None

    @property
    def shrinkage(self):
        return self._shrinkage

    @shrinkage.setter
    def shrinkage(self, value: bool):
        self._shrinkage = value

    @property
    def lag_time(self):
        return self._lag_time
    
    @lag_time.setter
    def lag_time(self, value: int):
        self._lag_time = value

    @property
    def mean(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._mean

    @property
    def eigenvalues(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._eigenvectors
    
    @property
    def time_scales(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._time_scales

    def fit(self, data):
        """ Fit the model for transformation.

        Parameters
        ----------
        data : list or ndarray
        """
        
        self._mean = self._cal_mean(data)
        self._eigenvalues, self._eigenvectors = self._cal_eigvals_eigvecs(data)
        self._time_scales = -self._lag_time / np.log(np.abs(self._eigenvalues))
        self._is_fitted = True
        
        return self

    def _cal_mean(self, data):

        dataset = self.create_dataset(data, self._lag_time)
        d0, d1 = map(np.array, zip(*dataset))
        mean = (d0.mean(0) + d1.mean(0)) / 2.

        return mean

    def _cal_cov_matrices(self, data):

        num_trajs = 1 if not isinstance(data, list) else len(data)
        dataset = self.create_dataset(data, self._lag_time)

        batch_size = len(dataset)
        d0, d1 = map(np.array, zip(*dataset))

        mean = 0.5 * (d0.mean(0) + d1.mean(0))

        d0_rm = d0 - mean
        d1_rm = d1 - mean

        c00 = 1. / batch_size * np.dot(d0_rm.T, d0_rm)
        c11 = 1. / batch_size * np.dot(d1_rm.T, d1_rm)
        c01 = 1. / batch_size * np.dot(d0_rm.T, d1_rm)
        c10 = 1. / batch_size * np.dot(d1_rm.T, d0_rm)

        c0 = 0.5 * (c00 + c11)
        c1 = 0.5 * (c01 + c10)

        if self.shrinkage:
            n_observations_ = batch_size + self._lag_time * num_trajs
            c0, _ = rao_blackwell_ledoit_wolf(c0, n_observations_)

        return c0, c1

    def _cal_eigvals_eigvecs(self, data):

        c0, c1 = self._cal_cov_matrices(data)

        import scipy.linalg
        eigvals, eigvecs = scipy.linalg.eigh(c1, b=c0)

        idx = np.argsort(eigvals)[::-1]

        if self._n_dims is not None:
            assert self._n_dims <= len(idx)
            idx = idx[:self._n_dims]

        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return eigvals, eigvecs

    def transform(self, data):
        """ Transfrom the basis funtions (or outputs of neural networks) to the slow CVs.
            Note that the model must be fitted first before transformation.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = []

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data = self._seq_trajs(data)
        num_trajs = len(data)

        for i in range(num_trajs):
            x_rm = data[i] - self._mean
            modes.append(np.dot(x_rm, self._eigenvectors).astype(self._dtype))

        return modes if num_trajs > 1 else modes[0]
    
    def fit_transform(self, data):
        """ Fit the model and transfrom to the slow CVs.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = self.fit(data).transform(data)

        return modes

    def gmrq(self, data):
        """ Score the model based on generalized matrix Rayleigh quotient.
            Note that the model should be fitted before computing computing GMRQ score.

        Parameters
        ----------
        data : list or ndarray, optional, default = None

        Returns
        -------
        score : float
            Generalized matrix Rayleigh quotient. This number indicates how
            well the top ``n_timescales+1`` eigenvectors of this tICA model perform
            as slowly decorrelating collective variables for the new data in
            ``sequences``.
        """

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        A = self._eigenvectors
        ### Q: use the mean of fitted data or input data?
        S, C = self._cal_cov_matrices(data)

        P = A.T.dot(C).dot(A)
        Q = A.T.dot(S).dot(A)

        score = np.trace(P.dot(np.linalg.inv(Q)))

        return score

    def empirical_correlation(self, data):
        """ Score the model based on empirical correlations between the instantaneous and time-lagged slowest CVs.
            Note that the model should be fitted before computing empirical correlations.
            The empirical correlations equal to eigenvalues for the fitted equilibrium dataset.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        corr : ndarray
            Empirical correlations between the slowest instantaneous and time-lagged CVs.
        """

        modes = self.transform(data)

        dataset = self.create_dataset(modes, self._lag_time)
        modes0, modes1 = map(np.array, zip(*dataset))

        modes0_rm = modes0 - np.mean(modes0, axis=0)
        modes1_rm = modes1 - np.mean(modes1, axis=0)

        corr = np.mean(modes0_rm * modes1_rm, axis=0) / (
                np.std(modes0_rm, axis=0) * np.std(modes1_rm, axis=0))

        return corr
    
class Postprocessing_vamp(Preprocessing):
    """ Transform the outputs from neural networks to slow CVs.
        Note that this method doesn't force the detailed balance constraint,
        which can be used to process the simulation data with insufficient sampling.

    Parameters
    ----------
    lag_time : int
        The lag time used for transformation.

    dtype : dtype, default = np.float32
    
    n_dims : int, default = None
        The number of slow collective variables to obtain.
    """

    def __init__(self, lag_time=1, dtype=np.float32, n_dims=None):
        super().__init__(dtype)
        self._n_dims = n_dims
        self._lag_time = lag_time
        self._dtype = dtype

        self._is_fitted = False
        self._mean_0 = None
        self._mean_t = None
        self._singularvalues = None
        self._left_singularvectors = None
        self._right_singularvectors = None
        self._time_scales = None

    @property
    def lag_time(self):
        return self._lag_time
    
    @lag_time.setter
    def lag_time(self, value: int):
        self._lag_time = value

    @property
    def mean_0(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._mean_0
    
    @property
    def mean_t(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._mean_t
    
    @property
    def singularvalues(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._singularvalues
    
    @property
    def left_singularvectors(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._left_singularvectors
    
    @property
    def right_singularvectors(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._right_singularvectors
    
    @property
    def time_scales(self):
        if not self._is_fitted:
            raise ValueError('please fit the model first')
        return self._time_scales
    
    def fit(self, data):
        """ Fit the model for transformation.

        Parameters
        ----------
        data : list or ndarray
        """

        self._mean_0, self._mean_t = self._cal_mean(data)
        self._singularvalues, self._left_singularvectors, self._right_singularvectors = self._cal_singularvals_singularvecs(data)
        self._time_scales = -self._lag_time / np.log(np.abs(self._singularvalues))

        self._is_fitted = True

        return self

    def _inv_sqrt(self, cov_matrix):

        import numpy.linalg
        cov_matrix = 0.5*(cov_matrix + cov_matrix.T)

        eigvals, eigvecs = numpy.linalg.eigh(cov_matrix)
        sort_key = np.abs(eigvals)
        idx = np.argsort(sort_key)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        diag = np.diag(1.0 / np.maximum(np.sqrt(np.maximum(eigvals, 1e-12)), 1e-12))
        inv_sqrt = np.dot(eigvecs, diag)

        return inv_sqrt

    def _cal_mean(self, data):

        dataset = self.create_dataset(data, self._lag_time)
        d0, d1 = map(np.array, zip(*dataset))

        return d0.mean(0), d1.mean(0)
    
    def _cal_cov_matrices(self, data):

        dataset = self.create_dataset(data, self._lag_time)

        batch_size = len(dataset)
        d0, d1 = map(np.array, zip(*dataset))

        d0_rm = d0 - d0.mean(0)
        d1_rm = d1 - d1.mean(0)

        c00 = 1. / batch_size * np.dot(d0_rm.T, d0_rm)
        c11 = 1. / batch_size * np.dot(d1_rm.T, d1_rm)
        c01 = 1. / batch_size * np.dot(d0_rm.T, d1_rm)

        return c00, c01, c11
    
    def _cal_singularvals_singularvecs(self, data):

        c00, c01, c11 = self._cal_cov_matrices(data)

        c00_inv_sqrt = self._inv_sqrt(c00)
        c11_inv_sqrt = self._inv_sqrt(c11)

        ks = np.dot(c00_inv_sqrt.T,c01).dot(c11_inv_sqrt)

        import scipy.linalg
        U, s, Vh = scipy.linalg.svd(ks, compute_uv=True, lapack_driver='gesvd')

        left = np.dot(c00_inv_sqrt,U)
        right = np.dot(c11_inv_sqrt,Vh.T)

        idx = np.argsort(s)[::-1]

        if self._n_dims is not None:
            assert self._n_dims <= len(idx)
            idx = idx[:self._n_dims]

        s = s[idx]
        left = left[:, idx]
        right = right[:, idx]

        return s, left, right
    
    def transform(self, data, instantaneous=True):
        """ Transfrom the basis funtions (or outputs of neural networks) to the slow CVs.
            Note that the model must be fitted first before transformation.

        Parameters
        ----------
        data : list or ndarray

        instantaneous : boolean, default = True
            If true, projected onto left singular functions of Koopman operator.
            If false, projected onto right singular functions of Koopman operator.

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., slow dynamic modes).
        """

        modes = []

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data = self._seq_trajs(data)
        num_trajs = len(data)

        if instantaneous:
            for i in range(num_trajs):
                x_rm = data[i] - self._mean_0
                modes.append(np.dot(x_rm, self._left_singularvectors).astype(self._dtype))
        else:
            for i in range(num_trajs):
                x_rm = data[i] - self._mean_t
                modes.append(np.dot(x_rm, self._right_singularvectors).astype(self._dtype))

        return modes if num_trajs > 1 else modes[0]

    def fit_transform(self, data, instantanuous=True):
        """ Fit the model and transfrom to the slow CVs.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = self.fit(data).transform(data, instantaneous=instantanuous)

        return modes
    