import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class Meter(object):
	def __init__(self, mean=None, std=None):
		"""
		Initialize the Meter object.

		Parameters:
		- mean (torch.Tensor, optional): The mean value used for normalization. Default is None.
		- std (torch.Tensor, optional): The standard deviation value used for normalization. Default is None.
		"""
		self.mask = []
		self.y_pred = []
		self.y_true = []
		if (mean is not None) and (std is not None):
			self.mean = mean.cpu()
			self.std = std.cpu()
		else:
			self.mean = None
			self.std = None

	def update(self, y_pred, y_true, mask=None):
		"""
		Update the Meter object with predicted and true values.

		Parameters:
		- y_pred (torch.Tensor): The predicted values.
		- y_true (torch.Tensor): The true values.
		- mask (torch.Tensor, optional): The mask indicating which values to consider. Default is None.
		"""
		self.y_pred.append(y_pred.detach().cpu())
		self.y_true.append(y_true.detach().cpu())
		if mask is None:
			self.mask.append(th.ones(self.y_pred[-1].shape))
		else:
			self.mask.append(mask.detach().cpu())

	def _finalize(self):
		"""
		Finalize the Meter object by concatenating the stored values.

		Returns:
		- mask (torch.Tensor): The concatenated mask.
		- y_pred (torch.Tensor): The concatenated predicted values.
		- y_true (torch.Tensor): The concatenated true values.
		"""
		mask = th.cat(self.mask, dim=0)
		y_pred = th.cat(self.y_pred, dim=0)
		y_true = th.cat(self.y_true, dim=0)
		if (self.mean is not None) and (self.std is not None):
			y_pred = y_pred * self.std + self.mean
		return mask, y_pred, y_true

	def _reduce_scores(self, scores, reduction='none'):
		"""
		Reduce the scores based on the specified reduction method.

		Parameters:
		- scores (list): The list of scores to be reduced.
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- reduced_scores: The reduced scores.
		"""
		if reduction == 'none':
			return scores
		elif reduction == 'mean':
			return np.mean(scores)
		elif reduction == 'sum':
			return np.sum(scores)
		else:
			raise ValueError("Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

	def multilabel_score(self, score_func, reduction='none'):
		"""
		Calculate the multilabel score based on the specified score function.

		Parameters:
		- score_func (function): The score function to be used.
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- multilabel_score: The multilabel score.
		"""
		mask, y_pred, y_true = self._finalize()
		n_tasks = y_true.shape[1]
		scores = []
		for task in range(n_tasks):
			task_w = mask[:, task]
			task_y_true = y_true[:, task][task_w != 0]
			task_y_pred = y_pred[:, task][task_w != 0]
			task_score = score_func(task_y_true, task_y_pred)
			if task_score is not None:
				scores.append(task_score)
		return self._reduce_scores(scores, reduction)

	def pearson_r(self, reduction='none'):
		"""
		Calculate the Pearson correlation coefficient.

		Parameters:
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- pearson_r: The Pearson correlation coefficient.
		"""
		def score(y_true, y_pred):
			return pearsonr(y_true.numpy(), y_pred.numpy())[0]
		return self.multilabel_score(score, reduction)

	def spearman_r(self, reduction='none'):
		"""
		Calculate the Spearman correlation coefficient.

		Parameters:
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- spearman_r: The Spearman correlation coefficient.
		"""
		def score(y_true, y_pred):
			return spearmanr(y_true.numpy(), y_pred.numpy())[0]
		return self.multilabel_score(score, reduction)

	def mae(self, reduction='none'):
		"""
		Calculate the mean absolute error.

		Parameters:
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- mae: The mean absolute error.
		"""
		def score(y_true, y_pred):
			return F.l1_loss(y_true, y_pred).data.item()
		return self.multilabel_score(score, reduction)

	def rmse(self, reduction='none'):
		"""
		Calculate the root mean squared error.

		Parameters:
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- rmse: The root mean squared error.
		"""
		def score(y_true, y_pred):
			return th.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
		return self.multilabel_score(score, reduction)

	def roc_auc_score(self, reduction='none'):
		"""
		Calculate the ROC AUC score.

		Parameters:
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- roc_auc_score: The ROC AUC score.
		"""
		assert (self.mean is None) and (self.std is None), 'Label normalization should not be performed for binary classification.'
		def score(y_true, y_pred):
			if len(y_true.unique()) == 1:
				print('Warning: Only one class {} present in y_true for a task. ROC AUC score is not defined in that case.'.format(y_true[0]))
				return None
			else:
				return roc_auc_score(y_true.long().numpy(), th.sigmoid(y_pred).numpy())
		return self.multilabel_score(score, reduction)

	def pr_auc_score(self, reduction='none'):
		"""
		Calculate the PR AUC score.

		Parameters:
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- pr_auc_score: The PR AUC score.
		"""
		assert (self.mean is None) and (self.std is None), 'Label normalization should not be performed for binary classification.'
		def score(y_true, y_pred):
			if len(y_true.unique()) == 1:
				print('Warning: Only one class {} present in y_true for a task. PR AUC score is not defined in that case.'.format(y_true[0]))
				return None
			else:
				precision, recall, _ = precision_recall_curve(y_true.long().numpy(), th.sigmoid(y_pred).numpy())
				return auc(recall, precision)
		return self.multilabel_score(score, reduction)

	def compute_metric(self, metric_name, reduction='none'):
		"""
		Compute the specified metric.

		Parameters:
		- metric_name (str): The name of the metric to be computed.
		- reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default is 'none'.

		Returns:
		- metric_value: The computed metric value.
		"""
		if metric_name == 'rp':
			return self.pearson_r(reduction)
		elif metric_name == 'rs':
			return self.spearman_r(reduction)
		elif metric_name == 'mae':
			return self.mae(reduction)
		elif metric_name == 'rmse':
			return self.rmse(reduction)
		elif metric_name == 'roc_auc_score':
			return self.roc_auc_score(reduction)
		elif metric_name == 'pr_auc_score':
			return self.pr_auc_score(reduction)
		else:
			raise ValueError('Expect metric_name to be "rp" or "rs" or "mae" or "rmse" or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))


class EarlyStopping(object):
    """Early stop tracker
	
    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.
	
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
	
    Examples
    --------
    Below gives a demo for a fake training process.
	
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from model.utils import EarlyStopping
	
    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)
	
    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break
	
    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            #dt = datetime.datetime.now()
            filename = 'early_stop.pth'
		
        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'
		
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
	
    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
	
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score
	
    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
	
        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
	
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
	
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        th.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)
	
    def load_checkpoint(self, model):
        '''Load the latest checkpoint
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(th.load(self.filename)['model_state_dict'])

