import logging
import os
import signal

import numpy as np
import torch

from glob import glob


def mkdir(s):
  """
  Create a directory if it doesn't already exist.
  """
  if not os.path.exists(s):
    os.makedirs(s)


def get_files(d, pattern, sort=True):
  """
  Return a list of files in a given directory.

  Args:
    d (str): The path to the directory.
    pattern (str): The wildcard to filter files with.
    sort (bool): Whether to sort the returned list.
  """
  files = glob(os.path.join(d, pattern))
  files = [f for f in files if os.path.isfile(f)]
  if sort:
    files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
  return files


class Checkpoint:
  """
  Save and restore model and optimizer states.
  """
  def __init__(self, model, optimizer=None):
    """
    Constructor.
    """
    self.model = model
    self.optimizer = optimizer

  def restore(self, save_path, device=None):
    """
    Restore a state from a saved checkpoint.

    Args:
      save_path (str): The filepath to the saved checkpoint.
      device (torch.device): The device on which to
        restore the state.
    """
    try:
      state = torch.load(save_path, map_location=device)
      try:
        self.model.load_state_dict(state['model_weights'])
        if self.optimizer is not None:
          self.optimizer.load_state_dict(state['optim_state'])
        logging.info('Successfully loaded model weights from {}.'.format(save_path))
        return True
      except Exception as e:
        # there was an issue loading the state which means
        # either the model definition and saved weights
        # do not agree or they were not saved in the first
        # place.
        # since this is a severe issue, we raise an error
        # rather than allowing the program to proceed.
        raise e
    except FileNotFoundError as e:
      logging.error(e)
      return False

  def save(self, save_path):
    """
    Save a state to disk.

    Modified from brentyi/fannypack.

    Args:
      save_path (str): The name of the checkpoint to save.
    """
    state = {'model_weights': self.model.state_dict()}
    if self.optimizer is not None:
      state['optim_state'] = self.optimizer.state_dict()

    # ignore ctrl+c while saving
    try:
      orig_handler = signal.getsignal(signal.SIGINT)
      signal.signal(signal.SIGINT, lambda _sig, _frame: None)
    except ValueError:
      # signal throws a ValueError if we're not in the main thread
      orig_handler = None
    
    # atomic save
    save_dir = os.path.dirname(save_path)
    tmp_path = os.path.join(save_dir, "tmp-{}.ckpt".format(np.random.randint(1e10)))
    torch.save(state, tmp_path)
    # rename is an atomic operation in python
    # it is POSIX compliant according to docs
    # https://docs.python.org/3/library/os.html#os.rename
    os.rename(tmp_path, save_path)
    logging.info('Saved checkpoint at {}.'.format(save_path))
    
    # restore SIGINT handler
    if orig_handler is not None:
      signal.signal(signal.SIGINT, orig_handler)


class CheckpointManager:
  """
  A model and optimizer checkpoint manager.
  """
  def __init__(self, checkpoint, directory, device, max_to_keep=10):
    """
    Constructor.

    Args:
      checkpoint (Checkpoint): An instance of `Checkpoint`.
      directory (str): The directory in which checkpoints will be saved.
      device (torch.device): The computing device on which to restore
        checkpoints.
      max_to_keep (int): The maximum number of checkpoints to keep.
        Amongst all saved checkpoints, checkpoints will be deleted
        oldest first, until `max_to_keep` remain.
    """
    assert max_to_keep > 0, "max_to_keep should be a positive integer."

    self.checkpoint = checkpoint
    self.directory = directory
    self.max_to_keep = max_to_keep
    self.device = device
    self.latest_checkpoint = None

    # create checkpoint directory if it doesn't
    # already exist
    mkdir(self.directory)

  def restore_or_initialize(self):
    """
    Restore items in checkpoint from the latest checkpoint file.

    Returns:
      The global iteration step. This is parsed from the latest
        checkpoint file if one is found, else 0 is returned.
    """
    ckpts = get_files(self.directory, "*.ckpt")
    if ckpts:
      last_ckpt = ckpts[-1]
      status = self.checkpoint.restore(last_ckpt, self.device)
      if not status:
        logging.info('Could not restore latest checkpoint file.')
        return 0
      self.latest_checkpoint = last_ckpt
      return int(os.path.basename(last_ckpt).split('.')[0])
    return 0

  def save(self, global_step):
    """
    Create a new checkpoint.

    Args:
      global_step (int): The iteration number which will be used
        to name the checkpoint.
    """
    save_path = os.path.join(self.directory, "{:016d}.ckpt".format(global_step))
    self.checkpoint.save(save_path)
    self.latest_checkpoint = save_path
    self._trim_checkpoints()

  def _trim_checkpoints(self):
    """
    Trim older checkpoints until `max_to_keep` remain.
    """
    # get a list of checkpoints in reverse
    # chronological order
    ckpts = get_files(self.directory, "*.ckpt")[::-1]

    # remove until `max_to_keep` remain
    num_remove = len(ckpts) - self.max_to_keep
    while num_remove > 0:
      ckpt_name = ckpts.pop()
      os.remove(ckpt_name)
      num_remove -= 1

  @staticmethod
  def load_latest_checkpoint(checkpoint, directory, device):
    ckpts = get_files(directory, "*.ckpt")
    if ckpts:
      last_ckpt = ckpts[-1]
      checkpoint.restore(last_ckpt, device)
    else:
      logging.error('No checkpoints found in {}.'.format(directory))