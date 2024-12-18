""" Main entry point of the ONMT library """
from __future__ import division, print_function
import onmt.inputters
import onmt.utils
from onmt.trainer import Trainer
import sys
import onmt.utils.optimizers
onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

__version__ = "0.4.1"
