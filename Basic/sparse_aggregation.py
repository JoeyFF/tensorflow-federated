import tensorflow as tf
import tensorflow_federated as tff
import collections
import itertools
import numpy as np
import nest_asyncio
from typing import Callable, List, Tuple

nest_asyncio.apply()
MAX_TOKENS_SELECTED_PER_CLIENT = 6