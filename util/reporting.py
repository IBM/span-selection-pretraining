import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Reporting:
    def __init__(self):
        self.check_count = 0
        self.check_every = 100
        self.start_time = time.time()
        self.last_time = self.start_time
        self.report_interval_secs = 300
        # For tracking moving averages of various values
        self.names = None
        self.averages = None
        self.count_for_averages = 0
        self.recency_weight = 0.001

    def is_time(self):
        self.check_count += 1
        if self.check_count % self.check_every == 0:
            if time.time() - self.last_time >= self.report_interval_secs:
                self.last_time = time.time()
                return True
        return False

    def moving_averages(self, **values):
        # create entries in avgs and counts when needed
        # update the avgs and counts
        if self.names is None:
            self.names = list(values.keys())
            self.averages = np.zeros(len(self.names))
        if self.count_for_averages < 1.0 / self.recency_weight:
            self.count_for_averages += 1
        rweight = max(self.recency_weight, 1.0 / self.count_for_averages)
        for ndx, name in enumerate(self.names):
            self.averages[ndx] = rweight * values[name] + (1.0 - rweight) * self.averages[ndx]

    def elapsed_seconds(self):
        return time.time()-self.start_time

    def display(self):
        # display the batch
        logger.info('==========================================')
        for n, v in zip(self.names, self.averages):
            logger.info(f'{n} = {v}')
