import os
from abc import ABC, abstractmethod
from typing import Iterable
from multiprocessing import Process


class DancePreprocessor(ABC):
    def __init__(self, data_dir: str, out_dir: str, categories: Iterable[str], n_jobs=8):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.categories = categories
        self.n_jobs = n_jobs

    @abstractmethod
    def preprocess_category(self, category: str):
        pass

    def run(self):
        procs = []
        for cat_dir in os.scandir(self.data_dir):  # categories
            if cat_dir.is_dir():
                category = cat_dir.name
                if category in self.categories:
                    procs.append(Process(target=self.preprocess_category, args=(category,)))

        for p in procs:
            p.start()

        for p in procs:
            p.join()
