import os, sys
from typing import List, Dict, Union
from abc import ABCMeta, abstractmethod

from tqdm import tqdm
import torch
import numpy as np
import sklearn

from utils import Dataset, LabelledDataset, buildMLP

class BaseClassifier(metaclass = ABCMeta):

    def __init__(self, config: Dict):
        super().__init__()

        self._init(config)
    
    @abstractmethod
    def _init(self, config):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X, y, normalizeX = False):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X, normalizeX = False):
        raise NotImplementedError
    
    def score(self, X, y, normalizeX = False):
        ypred = self.predict(X, normalizeX)

        score = np.sum(ypred == y) / len(ypred)

        return score

    def save(self, path):
        if type(self.model) == torch.nn.Module:
            torch.save(self.model.state_dict(), 
            path + '.pth')
        else:
            sklearn.externals.joblib.dump(
                self.model, path + '.pkl'
            )
    
    def load(self, path):
        if type(self.model) == torch.nn.Module:
            self.model.load_state_dict(torch.load(
                path + '.pth',
                map_location=lambda storage, loc: storage.cuda(self.device[-1])
            ))
        
        else:
            self.model = sklearn.externals.joblib.load(
                path +'.pkl'
            )
        
    
class SVCClassifier(BaseClassifier):

    def _init(self, config):
        self.conf = config
        self.penalty = config['penalty']
        self.class_weight = config['class_weight']

        self.model = sklearn.svm.SVC(
            C=1 / self.penalty
            if self.penalty is not None and self.penalty > 0
            else np.inf,
            gamma='scale',
            class_weight = self.class_weight
        )

    def fit(self, X, y, normalizeX = False):
        if normalizeX:
            X = X / np.linalg.norm(X, axis = -1)[..., None]
        
        if not self.conf.get('gridsearch'):
            return self.model.fit(X, y)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(
                self.model, self.conf['gridsearch'],
                cv = self.conf.get('cv', 5),
                n_jobs = self.conf.get('n_jobs', 5)
            )
            train_size = np.shape(X)[0]
            if train_size <= 10000:
                grid_search.fit(X, y)
            else:
                # If the training set is too large, subsample 10000 train
                # examples
                split = sklearn.model_selection.train_test_split(
                    X, y,
                    train_size=10000, random_state=0, stratify=y
                )
                grid_search.fit(split[0], split[2])
            self.model = grid_search.best_estimator_
            return self.model

    def predict(self, X, normalizeX = False):
        if normalizeX:
            X = X / np.linalg.norm(X, axis = -1)[..., None]
        
        return self.model.predict(X)
    
    def score(self, X, y, normalizeX = False):
        if normalizeX:
            X = X / np.linalg.norm(X, axis = -1)[..., None]
        
        return self.model.score(X, y)


class MLPNet(BaseClassifier):

    def _init(self, config):
        print("initialized MLPNet")

        self.config = config
        self.device = config['training']['device']

        self.model = buildMLP(**config['network'])

        self.model.to(config['training']['device'])
        self.model.to(dtype=torch.double)
    
    def fit(self, X, y, normalizeX = False):
        print(np.unique(y, return_counts=True))
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
            y = torch.from_numpy(y).to(dtype = torch.long)
        if normalizeX:
            X = X / torch.norm(X, dim = -1)[...,None]
        dataset = LabelledDataset(X, y)
        datagenerator = torch.utils.data.DataLoader(dataset,
            batch_size = self.config['training']['batch_size'] )

        nepochs = self.config['training']['nepochs']
        optimizer = getattr(torch.optim, self.config['training']['optimizer'])(
            self.model.parameters(), **self.config['training']['optim_params']
        )
        self.lossfn = getattr(torch.nn, self.config['training']['lossfn'])()

        self.model.train()

        for epoch in range(nepochs):
            epochloss = 0.
            for xbatch, ybatch in tqdm(datagenerator):
                xbatch = xbatch.to(self.device)
                ybatch = ybatch.to(self.device)
                optimizer.zero_grad()

                ypred = self.model(xbatch)
                loss = self.lossfn(ypred, ybatch)
                
                loss.backward()
                optimizer.step()

                epochloss += loss.detach()

            print(f"epoch {epoch} loss = {epochloss}")
        
        return self.model

    def predict(self, X, normalizeX = False):
        if normalizeX:
            X = X / torch.norm(X, dim=-1)[...,None]
        
        dataset = Dataset(X)
        datagenerator = torch.utils.data.DataLoader(
            dataset, batch_size = 128
        )

        self.model.eval()

        preds = []

        with torch.no_grad():
            for xbatch in datagenerator:
                xbatch = xbatch.to(self.device)
                ypred = self.model(xbatch)
                ypred = torch.argmax(ypred, dim = -1) + 1

                preds.append(ypred.cpu())
        
        preds = torch.cat(preds, dim=0).to(dtype=torch.long)

        return preds.numpy()
    
    def score(self, X, y, normalizeX = False):

        predictions = self.predict(X, normalizeX)
        meanError = np.sum(predictions == y) / len(y)

        return meanError
