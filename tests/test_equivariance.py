
import torch
from os import listdir
from os.path import join
from random import choice
from pandas import read_csv
from project.utils.trajectory_datamodule import rotate_scenario

from project.models import (
    MACEModel, TFNModel, GVPGNNModel, EGNNModel, MPNNModel, 
    SchNetModel, DimeNetPPModel, SEGNNModel, SEConvModel
)

DATA_DIR = join('experiments', 'data')

def get_random_vantage_and_rotated_vantage():
    global DATA_DIR
    rand_datafile = lambda : join(DATA_DIR, choice(listdir(DATA_DIR)))
    open_scenario = lambda path: torch.from_numpy(read_csv(path, dtype='float64', usecols=['x', 'y', 'z']).values)
    rand_scenario = open_scenario(rand_datafile())
    return rand_scenario, rotate_scenario(rand_scenario)
    
def test_mace():
    model = MACEModel()
    assert model is not None
    
def test_tfn():
    model = TFNModel()
    assert model is not None

def test_gvpgnn():
    model = GVPGNNModel()
    assert model is not None
    
def test_egnn():
    model = EGNNModel()
    assert model is not None
    
def test_mpnn():
    model = MPNNModel()
    assert model is not None
    
def test_schnet():
    model = SchNetModel()
    assert model is not None
    
def test_dimenetpp():
    model = DimeNetPPModel()
    assert model is not None

def test_seconv():
    model = SEConvModel()
    vantage, rotated_vantage = get_random_vantage_and_rotated_vantage()
    
    pred1 = model(vantage)
    pred2 = model(rotated_vantage)
    assert torch.allclose(pred1, pred2)
    
def test_maneuvergpt():
    model = SEGNNModel()
    assert model is not None