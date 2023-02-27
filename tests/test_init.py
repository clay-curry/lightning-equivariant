from project.models import (
    MACEModel, TFNModel, GVPGNNModel, EGNNModel, MPNNModel, 
    SchNetModel, DimeNetPPModel, SEConvModel, SEGNNModel)

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
    assert model is not None
    
def test_segnn():
    model = SEGNNModel()
    assert model is not None