### Check if packages are installed
def test_has_pytorch_installed():
    import torch
    assert torch is not None

def test_has_lightning_installed():
    import lightning
    assert lightning is not None

def test_has_pyg_installed():
    import torch_geometric
    assert torch_geometric is not None
    
def test_has_e3nn_installed():
    import e3nn
    assert e3nn is not None
