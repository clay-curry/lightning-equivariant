import lightning.pytorch as L

from e3nn.o3 import Irreps
from project.models import SEConvModel
from project.models.segnn import WeightBalancedIrreps
from project.utils.trajectory_datamodule import FlightTrajectoryDataModule

def test_conv_train():
    L.seed_everything(0)
    
    input_irreps=Irreps("1x0e")
    output_irreps=Irreps("6x1e")
    pool="avg"
    layers=7
    lmax_attr=3
    hidden_features = 28
    edge_attr_irreps=Irreps.spherical_harmonics(lmax_attr)
    node_attr_irreps=Irreps.spherical_harmonics(lmax_attr)
    hidden_irreps = WeightBalancedIrreps(Irreps("{}x0e".format(hidden_features)), node_attr_irreps, sh=True, lmax=lmax_attr)

    model = SEConvModel(
        input_irreps=input_irreps,
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        edge_attr_irreps=edge_attr_irreps,
        node_attr_irreps=node_attr_irreps,
        additional_message_irreps = Irreps("2x0e"),
        norm="batch",
        task="node",
        num_layers=layers,
        
    )
    
    datamodule = FlightTrajectoryDataModule("experiments/data", 25, 10, 5)
    trainer = L.Trainer(fast_dev_run=5)
    trainer.fit(model, datamodule=datamodule)
    assert True