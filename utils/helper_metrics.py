from datetime    import datetime
from dataclasses import dataclass

@dataclass
class InternalTrackedParameters:
    
    training_loss : float
    
    vmap_avg : float
    vmap_50 : float
    vmap_75 : float
    
    epoch   : int

    time_stamp    : datetime = datetime.now()

    def __str__(self) -> str:
        return f"{self.epoch} | {self.training_loss} | {self.vmap_avg}"

    