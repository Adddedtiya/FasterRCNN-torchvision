import torch
import numpy                as np
from sklearn.metrics        import classification_report, confusion_matrix
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from .helper_metrics        import InternalTrackedParameters

class DetectionMetrics:
    def __init__(self) -> None:
        self.loss   : list[float] = []
        self.metric = MeanAveragePrecision(
            iou_type   = "bbox", 
            box_format = 'xyxy'
        )
    
    def compute_val(self, prediction : list[dict[str, torch.Tensor]], ground_truth : list[dict[str, torch.Tensor]]) -> None:
        # im sorry, this will crash when empty, so im fixing the dataloader.. 
        self.metric.update(prediction, ground_truth)

    def compute_loss(self, ls_dct : dict[str, torch.Tensor], div = 1) -> torch.Tensor:
        ls = torch.stack([ls_dct[x] for x in ls_dct])
        lx = torch.sum(ls) / div

        ld = float(lx.item())
        self.loss.append(ld)   

        return lx

    def reset(self) -> None:
        self.loss = []
        self.metric.reset()

    def average_loss(self) -> float:
        return np.mean(self.loss)
    
    def __limiter__(self, x : float, lower = float('-inf'), upper = float('inf')) -> float:
        x = max(lower, x)
        x = min(upper, x)
        return x
    
    def average_map(self) -> tuple[float, float, float]:
        x = self.metric.compute()
        
        map_avg = self.__limiter__(x['map'],    lower = 0.0)
        map_m50 = self.__limiter__(x['map_50'], lower = 0.0)
        map_m75 = self.__limiter__(x['map_75'], lower = 0.0)

        return (map_avg, map_m50, map_m75)

    def summary(self, epoch : int) -> InternalTrackedParameters:
        mavg, m50, m75 = self.average_map() 
        x = InternalTrackedParameters(
            training_loss = self.average_loss(),
            vmap_avg = mavg,
            vmap_50  = m50,
            vmap_75  = m75,
            epoch    = epoch
        )
        return x

if __name__ == "__main__":
    print("Helper - Tester !")

    preds_1 = [
        dict(
          boxes  = torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
          scores = torch.tensor([0.536]),
          labels = torch.tensor([0]),
        )
    ]        

    preds_2 = [
        dict(
          boxes  = torch.tensor([[214.0, 42.0, 542.0, 225.0]]),
          labels = torch.tensor([0]),
        )
    ]

    metric = MeanAveragePrecision(iou_type = "bbox", box_format = 'xyxy')
    metric.update(preds_1, preds_2)

    preds_1 = [
        dict(
          boxes  = torch.tensor([[146.0, 53.0, 332.0, 112.0]]),
          scores = torch.tensor([0.536]),
          labels = torch.tensor([0]),
        )
    ]        

    preds_2 = [
        dict(
          boxes  = torch.tensor([[129.0, 62.0, 350.0, 110.0]]),
          labels = torch.tensor([0]),
        )
    ]
    metric.update(preds_1, preds_2)


    x = metric.compute()
    print(x['map'])
    print(x)