import torch
import warnings
import torchvision
from torchvision.models.detection.faster_rcnn   import FastRCNNPredictor
from torchvision.models.detection               import FasterRCNN_MobileNet_V3_Large_FPN_Weights

warnings.filterwarnings("ignore", category = UserWarning)

def create_model_mobilenet(num_classes : int):
    """
    Create a model for object detection using the Faster R-CNN architecture.

    Parameters:
    - num_classes (int) : The number of classes for object detection.
    - checkpoint  (str) : checkpoint path for the pretrained custom model
    - device      (str) : torch device
    
    Returns:
    - model (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn): The created model for object detection.
    """
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights             = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        pretrained          = True,
        pretrained_backbone = True,
        min_size            = 600,
        max_size            = 900,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


# sanity check 
if __name__ == "__main__":
    print("## Sanity Testing MobileNetV3 FRC ##")

    model = create_model_mobilenet(1)
    model.train()
    
    faux_x = [torch.rand(3, 720, 720)]
    faux_y = [{
        'labels' : torch.randint(1, 5,   (3,  ), dtype = torch.int64),
        'boxes'  : torch.tensor([
            [ 0.0000,  89.8974, 300.0000, 181.5033],
            [ 0.0000, 178.1026, 300.0000, 272.1582],
            [ 0.0000, 368.8718, 300.0000, 470.6772],            
        ]),
    }]

    print(faux_y[0]['boxes'].shape)

    lossx = model(faux_x, faux_y)
    print(lossx)
