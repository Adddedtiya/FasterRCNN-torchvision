import torch
import random
import numpy as np

import albumentations as A

from tqdm             import tqdm
from torch.utils.data import DataLoader

from model.model_mobilenet import create_model_mobilenet
from utils.dataset_yolo    import YoloDataset
from utils.helper_tester   import DetectionMetrics
from utils.metrics_logger  import DataLogger

SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.use_deterministic_algorithms(True)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_epochs = 3
batch_size   = 4
accumulation = 16

if __name__ == "__main__":

    print("| Pytorch Model Training !")
    
    print("| Total Epoch  :", total_epochs)
    print("| Batch Size   :", batch_size)
    print("| Device       :", device)
    print("| Accumulation :", accumulation)

    print("| Configuring Dataset...")

    print("| Training Dataset")
    training_dataset = YoloDataset(
        './example_yolo/valid', '.', '.', [
            # Training Augmentations

            A.RandomCrop(
                height = 300,
                width  = 300
            )
        ]
    )

    print("| Testing Dataset")
    testing_dataset = YoloDataset(
        './example_yolo/valid', '.', '.', [
            # Testing Augmentations,
            A.Resize(
                height = 800,
                width  = 800 
            )
        ]
    )

    print("| Validation Dataset")
    validation_dataset = YoloDataset(
        './example_yolo/valid', '.', '.', [
            A.Resize(
                height = 800,
                width  = 800 
            )
        ]
    )

    model = create_model_mobilenet(
        num_classes = training_dataset.get_total_class(background = False)
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
    
    logger  = DataLogger("FasterRCNN-MobileNetV3")
    metrics = DetectionMetrics()

    validation_datasetloader = validation_dataset.dataloader(batch_size // 2)
    training_datasetloader   = training_dataset.dataloader(batch_size)
    testing_datasetloader    = testing_dataset.dataloader(1)
    
    # Training Evaluation Loop
    for current_epoch in range(total_epochs):
        print("Epoch :", current_epoch)
        metrics.reset() # reset the matrics
        
        # Training Setup
        model.train()   # set the model to train

        # Training Loop
        for i, data_values in enumerate(tqdm(training_datasetloader, desc = "Training :")):
            
            # convert the data to faster-rcnn format
            images, targets = YoloDataset.arrange(data_values, device = device)

            # FasterRCNN returns the loss
            ldct = model(images, targets)
            loss = metrics.compute_loss(ldct)
            loss = loss / accumulation
            loss.backward()
            
            # Accumulate Gradients And Step the parameters
            if ((i + 1) % accumulation) == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Validation Setup
        model.eval()

        # Validation Loop
        for data_values in tqdm(validation_datasetloader, desc = "Testing :"):
            
            # convert the data to faster-rcnn format
            images, targets = YoloDataset.arrange(data_values, device = device)

            # FasterRCNN returns the loss
            predictions = model(images)
            metrics.compute_val(predictions, targets)

        logger.append( metrics.summary(current_epoch) )

        if logger.current_epoch_is_best:
            print("> Best map50-95 :", logger.best_accuracy())
            model_state     = model.state_dict()
            optimizer_state = optimizer.state_dict()
            state_dictonary = {
                "model_state"     : model_state,
                "optimizer_state" : optimizer_state
            }
            torch.save(
                state_dictonary, 
                logger.get_filepath("best_checkpoint.pth")
            )

        logger.save()
        print("")
    
    # Testing Part !
    print("| Training Complete, Loading Best Checkpoint")
    
    # Load Model State
    state_dictonary = torch.load(
        logger.get_filepath("best_checkpoint.pth"), 
        map_location = device,
        weights_only = True
    )
    model.load_state_dict(state_dictonary['model_state'])
    model = model.to(device)
    
    # Testing System 
    model.eval()    # set the model to evaluation
    metrics.reset() # reset the metrics

    for data_values in tqdm(testing_datasetloader, desc = "Final Testing :"):
            
        # convert the data to faster-rcnn format
        images, targets = YoloDataset.arrange(data_values, device = device)

        # FasterRCNN returns the loss
        predictions = model(images)
        metrics.compute_val(predictions, targets)
    
    mavg, m50, m75 = metrics.average_map()
    print("")
    logger.write_text("#### Complete - Final Metrics ###")
    logger.write_text(f"# MAP50-95 : {mavg}")
    logger.write_text(f"# MAP50    : {m50}")
    logger.write_text(f"# MAP75    : {m75}")
    logger.write_text("####")
    print("Complete !")