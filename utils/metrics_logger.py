import os
import pandas    as pd
from datetime    import datetime
from matplotlib  import pyplot      as plt

from .helper_metrics import InternalTrackedParameters

class DataLogger():
    def __init__(self, experiment_name : str) -> None:
        self.logs : list[InternalTrackedParameters] = []
        
        self.root_dir = self.__setup_dir__(experiment_name)

        self.current_best_map      = 0.0
        self.current_best_epoch    = 0
        self.current_epoch_is_best = False

        print("| Datalogger Setup Complete !")

    def __setup_dir__(self, experiment_name : str) -> str:
        base_dir = './runs'
        base_dir = os.path.abspath(base_dir)
        os.makedirs(base_dir, exist_ok = True)

        experiment_counter = 0
        for dir_entry in os.scandir(base_dir):
            if dir_entry.is_dir() and (experiment_name in dir_entry.name):
                experiment_counter += 1

        experiment_run = f"{experiment_name}_{experiment_counter + 1}" 
        dpath = os.path.join(base_dir, experiment_run)
        dpath = os.path.abspath(dpath)
        os.makedirs(dpath)           

        return dpath

    def get_filepath(self, file_name : str) -> str:
        return os.path.join(self.root_dir, file_name)

    def best_accuracy(self) -> str:
        return f"{(self.current_best_map):.2f}"
    
    def append(self, value : InternalTrackedParameters) -> None:
        self.current_epoch_is_best = False

        if value.vmap_avg > self.current_best_map:
            self.current_epoch_is_best = True
            self.current_best_epoch    = value.epoch
            self.current_best_map      = value.vmap_avg

        self.logs.append(value)
    
    def __to_df__(self) -> pd.DataFrame:
        data = [x.__dict__ for x in self.logs]
        return pd.DataFrame(data)

    def __plot_loss__(self) -> None:
        training_loss   = [i.training_loss   for i in self.logs]
        epoch           = [i.epoch           for i in self.logs]
        
        fpath = os.path.join(self.root_dir, "loss.png")
        plt.plot(epoch, training_loss,   label = 'Training Loss')
        plt.title('Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(fpath)
        plt.yscale('linear')
        plt.clf()
    
    def __plot_accuracy__(self) -> None:
        valid_mavg = [i.vmap_avg for i in self.logs]
        valid_mp50 = [i.vmap_50  for i in self.logs]
        valid_mp75 = [i.vmap_75  for i in self.logs]
        epoch      = [i.epoch    for i in self.logs]
        
        fpath = os.path.join(self.root_dir, "accuracy.png")
        plt.plot(epoch, valid_mavg,   label = 'Average MAP50-95')
        plt.plot(epoch, valid_mp50,   label = 'Average MAP50')
        plt.plot(epoch, valid_mp75,   label = 'Average MAP75')
        plt.plot(
            [self.current_best_epoch, self.current_best_epoch],
            plt.ylim(), 
            label = f'Best : {self.current_best_map:.2f}'
        )
        plt.title('Accuracy')
        plt.yscale('linear')
        plt.legend()
        plt.savefig(fpath)
        plt.yscale('linear')
        plt.clf()
    
    def save(self) -> None:
        self.__plot_loss__()
        self.__plot_accuracy__()

        dfx = self.__to_df__()
        dfx.to_csv( 
            os.path.join(self.root_dir, "log.csv"), 
            index = False
        )
    
    def write_text(self, message : str) -> None:
        log_file = os.path.join(self.root_dir, "log.txt")
        with open(log_file, 'a+') as file:
            file.write(message)
            file.write("\n")
        print(message)

    