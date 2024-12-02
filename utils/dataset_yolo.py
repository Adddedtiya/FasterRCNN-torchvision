import os
import torch
import cv2            as cv
import numpy          as np
import albumentations as A
from tqdm             import tqdm
from PIL              import Image
from torch.utils.data import DataLoader, Dataset

def __pbf__(input_string : str) -> float:
    x = float(input_string)
    x = max(0, x)
    x = min(1, x)
    return x 

def __itr_convert_yolo__(img_width : int, img_height : int, bbx_line : str) -> tuple[int, list[float]]:
    bbxp = bbx_line.strip().split()

    class_id   = int(bbxp[0]) + 1 # offset by 1 (0 is background)
    x_center   = __pbf__(bbxp[1]) * img_width
    y_center   = __pbf__(bbxp[2]) * img_height
    box_width  = __pbf__(bbxp[3]) * img_width
    box_height = __pbf__(bbxp[4]) * img_height

    x1 = x_center - (box_width  / 2)
    y1 = y_center - (box_height / 2)
    x2 = x_center + (box_width  / 2)
    y2 = y_center + (box_height / 2)

    b_cls = class_id
    b_bbx = [x1, y1, x2, y2]

    return (b_cls, b_bbx)

def __itr_convert_annot__(image_path : str, label_path : str) -> tuple[list[int], list[list[float]]]:
    # Load image to get its dimensions
    image = Image.open(image_path)
    width, height = image.size
    
    # Read YOLO annotation file
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    bbx_boxes : list[list[float]] = []
    bbx_class : list[int]         = []
    for bbx_line in lines:
        class_id, bounding_box = __itr_convert_yolo__(
            width,
            height,
            bbx_line
        )

        bbx_boxes.append(bounding_box)
        bbx_class.append(class_id)
    
    return bbx_class, bbx_boxes

class InternalImage:
    def __init__(self, fpath : str, lpath : str) -> None:
        self.image_path = fpath 
        self.label_path = lpath
        
        rec_tbx = __itr_convert_annot__(
            self.image_path,
            self.label_path
        )

        self.bbx_count = len(rec_tbx[0]) # total boxes
        self.bbx_class = rec_tbx[0]      # classes
        self.bbx_boxes = rec_tbx[1]      # boxes

class YoloDataset(Dataset):
    def __init__(self, directory_root : str, image_path : str, label_path : str, aug : list = []) -> None:
        super().__init__()
        self.dataset : list[InternalImage] = []
        
        self.image_root = os.path.join(directory_root, image_path)
        self.image_root = os.path.abspath(self.image_root)

        self.label_root = os.path.join(directory_root, label_path) 
        self.label_root = os.path.abspath(self.label_root)

        self.class_ids : list[int] = []    

        self.augmentation = A.Compose(
            aug, 
            bbox_params =  A.BboxParams(
                format       = 'pascal_voc', 
                label_fields = ['class_labels']
            )
        )
        self.__prase_dir__()
        print("# Loaded Image and Lables :", len(self.dataset))
        print("# Detected Classes :", self.class_ids)

    def __prase_dir__(self) -> None:
        # this assume your file doesnt have multple '.'
        file_ids = [x.split('.')[0] for x in os.listdir(self.image_root) if x.endswith('jpg')] # only jpg files
        cls_vals : list[int] = [0] # 0 is for background
        for file_index in tqdm(file_ids):
            image_file = os.path.join(self.image_root, f'{file_index}.jpg')
            label_file = os.path.join(self.label_root, f'{file_index}.txt')

            if not os.path.exists(image_file):
                continue

            if not os.path.exists(label_file):
                continue
                
            img_info = InternalImage(
                image_file,
                label_file
            )

            if img_info.bbx_count == 0:
                continue # im sorry this is the only sane way...

            cls_vals = cls_vals + img_info.bbx_class
            self.dataset.append(img_info)
        clsx = set(cls_vals)
        self.class_ids = list(clsx)
        self.class_ids.sort()
 
    def get_total_class(self, background = True) -> int:
        if background: 
            return len(self.class_ids)
        return len(self.class_ids) - 1

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index : int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
        image_info = self.dataset[index]

        image_array = cv.imread(image_info.image_path)
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)

        transformed = self.augmentation(
            image        = image_array, 
            bboxes       = image_info.bbx_boxes, 
            class_labels = image_info.bbx_class
        )

        trf_image : np.ndarray        = transformed['image']
        trf_box   : list[list[float]] = transformed['bboxes']
        trf_cls   : list[int]         = transformed['class_labels']

        image_tensor = np.array(trf_image, dtype = np.float32)
        image_tensor = np.moveaxis(image_tensor, -1, 0)
        image_tensor = torch.from_numpy(image_tensor)
        
        boxes_tensor = np.array(trf_box, dtype = np.float32)
        boxes_tensor = torch.from_numpy(boxes_tensor)

        class_tensor = np.array(trf_cls, dtype = np.int64)
        class_tensor = torch.from_numpy(class_tensor)

        # sanity check if edge case of 0 boxes
        if len(trf_box) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype = torch.float32)
            class_tensor = torch.zeros((0, 1), dtype = torch.int64)

        data_dict = {
            'boxes' : boxes_tensor,
            'labels': class_tensor
        }

        return (image_tensor, data_dict)

    @staticmethod
    def collate(data : any) -> any:
        return data

    @staticmethod
    def arrange(collated_data : list, device : torch.device) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        images : list[torch.Tensor]            = []
        target : list[dict[str, torch.Tensor]] = []

        for image, labels in collated_data:
            image : torch.Tensor = image.to(device)
            
            data_dict = {
                'boxes' : labels['boxes'].to(device),
                'labels': labels['labels'].to(device)
            }

            images.append(image)
            target.append(data_dict)

        return (images, target)
    
    def dataloader(self, batch_size : int) -> DataLoader:
        return DataLoader(
            self, 
            batch_size = batch_size, 
            collate_fn = YoloDataset.collate, 
            shuffle    = True
        )

if __name__ == "__main__":
    print("Yolo Dataset Processor")

    dd = YoloDataset(
        './example_yolo/valid', '.', '.', [
            A.RandomCrop(500, 300)
        ]
    )

    # Dataloaders --
    dd_dl = DataLoader(
        dd,
        batch_size = 4,
        shuffle    = True,
        collate_fn = YoloDataset.collate,
        pin_memory = True if torch.cuda.is_available() else False,
    )

    print("Setup Complete !")
    print(len(dd_dl))

    for x in dd_dl:
        print(type(x)) # list

        images, targets = YoloDataset.arrange(x, device = 'cpu')

        print(type(images),  len(images))
        print(type(targets), len(targets))

        # for y in x:
        #     print(type(y)) # tuple
        #     i_y, m_y = y # torch.tensor, dict  
        #     print(type(i_y), type(m_y))
        #     print("BOX")
        #     print(m_y['boxes'].shape)
        #     print(m_y['boxes'])
        #     print("LABEL")
        #     print(m_y['labels'].shape)
        #     print(m_y['labels'])
        #     print()
        
        print("")