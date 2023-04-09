from django.shortcuts import render
from django.http import HttpResponse
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import os
from pathlib import Path
import random
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from PIL import Image
from django.http import JsonResponse
import pytesseract
from django.core.files.storage import FileSystemStorage 
from pdf2jpg import pdf2jpg

# def get_board_dicts(imgdir,js_file):
#     json_file = imgdir+"/" + js_file #Fetch the json file
#     with open(json_file) as f:
#         dataset_dicts = json.load(f)
#     l=len(dataset_dicts["annotations"])
#     x=1
#     t=dataset_dicts["annotations"]
#     record=[]
#     ptr=0
#     for i in dataset_dicts["images"]:
#        # print(x)
#         temp={}
#         filename = i["file_name"] 
#         temp["file_name"] = "/mnt/f/final_year_project/DocLayNet_dataset/PNG/"+filename
#         temp["image_id"] = i["id"]
#         temp["height"]=i["height"]
#         temp["width"]=i["width"]
#         temp_list=[]
#         for j in range(ptr,l):
#           temp_dict={}                                    
#           if t[j]["image_id"]==i["id"]:
#             temp_dict=t[j]
#             temp_dict["bbox_mode"]=BoxMode.XYWH_ABS
#             temp_dict["category_id"]=int(temp_dict["category_id"])-1
#             temp_list.append(temp_dict)
#           else:
#             break
#         temp["annotations"]=temp_list
#         record.append(temp)
#         ptr=j
#         x=x+1
#     return record

def set_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("doc_lay_net_train",)
    # cfg.DATASETS.TEST = ("doc_lay_net_val",)
    cfg.DATALOADER.NUM_WORKERS = 7
    cfg.MODEL.WEIGHTS = "/mnt/g/notebooks/model_0004999.pth"
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 270000  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    cfg.OUTPUT_DIR= "/mnt/g/notebooks/output"
    cfg.TEST.EVAL_PERIOD = 1
    cfg.SOLVER.CHECKPOINT_PERIOD=4000
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def crop(bbox, in_img: np.ndarray):
  """ bbox is a list with xmin, ymin, xmax, ymax """
  (x, y) = (int(bbox[0]), int(bbox[1]))
  (w, h) = (int(bbox[2]), int(bbox[3]))
  cropped_im = in_img[y:h, x:w]
  return cropped_im

def home(request):
   return render(request,'home.html')

def convertPdf(file):
   BASE_DIR = Path(__file__).resolve().parent.parent
   inputpath = os.path.join(BASE_DIR,"media",file)
   outputpath = os.path.join(BASE_DIR,"media")
   result = pdf2jpg.convert_pdf2jpg(inputpath,outputpath, pages="ALL")

def index(request):
    BASE_DIR = Path(__file__).resolve().parent.parent
    if request.method=="POST":
    #    email=request.POST.get("email")
       file=request.FILES["file"]
       fs=FileSystemStorage()
       fileName=fs.save(file.name,file)
       convertPdf(file.name)
    #    uploadedFileUrl=fs.url(fileName)
       cfg=set_cfg()
    #    print("cfg set")
       l=[]
       predictor = DefaultPredictor(cfg)
       path=os.path.join(BASE_DIR,"media",file.name+"_dir")
       count=1
       print(len(os.listdir(path)))
       for index in range(len(os.listdir(path))):
            image=os.path.join(path,str(index)+"_"+file.name+".jpg")
            im = cv2.imread(image)
            outputs = predictor(im)
            instances=outputs["instances"]
            detected_class_indexes=instances.pred_classes
            prediction_boxes=instances.pred_boxes
            class_catalog=["Caption","Footnote","Formula","List-item","Page-footer","Page-header","Picture","Section-header","Table","Text","Title"]
      #  v = Visualizer(im[:, :, ::-1], 
      #             scale=0.8,
      #             instance_mode=ColorMode.IMAGE_BW  
      #  )
      #  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      #  data=Image.fromarray(v.get_image())
      #  temp=file.name.split(".")
      #  data.save(os.path.join(BASE_DIR,"media",temp[0]+"_detection."+temp[1]))
      #  context={'uploaded_file_url':uploadedFileUrl,"result":"/media/"+temp[0]+"_detection."+temp[1]}
      #  print("Successful",os.path.join(BASE_DIR,"media",temp[0]+"_detection."+temp[1]))
            for idx, coordinates in enumerate(prediction_boxes):
                class_index=detected_class_indexes[idx]
                class_name=class_catalog[class_index]
                bbox=[coordinates[0].item(),coordinates[1].item(),coordinates[2].item(),coordinates[3].item()]
                crop_im = crop(bbox, in_img=im)
                img_rgb=cv2.cvtColor(crop_im,cv2.COLOR_BGR2RGB)
                #   print(pytesseract.image_to_string(img_rgb,config="--psm 6"))
                temp={}
                temp["page"]=count
                temp["bbox"]=bbox
                temp["class"]=class_name
                if class_name in ["Picture","Table","Formula","Caption"]:
                    temp["ndarray"]=crop_im.tolist()
                    l.append(temp)
                    continue
                else:
                    temp["content"]=pytesseract.image_to_string(img_rgb,config="--psm 6")[:-1]
                l.append(temp)
            count=count+1
        #  context={'uploaded_file':"/media/"+file.name,'jsonData':l}
    context={'jsonData':l}
    #    print("List:",l)
    # #    print(l[0].content)
    return render(request,'index.html',context)

# def document_layout_analysis(request):
#    for d in ["train","val","test"]:
#     DatasetCatalog.register("doc_lay_net_" + d, lambda d=d: get_board_dicts("/mnt/f/final_year_project/DocLayNet_dataset/COCO",d + ".json"))
#     MetadataCatalog.get("doc_lay_net_" + d).set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer","Page-header","Picture","Section-header","Table","Text","Title"])
#     board_metadata = MetadataCatalog.get("doc_lay_net_train")
#     cfg=set_cfg()
#     l=[]
#     predictor = DefaultPredictor(cfg)
    # dataset_dicts = get_board_dicts("/mnt/f/final_year_project/DocLayNet_dataset/COCO","test.json")
    # for d in random.sample(dataset_dicts,1):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     # run_easy_ocr(outputs, im)
    #     instances=outputs["instances"]
    #     detected_class_indexes=instances.pred_classes
    #     prediction_boxes=instances.pred_boxes
    #     # metadata=board_metadata
    #     class_catalog=["Caption","Footnote","Formula","List-item","Page-footer","Page-header","Picture","Section-header","Table","Text","Title"]
    #     v = Visualizer(im[:, :, ::-1],
    #                 metadata=board_metadata, 
    #                 scale=0.8,
    #                 instance_mode=ColorMode.IMAGE_BW  
    #     )
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # count=1
        # for idx, coordinates in enumerate(prediction_boxes):
        #     class_index=detected_class_indexes[idx]
        #     class_name=class_catalog[class_index]
        #     bbox=[coordinates[0].item(),coordinates[1].item(),coordinates[2].item(),coordinates[3].item()]
        #     crop_im = crop(bbox, in_img=im)
        #     img_rgb=cv2.cvtColor(crop_im,cv2.COLOR_BGR2RGB)
        #     print(pytesseract.image_to_string(img_rgb,config="--psm 6"))
        #     temp={}
        #     temp["page"]=count
        #     temp["bbox"]=bbox
        #     temp["class"]=class_name
        #     if class_name=="Picture" or class_name=="Table":
                # temp["ndarray"]=crop_im.tolist()
                # l.append(temp)
        #         continue
        #     else:
        #         temp["content"]=pytesseract.image_to_string(img_rgb,config="--psm 6")
        #     l.append(temp)
        # count=count+1
        # return HttpResponse(l)

