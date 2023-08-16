from django.shortcuts import render
from .models import ImageModel
from rest_framework import permissions
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageSerializer
from rest_framework.viewsets import ModelViewSet
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from django.conf import settings
from django.http import JsonResponse
import base64
import random
# import threading
# from django.core.files import FileSystemStorage
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import mrcnn.config
import mrcnn.model
import tensorflow as tf


from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw

class CustomFileSystemStorage():
    def get_available_name(self, name, max_length=None):
        self.delete(None)
        return name


class ImageAiViewSet(ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = (permissions.IsAuthenticated,)

    @csrf_exempt
    def create(self, request, *args, **kwargs):
        identify_object = request.data['identify_object']
        inputted_image = request.data['inputted_image']
        image = request.FILES['inputted_image']
        # return JsonResponse(image.name, safe=False)
        user = request.user
        data = ImageModel.objects.create(identify_object=identify_object, inputted_image=inputted_image, user=user)
        serializer = ImageSerializer(data, many=False)
        # return Response(serializer.data['id'])
        outputted_image = predict(image.name)
        data = ImageModel.objects.filter(id=serializer.data['id']).update(outputted_image=outputted_image)
        # serializer = ImageSerializer(data, many=False) 
        return Response({
            'message': 'Image Sent Successfully',
            # 'data': serializer.data,
            'status': 'success',
        },200)
    
    def get(self, request, *args, **kwargs):
        data = ImageModel.objects.filter(user=request.user).all()
        serializer = ImageSerializer(data, many=True)
        return Response(serializer.data, 200)
    
class AConfig(mrcnn.config.Config):
	# define the name of the configuration
    

    NAME = "aerial_cfg_coco"

    NUM_CLASSES = 6
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def predict(image_to_be_predicted):
    CLASS_NAMES = ['BG', 'building', 'vegetation', 'flood', 'car', 'road']
     
    # model = tf.keras.models.load_model('./aerial-transfer-learning/aerial_mask_rcnn_trained.h5')
     
    class AConfig(mrcnn.config.Config):

        NAME = "aerial_cfg_coco"
        NUM_CLASSES = 6
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
     
    model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=AConfig(),
                             model_dir=os.getcwd())
    model.load_weights(filepath="./aerial-transfer-learning/aerial_mask_rcnn_trained.h5", by_name=True)
    # decoded_image = base64.b64decode(image_to_be_predicted)
    # nparr = np.frombuffer(decoded_image, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    path = './mediafiles/images/' + image_to_be_predicted
    image = cv2.imread(path)
    if image is None:
        return 'Invalid image type'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.detect([image], verbose=0)
    # return results
    results = results[0]
    # class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    # for i in range(results['rois'].shape[0]):
    #     class_id = results['class_ids'][i]
    #     score = results['scores'][i]
    #     mask = results['masks'][:, :, i]
    #     mask_image = np.zeros_like(image)
    #     mask_color = class_colors[class_id] if class_id < len(class_colors) else (0, 0, 0)
    #     mask_image[mask] = mask_color
    #     overlay = cv2.addWeighted(image, 0.7, mask_image, 0.3, 0)
    #     y1, x1, y2, x2 = results['rois'][i]
    #     cv2.rectangle(overlay, (x1, y1), (x2, y2), mask_color, 2)
    #     label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
    #     cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)
    #     image = overlay
        
    # output_path = './mediafiles/images/newimage.jpg'
    # new_predicted_image_path  = cv2.imwrite(output_path, image)
    # return new_predicted_image_path
    image_new_name = 'predicted_' + str(random.randint(10, 900000))
    output_path = f"./mediafiles/images/{image_new_name}.jpg"
    saved_path = f"/images/{image_new_name}.jpg"
    predicted_image = display_instances(image, results['rois'], results['masks'], 
                  results['class_ids'], CLASS_NAMES, results['scores'], save_fig_path=output_path)
    return saved_path
    # file_path = str(settings.MEDIA_ROOT) + "/" + str(predicted_image) + ".png"
    # new_predicted_image_path = cv2.imwrite(file_path, predicted_image)
    # return new_predicted_image_path
    
        
