import os
import to_rgb
import h5py
from PIL import Image
import numpy as np
from classification import Classification
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

classification = Classification()

result_file = 'results.csv'

path = "ssl_legacy/ssl_legacyB/file/ssl-legacysurvey_b/images_npix152_026000000_027000000.h5"
data = h5py.File(path, "r")
images = data['images']

with open(result_file, "w") as f:
    for i in range(len(images)):
    
        image_rgb = to_rgb.dr2_rgb(images[i], ['g', 'r', 'z'])
    
        image_pil = Image.fromarray((image_rgb * 255).astype(np.uint8))
    
        class_name, probability = classification.detect_image(image_pil)
    
        f.write(f"Image {i+1} -> {class_name}, Probability: {probability}\n")
     

