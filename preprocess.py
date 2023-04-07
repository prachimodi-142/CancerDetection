OPENSLIDE_PATH = r'D:\Cancer Metastasis Classifier\openslide-python\openslide-win64-20221217\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
    
from openslide import OpenSlide
from deephistopath.wsi import slide
from deephistopath.wsi import util
from deephistopath.wsi import tiles
from deephistopath.wsi import filter

print(slide.get_num_training_slides())

slide.slide_stats()

# img_path = slide.get_training_image_path(1)

slide.singleprocess_training_slides_to_images()

filter.singleprocess_apply_filters_to_images()

tiles.singleprocess_filtered_images_to_tiles()
