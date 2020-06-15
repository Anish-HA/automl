import subprocess
import os

toplvl_dir = "/home/anish-ha/Documents/obj-det/workspace/dets/models/v1.1_all/data/images/split_by_size"
dir_list = next(os.walk(toplvl_dir))[1]

max_infer_dim = 768

det_dict = {}

for size in dir_list:
    print(size)
    img_dims = (int(size.split('x')[0]), int(size.split('x')[1]))
    img_width = img_dims[0]
    img_height = img_dims[1]
    max_dim = max(img_height, img_width)
    scale = max_infer_dim / max_dim
    image_size = str(int(img_height * scale)) + "x" + str(int(img_width * scale))
    print(image_size)
    
    subprocess.check_call([
        'python3', 'model_inspect.py',
        '--runmode=model_infer_estimator',
        '--model_name=efficientdet-d0',
        '--ckpt_path=/home/anish-ha/Documents/obj-det/workspace/dets/models/v1.1_all/model_dir/efficientdet-d0_9-class-1/archive',
        '--hparams=num_classes=9,moving_average_decay=0,image_size=' + image_size + ',label_id_mapping=/home/anish-ha/Documents/obj-det/workspace/dets/models/v1.1_all/data/label_map_9-class.json',
        '--input_image=' + toplvl_dir + '/' + size + '/*.*',
        '--output_image_dir=' + toplvl_dir + '/' + size,
        '--min_score_thresh=0.4'
    ])