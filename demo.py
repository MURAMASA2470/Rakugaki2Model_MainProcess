"""
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
"""
import os
import sys
import shutil
import numpy as np
from subprocess import call
import glob

from PIL import Image, ImageFilter, ImageChops, ImageOps
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.solver import Solver
from lib.voxel import voxel2obj

if sys.version_info < (3, 0):
    raise Exception("Please follow the installation instruction on 'https://github.com/chrischoy/3D-R2N2'")


# DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/weights.npy'
# DEFAULT_WEIGHTS = r'../3D-R2N2/output/ResidualGRUNet/default_model/weights.npy'
# DEFAULT_WEIGHTS = r'C:/Users/minor/Desktop/3D-R2N2/output/ResidualGRUNet/default_model/ResidualGRUNet.npy'

DEFAULT_WEIGHTS = r'./output/ResidualGRUNet/default_model/ResidualGRUNet.npy'


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])


def load_demo_images():
    ims = []
    # 読み込む画像数を設定
    # print('現在のパス', os.getcwd())
    now_path = os.getcwd()
    # print('フォルダのファイル数を算出')
    # print("OSパス：", os.path)
    os.chdir("../3D-R2N2")
    print('変更後のパス：', os.getcwd())
    DIR = './test_image'
    image_recode = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    print("画像フォルダのファイル数：", image_recode - 1)

    imgList = glob.glob(DIR + '/*')

    for i in range(len(imgList)):
        # im = Image.open('imgs/%d.png' % i)
        im = Image.open(imgList[i])
        # print(f'ファイル名: {imgList[i]}')
        imW, imH = im.size
        # print('元画像のサイズ: 幅' + str(imW) + 'px, 縦' + str(imH) + 'px')
        # 127*127にアスペクト比を固定しつつリサイズ
        imgSize = 127
        # resizedImg = im.resize((round(im.width * imgSize / im.height), imgSize))

        resizedImg = im.resize((imgSize, imgSize))


        # resizedImg = expand2square(resizedImg, (255,255,255))
        # resizedImg = resizedImg.convert('RGBA')
        # mask.paste(resizedImg).show()
        # print(resizedImg)
        # resizedImg.show()

        # bg = Image.new("RGB", resizedImg.size, (255, 255, 255))
        # bg.paste(resizedImg, mask=resizedImg)
        # resizedImg = bg

        # resizedImg = resizedImg.filter(ImageFilter.DETAIL)
        # resizedImg = resizedImg.convert("L")

        # resizedImg.show()
        

        riW, riH = resizedImg.size
        # print('リサイズ後のサイズ: 幅' + str(riW) + 'px, 縦' + str(riH) + 'px')

        # resizedImg.filter(ImageFilter.FIND_EDGES).convert("LA").show()
        # resizedImg.filter(ImageFilter.CONTOUR).convert("LA").show()
        # resizedImg.convert("LA").crop(resizedImg.split()[-1].getbbox()).show()
        # resizedImg.filter(ImageFilter.GaussianBlur(3.0)).show()
        # resizedImg = resizedImg.filter(ImageFilter.GaussianBlur(3.0))
        # resizedImg.show()

        ims.append([np.array(resizedImg).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    os.chdir(now_path)
    return np.array(ims)

# 正方形になる様に余白を追加
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def main():
    """Main demo function"""
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    # load images
    demo_imgs = load_demo_images()

    # Download and load pretrained weights
    # print("重み付けのファイルのパス：", DEFAULT_WEIGHTS)
    download_model(DEFAULT_WEIGHTS)

    # Use the default network model
    # print("モデルのパスを読み込み...")
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass(compute_grad=False)  # instantiate a network
    net.load(DEFAULT_WEIGHTS)           # load downloaded weights
    solver = Solver(net)                # instantiate a solver

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)

    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > cfg.TEST.VOXEL_THRESH)

    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    if cmd_exists('meshlab'):
        call(['meshlab', pred_file_name])
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %
              pred_file_name)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    import time
    old = time.time()  
    main()
    loading_time = time.time() - old
    loading_time = str(loading_time).split('.')[0] + '.' + str(loading_time).split('.')[1][1]  # 小数点第一位まで
    print(loading_time, '秒')  # 3Dモデル生成にかかった時間
