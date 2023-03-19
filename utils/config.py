import os

# 创建文件夹
# if not os.path.exists('./img'):
#     os.mkdir('./img')

cudaNum = 0

cudaTest = 1

pattern = 'DIV2K'

batch_size = 32
patchSize = int(64)
num_epoch = 200
size = 96
featureDepth = 64
attentionDepth = 32

min_psnr = 26.5
min_ssim = 0.87
max_rmse = 0.0235

numPatch = 8

LR_G = 1e-3#1e-4
LR_D = 1e-3

generate_path = './result{}/model/generator'.format(pattern)
discriminator_path = './result{}/model/discriminator'.format(pattern)
