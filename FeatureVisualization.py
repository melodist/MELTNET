import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import NetworkKeras
from Extraction import PatchExtraction

tf.enable_eager_execution()


def load_image(img_path, target_size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)

    # 배치 사이즈 추가 + 스케일링 결과 반환
    return img_tensor[np.newaxis] / 255  # (1, 150, 150, 3)


def show_patch(patches):
    patches = patches.numpy()
    plt.figure(figsize=(16, 8))
    for i in range(32):
        patch = patches[i, :]
        img = np.reshape(patch, (17, 17))
        # cv2.imwrite(f'./patch_{i}.png', img*255)

        plt.subplot(4, 8, i + 1)
        # 눈금 제거. fignum은 같은 figure에 연속 출력
        plt.axis('off')
        plt.matshow(img, cmap='gray', fignum=0, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


# 첫 번째 등장하는 컨볼루션 레이어의 모든 피처맵(32개) 출력
def show_first_feature_map(loaded_model, img_path):
    first_output = loaded_model.layers[0].output
    print(first_output.shape, first_output.dtype)  # (?, 148, 148, 32) <dtype: 'float32'>

    # 1개의 출력을 갖는 새로운 모델 생성
    model = tf.keras.models.Model(inputs=loaded_model.input, outputs=first_output)

    # 입력으로부터 높이와 너비를 사용해서 target_size에 해당하는 튜플 생성
    target_size = (loaded_model.input.shape[1], loaded_model.input.shape[2])
    img_tensor = load_image(img_path, target_size)

    print(loaded_model.input.shape)  # (?, 150, 150, 3)
    print(img_tensor.shape)  # (1, 150, 150, 3)

    first_activation = model.predict(img_tensor)

    # 컨볼루션 레이어에서 필터 크기(3), 스트라이드(1), 패딩(valid)을 사용했기 때문에
    # 150에서 148로 크기가 일부 줄었음을 알 수 있다. 필터 개수는 32.
    print(first_activation.shape)  # (1, 148, 148, 32)
    print(first_activation[0, 0, 0])  # [0.00675746 0. 0.02397328 0.03818807 0. ...]

    # 19번째 활성 맵 출력. 기본 cmap은 viridis. gray는 흑백 컬러맵.
    # [0, :, :, feature_index]
    # 0은 첫 번째 데이터(원본 이미지)의 피처맵을 가리킨다. 사진은 1장만 사용했기 때문에 0만 가능
    # 가운데 콜론(:)은 높이와 너비를 가리키는 차원의 모든 데이터
    # feature_index는 보고 싶은 피처맵이 있는 채널을 가리킨다.
    # 32개의 필터를 사용했다면 0부터 31까지의 피처맵이 존재한다.
    plt.figure(figsize=(16, 8))
    for i in range(first_activation.shape[-1]):
        plt.subplot(4, 8, i + 1)

        # 눈금 제거. fignum은 같은 피켜에 연속 출력
        plt.axis('off')
        plt.matshow(first_activation[0, :, :, i], cmap='gray', fignum=0)
    plt.tight_layout()
    plt.show()


path_model = './model/20191104_111218/'
path_patient = './Test/00237841 LEE TAE SOOK'
path_result = ''

ind_CT = [[230, 380], [150, 370]]
ind_PT = [[230, 380], [150, 370]]

img_CT, img_PT = PatchExtraction.stackImages(path_patient, ind_CT, ind_PT)
patches_CT, patches_PT = PatchExtraction.patch_extraction_thres(img_CT, img_PT, 80)

show_patch(patches_CT)

# input_shape = (17, 17)
# embedding_size = 10
# trained_model = NetworkKeras.create_base_network(input_shape, embedding_size)
# trained_model.load_weights(path_model)
# show_first_feature_map(trained_model, path_result)
