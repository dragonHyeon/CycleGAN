import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from Lib import UtilLib
from Common import ConstVar


class VanGogh2PhotoDataset(Dataset):
    def __init__(self, data_dir, mode_train_test):
        """
        * FACADESDataset 데이터로더
        :param data_dir: 데이터 디렉터리
        :param mode_train_test: 학습 / 테스트 모드
        """

        # 데이터 해당 디렉터리
        self.data_dir = data_dir
        # 학습 / 테스트 모드
        self.mode_train_test = mode_train_test

        # a 데이터 디렉터리
        self.dir_a = UtilLib.getNewPath(path=self.data_dir,
                                        add='a')
        # b 데이터 디렉터리
        self.dir_b = UtilLib.getNewPath(path=self.data_dir,
                                        add='b')

        # a 파일 경로 모음
        self.files_a = [UtilLib.getNewPath(path=self.dir_a,
                                           add=filename)
                        for filename in os.listdir(self.dir_a)]
        # b 파일 경로 모음
        self.files_b = [UtilLib.getNewPath(path=self.dir_b,
                                           add=filename)
                        for filename in os.listdir(self.dir_b)]

        # 모드에 따른 데이터 전처리 방법
        self.transform = {
            ConstVar.MODE_TRAIN: transforms.Compose([
                transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ]),
            ConstVar.MODE_TEST: transforms.Compose([
                transforms.Resize(size=(ConstVar.RESIZE_SIZE, ConstVar.RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ])
        }

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, item):

        # a 데이터 인덱스
        a_index = item % len(self.files_a)
        # b 데이터 인덱스
        b_index = item % len(self.files_b)

        # a 데이터
        a = self.transform[self.mode_train_test](Image.open(fp=self.files_a[a_index]))
        # b 데이터
        b = self.transform[self.mode_train_test](Image.open(fp=self.files_b[b_index]))

        return a, b
