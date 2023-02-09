import torch
import numpy as np
from tqdm import tqdm

from Common import ConstVar
from DeepLearning import utils


class Tester:
    def __init__(self, modelG, modelD, metric_fn_BCE, metric_fn_L1, test_dataloader, device):
        """
        * 테스트 관련 클래스
        :param modelG: 테스트 할 모델. 생성자
        :param modelD: 테스트 할 모델. 판별자
        :param metric_fn_BCE: 학습 성능 체크하기 위한 metric (BCE loss)
        :param metric_fn_L1: 학습 성능 체크하기 위한 metric (L1 loss)
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.modelG = modelG
        self.modelD = modelD
        # 학습 성능 체크하기 위한 metric
        self.metric_fn_BCE = metric_fn_BCE
        self.metric_fn_L1 = metric_fn_L1
        # 테스트용 데이터로더
        self.test_dataloader = test_dataloader
        # GPU / CPU
        self.device = device

    def running(self, checkpoint_file=None):
        """
        * 테스트 셋팅 및 진행
        :param checkpoint_file: 불러올 체크포인트 파일
        :return: 테스트 수행됨
        """

        # 불러올 체크포인트 파일 있을 경우 불러오기
        if checkpoint_file:
            state = utils.load_checkpoint(filepath=checkpoint_file)
            self.modelG.load_state_dict(state[ConstVar.KEY_STATE_MODEL_G])
            self.modelD.load_state_dict(state[ConstVar.KEY_STATE_MODEL_D])

        # 테스트 진행
        self._test()

    def _test(self):
        """
        * 테스트 진행
        :return: 이미지 생성 및 score 기록
        """

        # 각 모델을 테스트 모드로 전환
        self.modelG.eval()
        self.modelD.eval()

        # 배치 마다의 G / D loss 담을 리스트
        batch_loss_listG = list()
        batch_loss_listD = list()

        # 생성된 이미지 담을 리스트
        self.pics_list = list()

        # a shape: (N, 3, 224, 224)
        # b shape: (N, 3, 224, 224)
        for a, b in tqdm(self.test_dataloader, desc='test dataloader', leave=False):

            # 현재 배치 사이즈
            batch_size = a.shape[0]
            # 패치 한 개 사이즈
            patch_size = (1, int(a.shape[2] / 16), int(a.shape[3] / 16))

            # real image label
            real_label = torch.ones(size=(batch_size, *patch_size), device=self.device)
            # fake image label
            fake_label = torch.zeros(size=(batch_size, *patch_size), device=self.device)

            # 각 텐서를 해당 디바이스로 이동
            a = a.to(self.device)
            b = b.to(self.device)

            # 판별자 real image 순전파
            output = self.modelD(b, a)
            scoreD_real = self.metric_fn_BCE(output=output,
                                             label=real_label)
            # 판별자 fake image 순전파
            fake_b = self.modelG(a)
            output = self.modelD(fake_b, a)
            scoreD_fake = self.metric_fn_BCE(output=output,
                                             label=fake_label)
            # 배치 마다의 D loss 계산
            scoreD = scoreD_real + scoreD_fake
            batch_loss_listD.append(scoreD)

            # 생성자 순전파
            fake_b = self.modelG(a)
            output = self.modelD(fake_b, a)
            scoreG_BCE = self.metric_fn_BCE(output=output,
                                            label=real_label)
            # L1 loss 계산
            scoreG_L1 = self.metric_fn_L1(fake_b=fake_b,
                                          b=b)
            # 배치 마다의 생성자 G loss 계산
            scoreG = scoreG_BCE + ConstVar.LAMBDA * scoreG_L1
            batch_loss_listG.append(scoreG)

            # a, b, fake_b 이미지 쌍 담기 (설정한 개수 만큼)
            if len(self.pics_list) < ConstVar.NUM_PICS_LIST:
                self.pics_list.append((a, b, fake_b))

        # score 기록
        self.score = {
            ConstVar.KEY_SCORE_G: np.mean(batch_loss_listG),
            ConstVar.KEY_SCORE_D: np.mean(batch_loss_listD)
        }