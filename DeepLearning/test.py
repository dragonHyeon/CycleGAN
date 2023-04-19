import torch
import numpy as np
from tqdm import tqdm

from Common import ConstVar
from DeepLearning import utils


class Tester:
    def __init__(self, G_AB, G_BA, D_A, D_B, metric_fn_GAN, metric_fn_cycle, metric_fn_identity, test_dataloader, device):
        """
        * 테스트 관련 클래스
        :param G_AB: 테스트 할 모델. 생성자
        :param G_BA: 테스트 할 모델. 생성자
        :param D_A: 테스트 할 모델. 판별자
        :param D_B: 테스트 할 모델. 판별자
        :param metric_fn_GAN: 학습 성능 체크하기 위한 metric (GAN loss)
        :param metric_fn_cycle: 학습 성능 체크하기 위한 metric (cycle loss)
        :param metric_fn_identity: 학습 성능 체크하기 위한 metric (identity loss)
        :param test_dataloader: 테스트용 데이터로더
        :param device: GPU / CPU
        """

        # 테스트 할 모델
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B
        # 학습 성능 체크하기 위한 metric
        self.metric_fn_GAN = metric_fn_GAN
        self.metric_fn_cycle = metric_fn_cycle
        self.metric_fn_identity = metric_fn_identity
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
            self.G_AB.load_state_dict(state[ConstVar.KEY_STATE_G_AB])
            self.G_BA.load_state_dict(state[ConstVar.KEY_STATE_G_BA])
            self.D_A.load_state_dict(state[ConstVar.KEY_STATE_D_A])
            self.D_B.load_state_dict(state[ConstVar.KEY_STATE_D_B])

        # 테스트 진행
        self._test()

    def _test(self):
        """
        * 테스트 진행
        :return: 이미지 생성 및 score 기록
        """

        # 각 모델을 테스트 모드로 전환
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()

        # 배치 마다의 G / D score 담을 리스트
        batch_score_listG = list()
        batch_score_listD = list()

        # 생성된 이미지 담을 리스트
        self.pics_list = list()

        # a shape: (N, 3, 256, 256)
        # b shape: (N, 3, 256, 256)
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

            # -----------------
            #  Test Generators
            # -----------------

            # GAN loss
            fake_b = self.G_AB(a)
            score_G_GAN_AB = self.metric_fn_GAN(self.D_B(fake_b), real_label)
            fake_a = self.G_BA(b)
            score_G_GAN_BA = self.metric_fn_GAN(self.D_A(fake_a), real_label)
            score_G_GAN = (score_G_GAN_AB + score_G_GAN_BA) / 2

            # Cycle loss
            rec_a = self.G_BA(fake_b)
            score_G_cycle_A = self.metric_fn_cycle(rec_a, a)
            rec_b = self.G_AB(fake_a)
            score_G_cycle_B = self.metric_fn_cycle(rec_b, b)
            score_G_cycle = (score_G_cycle_A + score_G_cycle_B) / 2

            # Identity loss
            score_G_identity_A = self.metric_fn_identity(self.G_BA(a), a)
            score_G_identity_B = self.metric_fn_identity(self.G_AB(b), b)
            score_G_identity = (score_G_identity_A + score_G_identity_B) / 2

            # Total loss
            score_G = score_G_GAN + ConstVar.LAMBDA_CYCLE * score_G_cycle + ConstVar.LAMBDA_IDENTITY * score_G_identity

            # 배치 마다의 생성자 G score 계산
            batch_score_listG.append(score_G)

            # ---------------------
            #  Test Discriminators
            # ---------------------

            # GAN loss
            score_D_GAN_A_real = self.metric_fn_GAN(self.D_A(a), real_label)
            score_D_GAN_A_fake = self.metric_fn_GAN(self.D_A(fake_a), fake_label)
            score_D_GAN_A = score_D_GAN_A_real + score_D_GAN_A_fake
            score_D_GAN_B_real = self.metric_fn_GAN(self.D_B(b), real_label)
            score_D_GAN_B_fake = self.metric_fn_GAN(self.D_B(fake_b), fake_label)
            score_D_GAN_B = score_D_GAN_B_real + score_D_GAN_B_fake
            score_D_GAN = (score_D_GAN_A + score_D_GAN_B) / 2

            # Total loss
            score_D = score_D_GAN

            # 배치 마다의 판별자 D score 계산
            batch_score_listD.append(score_D)

            # a, b, fake_a, fake_b, rec_a, rec_b 이미지 쌍 담기 (설정한 개수 만큼)
            if len(self.pics_list) < ConstVar.NUM_PICS_LIST:
                self.pics_list.append((a, b, fake_a, fake_b, rec_a, rec_b))

        # score 기록
        self.score = {
            ConstVar.KEY_SCORE_G: np.mean(batch_score_listG),
            ConstVar.KEY_SCORE_D: np.mean(batch_score_listD)
        }
