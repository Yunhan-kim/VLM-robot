import os
import json
import yaml

import numpy as np
import cv2

from pathlib import Path

from utils.llm_util import LLMBase
from utils.io_util import load_txt

class VLMPlanner(LLMBase):
    """LLM TAMP Planner"""

    def __init__(
        self,
        planner_prompt_file: str,
        env_desc: str,        
    ):
        
        LLMBase.__init__(self)

        # load planning prompt template
        prompt_template_folder = Path(__file__).resolve().parent.parent / "prompts"
        planning_prompt_template = load_txt(
            prompt_template_folder / f"{planner_prompt_file}.txt"
        )

        # load problem file
        domain_desc = self._domain_desc(env_desc)
        self._planning_prompt = planning_prompt_template.replace("{domain_desc}", domain_desc)        
    
    def _domain_desc(self, env_desc: str):
        domain_desc = ''
        for i in range(len(env_desc)):
            target_obj = env_desc[i][0]
            target_obb = env_desc[i][3]            
            target_obb_cal = self.calibration(list(target_obb[0][0]), z = 530.0)  # **** z (mm): 캘리브레이션 할 때마다 확인 **** #
            target_size = target_obb[0][1]
            target_angle = target_obb[0][2]            
            if target_size[0] >= target_size[1]:
                target_angle = -(90 - target_angle)
            target_angle = target_angle - 135 # **** Ready pose에서의 그리퍼 회전 각도 확인 **** #
            domain_desc += '   - **{}**: Position {} (x, y), Angle {}\n'.format(target_obj, str(target_obb_cal[0:2].tolist()), target_angle)
        return domain_desc            

    def calibration(self, img_point: list, z = 470.0): # 원하는 월드 좌표의 z값 (평면인 경우 고정, mm)
        cal_path = 'utils/camera_calibration.yaml'
        assert os.path.exists(cal_path), f"File does not exist: {cal_path}"

        with open(cal_path, "r") as file:
            cal_params = yaml.safe_load(file)
        
        # 카메라 매트릭스
        intrinsic = np.array(cal_params['camera_matrix'])

        # 왜곡 계수 (여기서는 보정된 값 사용)
        dist = np.array(cal_params['dist_coeffs'])

        img_points_2D = np.array([img_point], dtype=np.float32)

        # 1. 왜곡 보정 및 카메라 좌표로 변환
        undistorted = cv2.undistortPoints(
            img_points_2D.reshape(-1, 1, 2), intrinsic, dist
        )

        # 2. 회전 벡터를 회전 행렬로 변환
        idx = 0
        R, _ = cv2.Rodrigues(np.array(cal_params["rvecs"][idx]))
        t = np.array(cal_params["tvecs"][idx]).flatten()

        # 3. 역변환 계산
        obj_points_3D = []
        for point in undistorted:
            # 카메라 좌표
            x, y = point[0]            
            cam_coords = np.array([x * z, y * z, z], dtype=np.float32)

            # 월드 좌표
            world_coords = np.dot(np.linalg.inv(R), cam_coords - t)
            obj_points_3D.append(world_coords)

        obj_points_3D = np.array(obj_points_3D)

        # 4. 로봇 베이스 기준 좌표 이동 # **** 캘리브레이션 할 때마다 확인 **** #
        # obj_target = (np.array([[0, 1, 0],[1, 0, 0],[0 ,0, 1]]) @ (obj_points_3D - np.array([320,380,0])).T).T
        obj_target = (np.array([[0, 1, 0],[1, 0, 0],[0 ,0, 1]]) @ (obj_points_3D - np.array([410,300,0])).T).T
        obj_target = obj_target[0] / 1000 # (mm) -> (m)
        
        return obj_target

    def plan(self, question: str):
        # plan
        planning_prompt = self._planning_prompt        
        planning_prompt = planning_prompt.replace("{question}", question)
        
        plan, reasoning = None, None

        llm_output = self.prompt_llm(planning_prompt)                
        llm_output = json.loads(llm_output.strip('```json\n').strip('```'))

        plan = llm_output['Full Plan']
        reasoning = llm_output["Reasoning"]

        return plan, reasoning