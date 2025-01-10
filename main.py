import os, sys
sys.path.append(os.path.join(os.getcwd(), "segmentation_anything"))

from utils.vision_util import GroundingDINO_Vision
# from utils.vision_util import YOLOWorld_Vision
from planners.planners import VLMPlanner

def main():
    image_path = 'assets/test_2.png'

    vision_base = GroundingDINO_Vision(device = 'cuda')    
    obb_results = vision_base.obb_predict(image_path, text_prompt = "yellow.green.red.")
    # vision_base = YOLOWorld_Vision(device = 'cuda')    
    # obb_results = vision_base.obb_predict(image_input, text_prompt = ["yellow", "green", "red"])

    # b = [['yellow', 0.8888310790061951, [2024.3189697265625, 800.58837890625, 2518.831298828125, 1355.8414306640625], [((2271.23876953125, 1085.838134765625), (522.2755126953125, 307.27642822265625), 62.840206146240234)]], ['red', 0.7884427905082703, [1533.3343505859375, 196.00796508789062, 2076.036376953125, 690.5823364257812], [((1795.741943359375, 439.44207763671875), (263.4681701660156, 536.7791137695312), 55.35905075073242)]], ['green', 0.7970165610313416, [718.8109741210938, 1089.64990234375, 1253.7325439453125, 1385.6641845703125], [((987.5362548828125, 1235.5908203125), (520.7658081054688, 258.0560607910156), 4.28381872177124)]]]    
    planner = VLMPlanner(
        planner_prompt_file = "template",
        env_desc = obb_results
    )
        
    question = input("###########################################\n###########################################\nQuestion: ")
    plan, reasoning = planner.plan(question)
    print(plan)
    print(reasoning)
    


if __name__ == "__main__":    
    main()