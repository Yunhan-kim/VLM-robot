# VLM-Robot

## Prerequisites

### Step 1: Obtain API Keys for Large Language Models

**OpenAI**

1. **Create the `openai_keys` Folder:**
   - Navigate to your project directory and create a folder named `openai_keys`.

2. **Create the `openai_key.json` File:**
   - Inside the `openai_keys` folder, create a file named `openai_key.json`.

3. **Fill in the API Key:**
   - Open the `openai_key.json` file and add your OpenAI API key in the following format:

     ```json
     {
         "key": "YOUR_OPENAI_API_KEY",
         "org": "YOUR_OPENAI_ORG_ID",
         "proxy": "YOUR_PROXY_URL"
     }
     ```

   - Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.
   - Replace `YOUR_OPENAI_ORG_ID` with your OpenAI organization ID.
   - Replace `YOUR_PROXY_URL` with your proxy URL if applicable.

**Gemini**

1. **Create the `gemini_keys` Folder:**
   - Navigate to your project directory and create a folder named `gemini_keys`.

2. **Create the `gemini_keys.json` File:**
   - Inside the `gemini_keys` folder, create a file named `gemini_keys.json`.

3. **Fill in the API Key:**
   - Open the `gemini_keys.json` file and add your Gemini API key in the following format:

     ```json
     {
         "key": "YOUR_GEMINI_API_KEY"
     }
     ```

   - Replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key.

**Note:** API keys are sensitive information. Ensure they are stored securely and not exposed in public repositories.

### Step 2: Download the Pretrained Weights

In this step, you'll download the pretrained weights for the Grounding DINO model to initialize it properly.

**Files to Download:**

- `groundingdino_swinb_cogcoor.pth`: The pretrained weights for the Swin-B based version of the Grounding DINO model.
- `GroundingDINO_SwinB_cfg.py`: The configuration file defining the model's settings and hyperparameters.
- `sam_vit_h_4b8939.pth`: The pretrained weights for the Segment Anything Model (SAM) using the ViT-H architecture.

**Download Instructions:**

1. **Create the Checkpoints Directory:**
   ```bash
   mkdir -p checkpoints
   ```

2. **Download the Weight Files:**
    ```bash
     cd checkpoints

     wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
     wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/heads/main/groundingdino/config/GroundingDINO_SwinB_cfg.py
     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
     ```

### Step 3: Camera Calibration

1. **Prepare the Checkerboard:**
   - Ensure the checkerboard pattern is suitable for camera calibration.
   - Position the checkerboard in a well-lit area to avoid distortion and ensure clear visibility.

2. **Capture Images:**
   - Take multiple images of the checkerboard from different angles and distances.
   - It is recommended to capture at least 10 images.
   - Ensure the checkerboard pattern is clearly visible in each image.

3. **Save Images:**
   - Save the captured images in a specific directory.
   - Example: `./calibration_images/`
   - Ensure the save path matches the one used in the `calibration.py` script.

4. **Run the Python Script:**
   - Use the saved images to perform camera calibration by running the following command:

     ```bash
     python utils/calibration.py
     ```

## Acknowledgements

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [YOLO World](https://github.com/AILab-CVC/YOLO-World)

## Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{Cheng2024YOLOWorld,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  booktitle={Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```