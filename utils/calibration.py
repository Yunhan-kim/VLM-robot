import cv2 as cv
import numpy as np
import glob
import os
import yaml

# 체스보드 패턴 크기 및 셀 크기(mm)
pattern_size = (7, 10)
square_size = 20  # mm

# 3D 객체 점 준비 (체스보드의 각 코너 위치)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 3D 및 2D 점 리스트
obj_points_3D = []  # 3D 점
img_points_2D = []  # 2D 점

# 이미지 로드 및 처리
image_filenames = glob.glob("images/*.png")
if not image_filenames:
    print("images 폴더에 이미지가 없습니다. 캘리브레이션을 중단합니다.")
    exit()

for fname in image_filenames:
    img = cv.imread(fname)
    if img is None:
        print(f"이미지를 로드하지 못했습니다: {fname}")
        continue    

    grayScale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(grayScale, pattern_size)

    if ret:
        # 코너를 정밀하게 조정
        corners_subpix = cv.cornerSubPix(
            grayScale, corners, (11, 11), (-1, -1), 
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        img_points_2D.append(corners_subpix)
        obj_points_3D.append(objp)

        # # 체스보드 검출 시각화
        # cv.drawChessboardCorners(img, pattern_size, corners_subpix, ret)
        # cv.imshow("pass", img)
        # cv.waitKey(500)
    else:
        print(f"체스보드 코너를 검출하지 못했습니다: {fname}")
        print(f"이미지 크기:{grayScale.shape}")
        # cv.imshow('fail',grayScale)
        # cv.waitKey(1000)

# cv.destroyAllWindows()

# 캘리브레이션 실행
if len(obj_points_3D) > 0 and len(img_points_2D) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
    )
    print("캘리브레이션 성공!")
    print(f"Camera matrix: \n{mtx}")
    print(f"Distortion coefficients: {dist}")

    # YAML 파일 저장
    os.makedirs("utils", exist_ok=True)
    calib_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": grayScale.shape[::-1],
        "rvecs": [rvec.tolist() for rvec in rvecs],
        "tvecs": [tvec.tolist() for tvec in tvecs],
    }

    with open("utils/camera_calibration.yaml", "w") as yaml_file:
        yaml.dump(calib_data, yaml_file, default_flow_style=False)
    print("캘리브레이션 결과가 utils/camera_calibration.yaml 파일에 저장되었습니다.")
else:
    print("캘리브레이션에 필요한 데이터가 충분하지 않습니다.")