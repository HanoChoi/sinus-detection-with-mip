# sinus-detection-with-mip  

dcm 파일들을 넣으면 Minimum Intensity Projection을 3방향으로 진행하여 부비동과 비인두의 위치를 바운딩 박스로 처리한 nii파일을 돌려준다(골격만 있는 박스 or 박스 선택 가능)  

Requirements  
- python 3.6.8/3.6.9
- tensorflow 2.3.1
- Medpy
- Pillow
- Matploblib
- Numpy
  

사용법  
- python [id] [line_width] [is_skeleton_or_box]
- id(str) : 저장할 이름(숫자/영어) ex) 1111 -> 1111_skeleton.nii or 1111_box.nii
- line_width(int) : 선분 두께(px)
- is_skeleton_or_box : True -> 골격만 있는 상자, False -> 그냥 상자로 각 타겟들 표시
- ex) python main.py 10438532 5 False

현재 12/4 이후 main2.py를 사용하는 새로운 모델이 있으나 용량이 너무 커서 업로드가 되지 않고있음.
