import numpy as np

def scale_obj(input_file, output_file, scale_factor=1.2):
    """
    OBJ 파일의 크기를 조절하는 함수
    
    Args:
        input_file (str): 입력 OBJ 파일 경로
        output_file (str): 출력 OBJ 파일 경로
        scale_factor (float): 크기 조절 비율 (기본값: 1.2 = 20% 증가)
    """
    vertices = []
    faces = []
    
    # OBJ 파일 읽기
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue  # 주석 건너뛰기
            values = line.split()
            if not values: continue
            
            if values[0] == 'v':  # 정점
                v = [float(x) for x in values[1:4]]
                vertices.append(v)
            elif values[0] == 'f':  # 면
                face = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                faces.append(face)
    
    # 정점 크기 조절
    vertices = np.array(vertices)
    vertices = vertices * scale_factor
    
    # 새로운 OBJ 파일 저장
    with open(output_file, 'w') as f:
        # 정점 저장
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # 면 저장
        for face in faces:
            f.write("f")
            for v in face:
                f.write(f" {v}")
            f.write("\n")

def scale_txt_points(input_file, output_file, scale_factor=1.2):
    """
    TXT 파일의 정점들을 크기 조절하는 함수
    """
    points = []
    
    # TXT 파일 읽기
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            tokens = line.strip().split()
            # 'v'가 있으면 제거
            if tokens[0] == 'v':
                tokens = tokens[1:]
            try:
                x, y, z = map(float, tokens)
                points.append([x, y, z])
            except ValueError:
                continue
    
    # 정점 크기 조절
    points = np.array(points)
    points = points * scale_factor
    
    # 새로운 TXT 파일 저장 (출력에도 'v'를 붙임)
    with open(output_file, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")


if __name__ == "__main__":
    # OBJ 파일 처리
    obj_input_file = "/home/skku/Desktop/Adversarial_Example/X-adv/objs/simple_door_key2.obj"  # 입력 OBJ 파일 경로
    obj_output_file = "/home/skku/Desktop/Adversarial_Example/X-adv/objs/simple_door_key_scaled.obj"  # 출력 OBJ 파일 경로
    
    # TXT 파일 처리
    txt_input_file = "groove_points.txt"  # 입력 TXT 파일 경로
    txt_output_file = "groove_points_scaled.txt"  # 출력 TXT 파일 경로
    
    scale_factor = 1.4  # 40% 크기 증가
    
    # OBJ 파일 크기 조절
    scale_obj(obj_input_file, obj_output_file, scale_factor)
    print(f"OBJ 파일 크기가 {scale_factor}배로 조절되었습니다.")
    
    # TXT 파일 크기 조절
    scale_txt_points(txt_input_file, txt_output_file, scale_factor)
    print(f"TXT 파일의 정점들이 {scale_factor}배로 조절되었습니다.") 