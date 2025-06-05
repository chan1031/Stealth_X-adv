import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 주로 3D 플롯을 위해 import
import re

def load_obj(filepath):
    """
    간단한 .obj 파서 예시:
      - v (vertex)만 파싱
      - f (face) 중 삼각형만 취급 (세 개의 정점)
      - vn, vt 등은 별도 파싱 없이 무시
    """
    vertices = []
    faces = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 주석이나 공백 행은 무시
            if line.startswith('#') or not line.strip():
                continue
            
            # 정점 정보 (v)
            if line.startswith('v '):
                # "v x y z" 형태
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                vertices.append((x, y, z))
            
            # 면 정보 (f)
            elif line.startswith('f '):
                # f 11033//10506 11021//10495 11019//10493 와 같은 형태
                # vertex_index/texture_index/normal_index 로 구성되어 있으므로
                # vertex_index 부분만 가져온다.
                # 공백으로 나눈 뒤 각 조각의 맨 앞 숫자만 파싱
                parts = line.strip().split()[1:]  # ['11033//10506', '11021//10495', '11019//10493']
                
                # 삼각형이므로 3개의 정점만 있다고 가정
                face_vertex_indices = []
                for part in parts:
                    # 예: '11033//10506' -> '11033'
                    # 슬래시로 분리 후 첫 번째 부분(정점 인덱스)만 int로 변환
                    v_idx = part.split('/')[0]
                    v_idx = int(v_idx)
                    face_vertex_indices.append(v_idx)
                
                # OBJ는 인덱스가 1부터 시작하므로 파이썬 인덱스에 맞게 0부터 시작하도록 조정
                face_vertex_indices = [idx - 1 for idx in face_vertex_indices]
                faces.append(face_vertex_indices)
    
    return vertices, faces

def plot_obj(vertices, faces):
    """
    matplotlib 3D를 사용하여 정점/삼각형 정보를 시각화.
    """
    # x, y, z 좌표 따로 분리
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # faces 리스트의 각 face를 triangulation에 맞는 형태로 전달
    # plot_trisurf에 triangles=faces로 넘기면, faces 내에 삼각 인덱스가 있어야 함
    ax.plot_trisurf(x, y, z, triangles=faces, edgecolor='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("OBJ Mesh Visualization")
    plt.show()

if __name__ == "__main__":
    # 예시: 현재 폴더에 "mesh.obj" 라는 파일이 있다고 가정
    filepath = "/home/skku/Desktop/Adversarial_Example/X-adv/objs/simple_door_key2.obj"
    vertices, faces = load_obj(filepath)
    plot_obj(vertices, faces)
