import matplotlib
matplotlib.use('Agg')  # X 포워딩 없이 이미지 저장을 위해 'Agg' 백엔드 사용
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def load_obj(filepath):
    """
    간단한 .obj 파서:
    - v (vertex)만 파싱
    - f (face) 중 삼각형만 취급
    - vn, vt 등은 별도 파싱 없이 무시
    """
    vertices = []
    faces = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            # 정점 정보 (v)
            if line.startswith('v '):
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                vertices.append((x, y, z))
            
            # 면 정보 (f)
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face_vertex_indices = []
                for part in parts:
                    # 예: '11033//10506' -> '11033' (슬래시로 분리 후 첫 번째 인덱스)
                    v_idx = part.split('/')[0]
                    v_idx = int(v_idx)
                    face_vertex_indices.append(v_idx - 1)  # 1-based -> 0-based 변환
                faces.append(face_vertex_indices)
    
    return vertices, faces

def plot_obj(vertices, faces, output_filename="mesh.jpg"):
    """
    matplotlib 3D를 사용하여 정점/삼각형 정보를 시각화 후 .jpg 파일로 저장.
    """
    # x, y, z 좌표 분리
    x = [v[0] for v in vertices]
    y = [v[1] for v in vertices]
    z = [v[2] for v in vertices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # trisurf 사용 (faces에 삼각형 인덱스가 들어있어야 함)
    ax.plot_trisurf(x, y, z, triangles=faces, edgecolor='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("OBJ Mesh Visualization")

    # SSH 환경에서 X 포워딩 없이 바로 이미지로 저장
    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    filepath = "/home/skku/Desktop/Adversarial_Example/X-adv/objs/simple_door_key2.obj"  # 예시 .obj 파일 경로
    vertices, faces = load_obj(filepath)
    plot_obj(vertices, faces, output_filename="mesh.jpg")
