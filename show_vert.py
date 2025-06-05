import matplotlib.pyplot as plt
import numpy as np

# OBJ 파일에서 v 좌표 읽기
def read_obj_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
    return vertices

# 좌표 시각화
def plot_vertices(vertices, scale_factor=2.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = zip(*vertices)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    # 중앙을 기준으로 확대
    x_center, y_center, z_center = np.mean(xs), np.mean(ys), np.mean(zs)
    xs = (xs - x_center) * scale_factor + x_center
    ys = (ys - y_center) * scale_factor + y_center
    zs = (zs - z_center) * scale_factor + z_center

    ax.scatter(xs, ys, zs, c='blue', marker='o')

    # 축 범위를 자동으로 설정 + 여유 공간 추가
    ax.set_xlim(xs.min() - 1, xs.max() + 1)
    ax.set_ylim(ys.min() - 1, ys.max() + 1)
    ax.set_zlim(zs.min() - 1, zs.max() + 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('output.jpg')
    plt.show()

# 메인 함수
def main():
    file_path = 'input.txt'  # OBJ 형식의 v 좌표가 있는 텍스트 파일 경로
    vertices = read_obj_vertices(file_path)
    plot_vertices(vertices, scale_factor=3.0)  # 숫자를 키우면 더 크게 확대됨

if __name__ == '__main__':
    main()
