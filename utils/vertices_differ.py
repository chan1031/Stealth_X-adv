import sys

def load_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 정점 정보만 가져오기
                parts = line.strip().split()
                vertex = tuple(map(float, parts[1:4]))  # (x, y, z) 좌표 저장
                vertices.append(vertex)
    return vertices

def count_different_vertices(file1, file2):
    vertices1 = load_vertices(file1)
    vertices2 = load_vertices(file2)
    
    if len(vertices1) != len(vertices2):
        print("정점 개수가 다릅니다.")
        return
    
    different_count = sum(1 for v1, v2 in zip(vertices1, vertices2) if v1 != v2)
    print(f"서로 다른 좌표를 가지는 정점 개수: {different_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python compare_obj.py file1.obj file2.obj")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    count_different_vertices(file1, file2)