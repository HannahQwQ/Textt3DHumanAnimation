import json
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from projects.HumanGaussian.临时脚本收集.animation_with_semantic_label import Skeleton  

"""
Usage:
python attach_semantic_to_gaussians.py \
    --input old.ply \
    --output new_semantic.ply \
    --smplx_model path/to/smplx_model \
    --semantic_file smplx_vert_segmentation.json
"""

def load_ply_xyz(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    return xyz, ply


def save_ply_with_semantics_from_skeleton(opt, semantics, out_path):

    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(opt.ply)
    vertex = ply['vertex']
    N = len(vertex)

    assert N == len(semantics), "Semantic length mismatch"

    # --- 原字段类型描述 ---
    old_dtype = vertex.data.dtype.descr

    print("\n=== 原 PLY 字段列表 ===")
    for name, dtype in old_dtype:
        print(f" - {name}: {dtype}")

    # --- 新字段类型描述 ---
    new_dtype = old_dtype + [('semantic', 'i4')]

    # --- 创建新数据容器 ---
    new_data = np.empty(N, dtype=new_dtype)

    # 复制旧字段
    for name, _ in old_dtype:
        new_data[name] = vertex[name]

    # 添加 semantic 字段
    new_data['semantic'] = semantics

    # 重新创建 ply 元素
    new_vertex_el = PlyElement.describe(new_data, 'vertex')

    # --- 新字段类型描述 ---
    new_dtype = new_vertex_el.data.dtype.descr

    print("\n=== 新 PLY 字段列表（生成后） ===")
    for name, dtype in new_dtype:
        print(f" - {name}: {dtype}")

    # 保存
    PlyData([new_vertex_el], text=False).write(out_path)
    print(f"\n[OK] Saved PLY with semantic → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--smplx_model', type=str)
    parser.add_argument('--semantic_file', type=str)
    args = parser.parse_args()

    # ------------------------------
    # Load original Gaussian PLY
    # ------------------------------
    xyz, ply = load_ply_xyz(args.input)
    print(f"Loaded Gaussian PLY ({xyz.shape[0]} points)")
    N = xyz.shape[0]

    # -------------------------------------------
    # Initialize SMPL-X Skeleton (must pass opt.ply)
    # -------------------------------------------
    class FakeOpt:
        def __init__(self, ply):
            self.ply = ply
            self.motion = None

    opt = FakeOpt(args.input)

    # -------------------------------------------
    # Initialize SMPL-X Skeleton & load smplx mesh
    # -------------------------------------------
    sk = Skeleton(opt=opt)        # 现在 Skeleton 有 opt.ply 就不会报错
    sk.load_smplx(args.smplx_model)

    # Now we have:
    # sk.vertices (10475,3)
    # sk.faces (20908,3)
    # sk.mapping_face (N_gaussians,)
    # sk.mapping_uvw  (N_gaussians, 3)

    # 检查赋值正确
    # 定义需要检查的属性
    attrs = ["vertices", "faces", "mapping_face", "mapping_uvw"]

    for attr in attrs:
        value = getattr(sk, attr, None)
        if value is None:
            print(f"{attr} is None!")
        elif len(value) == 0:
            print(f"{attr} is empty!")
        else:
            # 打印第一个元素/行以确认
            print(f"{attr} first element:\n{value[0]}")

    # -------------------------------------------
    # Load SMPL-X vertex segmentation (27-part)
    # -------------------------------------------
    with open(args.semantic_file, 'r') as f:
        seg = json.load(f)

    # Build vertex → semantic id map
    vert2label = np.full((sk.vertices.shape[0],), -1, dtype=np.int32)

    part_names = sorted(seg.keys())   # ensure stable 0~26 mapping
    print("Semantic parts:", part_names)

    for sid, pname in enumerate(part_names):
        for vid in seg[pname]:
            vert2label[vid] = sid

    # -------------------------------------------
    # Compute Gaussian semantics from mapping_face
    # -------------------------------------------
    # skeleton.load_smplx 时的 mask（你需要在 Skeleton 里把它保存成 self.mask）
    mask = sk.mask     # True = 被保留；False = 被删掉
    semantics = np.full((N,), -1, dtype=np.int32)

    # 找到 mask 之后的点在原始点集中的下标
    original_indices = np.where(mask)[0]

    # 遍历 mask 后的每个高斯点，计算语义标签
    for new_idx, orig_idx in enumerate(original_indices):
        face_id = sk.mapping_face[new_idx]
        v0, v1, v2 = sk.faces[face_id]

        uvw = sk.mapping_uvw[new_idx]
        bary_idx = np.argmax(uvw)
        vert_id = [v0, v1, v2][bary_idx]

        semantics[orig_idx] = vert2label[vert_id]

    print("Final semantics shape:", semantics.shape)
    print("Num of missing (-1) labels:", np.sum(semantics == -1))

    # 统计 label=12 的高斯点数量，并输出信息
    label_of_interest = 12
    count = 0

    # print(f"==== DEBUG MAPPING (label={label_of_interest} points) ====")

    # for new_idx, orig_idx in enumerate(original_indices):
    #     face_id = sk.mapping_face[new_idx]
    #     uvw = sk.mapping_uvw[new_idx]
    #     v0, v1, v2 = sk.faces[face_id]
    #     bary_idx = np.argmax(uvw)
    #     vert_id = [v0, v1, v2][bary_idx]
    #     label = vert2label[vert_id]

    #     if label == label_of_interest:
    #         count += 1
    #         # print(f"[Gaussian #{orig_idx}]")
    #         # print(f"  xyz = {xyz[orig_idx]}")
    #         # print(f"  face_id = {face_id}")
    #         # print(f"  face vertices = ({v0},{v1},{v2})")
    #         # print(f"  uvw = {uvw}")
    #         # print(f"  chosen vert = {vert_id}")
    #         # print(f"  semantic label = {label}")
    #         # print()

    # print(f"Total number of Gaussians with label {label_of_interest}: {count}")
    unique, counts_unique = np.unique(semantics, return_counts=True)
    for u, c in zip(unique, counts_unique):
        print(f"Label {u}: {c} points")

    # -------------------------------------------
    # Save new semantic-labeled PLY
    # -------------------------------------------
    save_ply_with_semantics_from_skeleton(opt, semantics, args.output)


if __name__ == "__main__":
    main()
