import os
import cv2
import math
import json
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

from kiui.cam import OrbitCamera

import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr
from gs_renderer import Renderer, MiniCam


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def joint_mapper_smplx_to_openpose18(joints):
    indices = (
        np.array(
            [
                56,  # nose
                13,  # neck
                18,  # right_shoulder
                20,  # right_elbow
                22,  # right_wrist
                17,  # left_shoulder
                19,  # left_elbow
                21,  # left_wrist
                3,  # right_hip
                6,  # right_knee
                9,  # right_ankle
                2,  # left_hip
                5,  # left_knee
                8,  # left_ankle
                57,  # right_eye
                58,  # left_eye
                59,  # right_ear
                60,  # left_ear
            ],
            dtype=np.int64,
        )
        - 1
    )
    return joints[indices]


class Skeleton:
    def __init__(self, opt):
        # init pose [18, 3], in [-1, 1]^3
        self.points3D = np.array(
            [
                [-0.00313026, 0.16587697, 0.05414092],
                [-0.00857283, 0.1093518, -0.00522604],
                [-0.06817748, 0.10397182, -0.00657925],
                [-0.11421658, 0.04033477, 0.00040599],
                [-0.15643744, -0.02915882, 0.03309248],
                [0.05288884, 0.10729481, -0.00067854],
                [0.10355149, 0.04464601, -0.00735265],
                [0.15390812, -0.02282556, 0.03085238],
                [0.03897187, -0.0403506, 0.00220192],
                [0.04027461, -0.15746351, -0.00187036],
                [0.04605377, -0.26837209, -0.0018945],
                [-0.0507806, -0.04887162, 0.0022531],
                [-0.04873568, -0.16551849, -0.00128197],
                [-0.04840493, -0.27510208, -0.00128831],
                [-0.03098677, 0.19395538, 0.01987491],
                [0.01657042, 0.19560097, 0.02724142],
                [-0.05411603, 0.17336673, -0.01328044],
                [0.03733583, 0.16922003, -0.00946565],
            ],
            dtype=np.float32,
        )

        self.name = [
            "nose",
            "neck",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_hip",
            "right_knee",
            "right_ankle",
            "left_hip",
            "left_knee",
            "left_ankle",
            "right_eye",
            "left_eye",
            "right_ear",
            "left_ear",
        ]

        # homogeneous
        self.points3D = np.concatenate(
            [self.points3D, np.ones_like(self.points3D[:, :1])], axis=1
        )  # [18, 4]

        # lines [17, 2]
        self.lines = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [1, 5],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
                [9, 10],
                [1, 11],
                [11, 12],
                [12, 13],
                [0, 14],
                [14, 16],
                [0, 15],
                [15, 17],
            ],
            dtype=np.int32,
        )

        # keypoint color [18, 3]
        # color as in controlnet_aux (https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/open_pose/util.py#L94C5-L96C73)
        self.colors = [
            [255, 0, 0],
            [255, 85, 0],
            [255, 170, 0],
            [255, 255, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [85, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [255, 0, 170],
            [255, 0, 85],
        ]

        # smplx mesh if available
        self.smplx_model = None
        self.vertices = None
        self.faces = None
        self.ori_center = None
        self.ori_scale = None

        self.body_pose = np.zeros((21, 3), dtype=np.float32)
        #T-pose without initialize

        # let's default to A-pose
        # self.body_pose[15, 2] = -0.7853982
        # self.body_pose[16, 2] = 0.7853982
        # self.body_pose[0, 1] = 0.2
        # self.body_pose[0, 2] = 0.1
        # self.body_pose[1, 1] = -0.2
        # self.body_pose[1, 2] = -0.1

        """ SMPLX body_pose definition
        0: 'left_hip',#'L_Hip', XYZ -> (-X)(-Y)Z, 后外高 -> 前里高 (3) XYZ
        1: 'right_hip',#'R_Hip', (4) XYZ -> (-X)(-Y)Z, 后里低 -> 前外低 (4) XYZ
        2: 'spine1',#'Spine1', (-X)Y(-Z) -> (0) XYZ
        3: 'left_knee',#'L_Knee', 同左UpperLeg
        4: 'right_knee',#'R_Knee',同右UpperLeg
        5: 'spine2',
        6: 'left_ankle',
        7: 'right_ankle',#'R_Ankle',同右UpperLeg
        8: 'spine3',#'Spine3', (-X)Y(-Z) 同脊椎
        9: 'left_foot',#'L_Foot',同左UpperLeg
        10: 'right_foot',#'R_Foot',同右UpperLeg
        11: 'neck',#'Neck', (-X)Y(-Z) 同脊椎
        12: 'left_collar',#'L_Collar', XYZ -> ZXY (VRM), 前拧, 后, 高 -> 高, 前拧, 后 (1) YZX
        13: 'right_collar',#'R_Collar', XYZ -> (-Z)(-X)Y , 前拧, 前, 低 -> 高, 后拧, 前 (2) YZX
        14: 'head',#'Head', (-X)Y(-Z) 同脊椎
        15: 'left_shoulder',#'L_Shoulder', 同左肩膀
        16: 'right_shoulder',#'R_Shoulder', 同右肩膀
        17: 'left_elbow',#'L_Elbow', 同左肩膀
        18: 'right_elbow',#'R_Elbow', 同右肩膀
        19: 'left_wrist',#'L_Wrist', 同左肩膀
        20: 'right_wrist',#'R_Wrist', 同右肩膀
        """

        self.left_hand_pose = np.zeros((15, 3), dtype=np.float32)
        self.right_hand_pose = np.zeros((15, 3), dtype=np.float32)
        """ hand_pose definition
        index, middle, pinky, ring, thumb; each with 3 joints.
        """

        # gaussian model
        self.gs = Renderer(sh_degree=0, white_background=False)
        self.gs.gaussians.load_ply(opt.ply)

        # motion data
        self.motion_seq = np.load(opt.motion)["poses"][:, 1:22]

        # gaussian center to smplx faces mapping
        self.mapping_dist = None
        self.mapping_face = None
        self.mapping_uvw = None
        
        # 改进驱动参数
        self.use_vertex_normals = True  # 使用顶点法线代替面法线（更平滑）
        self.error_threshold = 0.01  # 映射误差阈值
        self.normal_smoothing = True  # 法线平滑
        self.distance_scale_factor = 1.0  # 距离缩放因子（适应形变）
        
        # 穿模避免参数
        self.enable_collision_check = True  # 启用碰撞检测
        self.min_surface_distance = 0.001  # 最小表面距离（避免穿模）
        self.original_mesh_vertices = None  # 保存初始mesh顶点（用于局部形变分析）
        self.adaptive_distance = True  # 根据局部形变自适应调整距离
        self.deformation_threshold = 0.05  # 局部形变阈值（超过此值才进行自适应调整）
        
        # 预防性约束参数
        self.use_preventive_driving = False  # 使用预防性驱动（暂时禁用，性能问题）
        self.joint_aware_scaling = True  # 关节感知的距离缩放
        self.max_approach_distance = 0.15  # 最大接近距离（超过此距离的点在关节处更保守）
        self.collision_prediction_threshold = 0.005  # 碰撞预测阈值（提前检测潜在穿模）
        
        # Gaussian点之间的相互排斥约束（解决不同部位叠加问题）
        self.enable_gaussian_repulsion = True  # 启用Gaussian点之间的排斥
        self.repulsion_threshold = 0.02  # 排斥阈值（点之间距离小于此值时产生排斥，可调大）
        self.repulsion_strength = 0.4  # 排斥强度（0-1之间）
        self.repulsion_knn = 32  # 每个点检查的最近邻数量（GPU加速）
        self.use_full_repulsion = True  # 对所有点进行排斥检测（而非采样）

    @property
    def center(self):
        return self.points3D[:, :3].mean(0)

    @property
    def center_upper(self):
        return self.points3D[0, :3]

    @property
    def torso_bbox(self):
        # valid_points = self.points3D[[0, 1, 8, 11], :3]
        valid_points = self.points3D[:, :3]
        # assure 3D thickness
        min_point = valid_points.min(0) - 0.1
        max_point = valid_points.max(0) + 0.1
        remedy_thickness = np.maximum(0, 0.8 - (max_point - min_point)) / 2
        min_point -= remedy_thickness
        max_point += remedy_thickness
        return min_point, max_point

    def sample_points(self, noise=0.05, N=1000):
        # just sample N points around each line
        pc = []
        for i in range(17):
            A = self.points3D[[self.lines[i][0]], :3]  # [1, 3]
            B = self.points3D[[self.lines[i][1]], :3]
            x = np.linspace(0, 1, N)[:, None]  # [N, 1]
            points = A * (1 - x) + B * x
            # add noise
            points += np.random.randn(N, 3) * noise
            pc.append(points)
        pc = np.concatenate(pc, axis=0)  # [17 * N, 3]
        return pc

    def write_json(self, path):
        with open(path, "w") as f:
            d = {}
            for i in range(18):
                d[self.name[i]] = self.points3D[i, :3].tolist()
            json.dump(d, f)

    def load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                d = json.load(f)

        # load keypoints
        for i in range(18):
            self.points3D[i, :3] = np.array(d[self.name[i]])

    def load_smplx(self, path, betas=None, expression=None, gender="neutral"):
        import smplx

        if self.smplx_model is None:
            self.smplx_model = smplx.create(
                path,
                model_type="smplx",
                gender=gender,
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext="npz",
                use_pca=False,  # explicitly control hand pose
                flat_hand_mean=True,  # use a flatten hand default pose
            )

        # betas = torch.randn([1, self.smplx_model.num_betas], dtype=torch.float32)
        # expression = torch.randn([1, self.smplx_model.num_expression_coeffs], dtype=torch.float32)

        smplx_output = self.smplx_model(
            body_pose=torch.tensor(self.body_pose, dtype=torch.float32).unsqueeze(0),
            left_hand_pose=torch.tensor(
                self.left_hand_pose, dtype=torch.float32
            ).unsqueeze(0),
            right_hand_pose=torch.tensor(
                self.right_hand_pose, dtype=torch.float32
            ).unsqueeze(0),
            betas=betas,
            expression=expression,
            return_verts=True,
        )

        self.vertices = smplx_output.vertices.detach().cpu().numpy()[0]  # [10475, 3]
        self.faces = self.smplx_model.faces  # [20908, 3]

        # tmp: save deformed smplx mesh
        # import trimesh
        # _mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        # _mesh.export('smplx.obj')

        joints = smplx_output.joints.detach().cpu().numpy()[0]  # [127, 3]
        joints = joint_mapper_smplx_to_openpose18(joints)

        self.points3D = np.concatenate(
            [joints, np.ones_like(joints[:, :1])], axis=1
        )  # [18, 4]

        # rescale and recenter
        if self.ori_center is None:
            vmin = self.vertices.min(0)
            vmax = self.vertices.max(0)
            self.ori_center = (vmax + vmin) / 2
            self.ori_scale = 0.6 / np.max(vmax - vmin)

        self.vertices = (self.vertices - self.ori_center) * self.ori_scale
        self.points3D[:, :3] = (self.points3D[:, :3] - self.ori_center) * self.ori_scale

        self.scale(-10)  # rescale
        
        # 保存初始mesh顶点（用于碰撞检测）
        if self.original_mesh_vertices is None:
            self.original_mesh_vertices = self.vertices.copy()

        # update gaussian location
        if self.mapping_face is None:
            import cubvh

            points = self.gs.gaussians.get_xyz.detach()

            BVH = cubvh.cuBVH(self.vertices, self.faces)
            mapping_dist, mapping_face, mapping_uvw = BVH.signed_distance(
                points, return_uvw=True, mode="raystab"
            )

            self.mapping_dist = mapping_dist.detach().cpu().numpy()
            self.mapping_face = mapping_face.detach().cpu().numpy().astype(np.int32)
            self.mapping_uvw = mapping_uvw.detach().cpu().numpy().astype(np.float32)

            faces = self.faces[self.mapping_face]
            v0 = self.vertices[faces[:, 0]]
            v1 = self.vertices[faces[:, 1]]
            v2 = self.vertices[faces[:, 2]]
            
            # 计算法线（使用顶点法线或面法线）
            if self.use_vertex_normals:
                vertex_normals = self._compute_vertex_normals(self.vertices, self.faces)
                # 使用重心坐标插值顶点法线
                n0 = vertex_normals[faces[:, 0]]
                n1 = vertex_normals[faces[:, 1]]
                n2 = vertex_normals[faces[:, 2]]
                fnormals = (
                    n0 * self.mapping_uvw[:, [0]]
                    + n1 * self.mapping_uvw[:, [1]]
                    + n2 * self.mapping_uvw[:, [2]]
                )
                fnormals = fnormals / (
                    np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20
                )
            else:
                # 使用面法线（原始方法）
                fnormals = np.cross(v1 - v0, v2 - v0)
                fnormals = fnormals / (
                    np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20
                )

            # it seems some point (3000 out of 526294) cannot be accurately remapped...
            cpoints = (
                v0 * self.mapping_uvw[:, [0]]
                + v1 * self.mapping_uvw[:, [1]]
                + v2 * self.mapping_uvw[:, [2]]
            )
             # 应用距离缩放因子
            points = cpoints + self.mapping_dist[:, None] * fnormals * self.distance_scale_factor

            gt_points = self.gs.gaussians.get_xyz.detach().cpu().numpy()

            # print(points, gt_points)
            err = np.sqrt(np.sum((points - gt_points) ** 2, axis=-1))
            print(f"[驱动映射] 误差统计 - max: {err.max():.6f}, mean: {err.mean():.6f}, min: {err.min():.6f}, 超过阈值({self.error_threshold}): {(err > self.error_threshold).sum()}")

            # cull these erronous points...
            mask = ~(err > self.error_threshold)
            self.gs.gaussians._xyz = self.gs.gaussians._xyz[mask]
            self.gs.gaussians._features_dc = self.gs.gaussians._features_dc[mask]
            self.gs.gaussians._features_rest = self.gs.gaussians._features_rest[mask]
            self.gs.gaussians._opacity = self.gs.gaussians._opacity[mask]
            self.gs.gaussians._scaling = self.gs.gaussians._scaling[mask]
            self.gs.gaussians._rotation = self.gs.gaussians._rotation[mask]
            self.mapping_dist = self.mapping_dist[mask]
            self.mapping_face = self.mapping_face[mask]
            self.mapping_uvw = self.mapping_uvw[mask]

        else:
            # 动画驱动更新：使用保存的映射关系更新Gaussian位置
            faces = self.faces[self.mapping_face]
            v0 = self.vertices[faces[:, 0]]
            v1 = self.vertices[faces[:, 1]]
            v2 = self.vertices[faces[:, 2]]
            
            # 计算法线（使用顶点法线或面法线）
            if self.use_vertex_normals:
                vertex_normals = self._compute_vertex_normals(self.vertices, self.faces)
                # 使用重心坐标插值顶点法线
                n0 = vertex_normals[faces[:, 0]]
                n1 = vertex_normals[faces[:, 1]]
                n2 = vertex_normals[faces[:, 2]]
                fnormals = (
                    n0 * self.mapping_uvw[:, [0]]
                    + n1 * self.mapping_uvw[:, [1]]
                    + n2 * self.mapping_uvw[:, [2]]
                )
                fnormals = fnormals / (
                    np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20
                )
            else:
                # 使用面法线（原始方法）
                fnormals = np.cross(v1 - v0, v2 - v0)
                fnormals = fnormals / (
                    np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20
                )

            # 计算面上的最近点（使用重心坐标）
            cpoints = (
                v0 * self.mapping_uvw[:, [0]]
                + v1 * self.mapping_uvw[:, [1]]
                + v2 * self.mapping_uvw[:, [2]]
            )

            # 使用预防性驱动：在计算位置前就预测并避免穿模
            if self.use_preventive_driving:
                points, drive_info = self._preventive_driving(
                    cpoints, fnormals, self.mapping_dist, 
                    self.mapping_face, self.vertices, self.faces
                )
                if drive_info.get("collisions_prevented", 0) > 0:
                    print(f"[预防性驱动] 预防了 {drive_info['collisions_prevented']} 个潜在穿模")
            else:
                # 原始方法：先计算位置，再检测修正
                if self.adaptive_distance:
                    deformation_factor = self._compute_local_deformation(self.mapping_face)
                    adaptive_scale = deformation_factor[:, None]
                else:
                    adaptive_scale = 1.0
                
                points = cpoints + self.mapping_dist[:, None] * fnormals * self.distance_scale_factor * adaptive_scale
                
                # 碰撞检测和修正：避免穿模（mesh内部）
                if self.enable_collision_check:
                    points, collision_mask = self._check_and_fix_collision(
                        points, self.vertices, self.faces
                    )
            
            # 应用Gaussian点之间的排斥，防止不同部位的点叠加
            if self.enable_gaussian_repulsion:
                points, repulsion_info = self._apply_gaussian_repulsion(points)
                if repulsion_info.get("repulsions_applied", 0) > 0:
                    print(f"[Gaussian排斥] 应用了 {repulsion_info['repulsions_applied']} 次排斥调整")
            
            self.gs.gaussians._xyz = torch.tensor(points, dtype=torch.float32).cuda()

    def _compute_vertex_normals(self, vertices, faces):
        """计算顶点法线（使用相邻面的加权平均）"""
        vertex_normals = np.zeros_like(vertices)
        
        # 计算每个面的法线
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-20
        face_normals = face_normals / face_areas
        
        # 将面法线加权累加到顶点
        for i, face in enumerate(faces):
            area = face_areas[i, 0]
            normal = face_normals[i]
            vertex_normals[face[0]] += normal * area
            vertex_normals[face[1]] += normal * area
            vertex_normals[face[2]] += normal * area
        
        # 归一化
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-20
        vertex_normals = vertex_normals / norms
        
        return vertex_normals

    def _compute_joint_safety_distance(self, face_indices, vertices):
        """计算关节区域的安全距离因子
        
        对于接近关节的点，使用更保守的距离以避免穿模
        
        Returns:
            safety_factor: [N] 安全因子（<1.0表示需要更小的距离）
        """
        if not self.joint_aware_scaling:
            return np.ones(len(face_indices))
        
        # 获取关节位置（基于SMPL-X的关节定义）
        # 这里使用简化的关节区域识别：基于mesh的曲率和局部密度
        safety_factors = np.ones(len(face_indices))
        
        # 获取每个面对应的顶点
        faces = self.faces[face_indices]
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        # 计算面的中心点
        face_centers = (v0 + v1 + v2) / 3.0
        
        # 简化方法：计算局部曲率（通过相邻面的法线差异）
        # 曲率高的区域通常是关节处，需要更保守的距离
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-20)
        
        # 简化方法：基于面的面积和法线变化估计曲率
        # 对于大面积的平滑区域（如躯干），曲率低；对于小面积或法线变化大的区域（如关节），曲率高
        
        # 计算每个面的面积
        face_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2.0
        avg_area = np.mean(face_areas)
        
        # 对于面积小的面（通常在关节处），使用更保守的距离
        area_factor = np.clip(face_areas / (avg_area + 1e-20), 0.3, 1.0)
        
        # 计算法线变化（与相邻面的差异）
        # 简化：使用所有面的平均法线作为参考
        avg_normal = np.mean(face_normals, axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-20)
        
        # 计算每个面法线与平均法线的差异
        normal_diffs = np.arccos(np.clip(
            np.abs(np.sum(face_normals * avg_normal[None, :], axis=1)),
            -1, 1
        ))
        
        # 曲率指标：法线差异越大，曲率越高
        curvature = normal_diffs / (np.pi / 2 + 1e-20)
        
        # 结合面积和曲率计算安全因子
        # 面积小且曲率高的区域（关节处）使用更小的因子
        safety_factors = 0.5 + 0.5 * (1.0 - 0.5 * curvature) * area_factor
        
        return safety_factors

    def _preventive_driving(self, cpoints, fnormals, mapping_dist, mapping_face, vertices, faces):
        """预防性驱动：在计算位置前预测并避免潜在穿模
        
        Args:
            cpoints: [N, 3] 面上的最近点
            fnormals: [N, 3] 法线方向
            mapping_dist: [N] 原始距离
            mapping_face: [N] 面索引
            vertices: [M, 3] mesh顶点
            faces: [F, 3] mesh面
            
        Returns:
            safe_points: [N, 3] 安全的位置
            adjustment_info: dict 调整信息
        """
        if not self.use_preventive_driving:
            # 使用原始方法
            adaptive_scale = 1.0
            if self.adaptive_distance:
                adaptive_scale = self._compute_local_deformation(mapping_face)[:, None]
            
            points = cpoints + mapping_dist[:, None] * fnormals * self.distance_scale_factor * adaptive_scale
            return points, {"method": "original"}
        
        import cubvh
        
        # 1. 计算初始安全距离（基于局部形变和关节感知）
        base_scale = self.distance_scale_factor
        
        # 自适应距离调整
        if self.adaptive_distance:
            deformation_factor = self._compute_local_deformation(mapping_face)
            base_scale = base_scale * deformation_factor[:, None]
        
        # 关节感知的安全因子
        if self.joint_aware_scaling:
            safety_factor = self._compute_joint_safety_distance(mapping_face, vertices)
            base_scale = base_scale * safety_factor[:, None]
        
        # 2. 计算初始位置
        initial_points = cpoints + mapping_dist[:, None] * fnormals * base_scale
        
        # 3. 预测性碰撞检测：检查是否有其他mesh部分会靠近
        BVH = cubvh.cuBVH(vertices, faces)
        
        # 计算到mesh的符号距离
        signed_dist, closest_faces, uvw = BVH.signed_distance(
            torch.tensor(initial_points, dtype=torch.float32).cuda(),
            return_uvw=True,
            mode="raystab"
        )
        signed_dist = signed_dist.detach().cpu().numpy()
        
        # 检测潜在碰撞：距离太近的点
        collision_risk_mask = signed_dist < (self.min_surface_distance + self.collision_prediction_threshold)
        
        if not collision_risk_mask.any():
            # 没有碰撞风险，直接返回
            return initial_points, {"method": "preventive", "collisions_prevented": 0}
        
        # 4. 对于有碰撞风险的点，计算安全距离
        safe_points = initial_points.copy()
        risk_indices = np.where(collision_risk_mask)[0]
        
        # 计算每个风险点的安全位置
        for idx in risk_indices:
            # 获取最近的表面点
            face_idx = closest_faces[idx].item()
            face = faces[face_idx]
            u, v, w = uvw[idx].detach().cpu().numpy()
            
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            surface_point = v0 * u + v1 * v + v2 * w
            
            # 计算表面法线（使用插值的顶点法线）
            if self.use_vertex_normals:
                vertex_normals = self._compute_vertex_normals(vertices, faces)
                n0 = vertex_normals[face[0]]
                n1 = vertex_normals[face[1]]
                n2 = vertex_normals[face[2]]
                surface_normal = n0 * u + n1 * v + n2 * w
            else:
                edge1 = v1 - v0
                edge2 = v2 - v0
                surface_normal = np.cross(edge1, edge2)
            
            surface_normal = surface_normal / (np.linalg.norm(surface_normal) + 1e-20)
            
            # 使用更保守的距离：原始距离的一部分，但至少保持最小距离
            original_dist = mapping_dist[idx]
            safe_dist = max(
                self.min_surface_distance,
                min(original_dist * 0.7, original_dist - abs(signed_dist[idx]))
            )
            
            # 计算安全位置
            safe_points[idx] = surface_point + surface_normal * safe_dist
        
        num_prevented = collision_risk_mask.sum()
        return safe_points, {
            "method": "preventive", 
            "collisions_prevented": num_prevented,
            "original_collision_count": num_prevented
        }

    def _check_and_fix_collision(self, points, vertices, faces):
        """检测并修正穿模问题
        
        Args:
            points: [N, 3] Gaussian点位置
            vertices: [M, 3] mesh顶点
            faces: [F, 3] mesh面
            
        Returns:
            corrected_points: [N, 3] 修正后的点位置
            collision_mask: [N] 是否发生穿模的mask
        """
        if not self.enable_collision_check:
            return points, np.zeros(len(points), dtype=bool)
        
        import cubvh
        
        # 使用BVH计算点到mesh的符号距离（负值表示在mesh内部）
        BVH = cubvh.cuBVH(vertices, faces)
        signed_dist, closest_faces, uvw = BVH.signed_distance(
            torch.tensor(points, dtype=torch.float32).cuda(),
            return_uvw=True,
            mode="raystab"
        )
        signed_dist = signed_dist.detach().cpu().numpy()
        
        # 检测穿模：符号距离为负或小于最小表面距离
        collision_mask = signed_dist < self.min_surface_distance
        
        if not collision_mask.any():
            return points, collision_mask
        
        # 对于穿模的点，计算最近的表面点并向外推
        corrected_points = points.copy()
        faces_collision = faces[closest_faces.detach().cpu().numpy()[collision_mask]]
        uvw_collision = uvw.detach().cpu().numpy()[collision_mask]
        
        # 计算表面上的最近点
        v0_coll = vertices[faces_collision[:, 0]]
        v1_coll = vertices[faces_collision[:, 1]]
        v2_coll = vertices[faces_collision[:, 2]]
        
        surface_points = (
            v0_coll * uvw_collision[:, [0]]
            + v1_coll * uvw_collision[:, [1]]
            + v2_coll * uvw_collision[:, [2]]
        )
        
        # 计算表面法线
        fnormals = np.cross(v1_coll - v0_coll, v2_coll - v0_coll)
        fnormals = fnormals / (np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20)
        
        # 将穿模的点推回到表面外侧（保持最小距离）
        corrected_points[collision_mask] = surface_points + fnormals * self.min_surface_distance
        
        num_collisions = collision_mask.sum()
        if num_collisions > 0:
            print(f"[碰撞检测] 修正了 {num_collisions} 个穿模点 (总计 {len(points)})")
        
        return corrected_points, collision_mask

    def _apply_gaussian_repulsion(self, points):
        """应用Gaussian点之间的相互排斥，防止不同部位的点叠加
        
        GPU加速版本：使用k-NN搜索找到近邻，对所有点应用排斥
        
        Args:
            points: [N, 3] Gaussian点位置（已根据mesh变形计算）
            
        Returns:
            adjusted_points: [N, 3] 调整后的点位置
            repulsion_info: dict 排斥信息
        """
        if not self.enable_gaussian_repulsion:
            return points, {"repulsions_applied": 0}
        
        N = len(points)
        if N == 0:
            return points, {"repulsions_applied": 0}
        
        try:
            # 转移到GPU
            points_tensor = torch.tensor(points, dtype=torch.float32).cuda()
            
            # 使用较小的批次避免内存问题
            # 对于50万个点，使用10000的批次大小更安全
            batch_size = min(10000, N)
            num_batches = (N + batch_size - 1) // batch_size
            
            adjustments = torch.zeros_like(points_tensor)
            total_repulsions = 0
            
            print(f"[Gaussian排斥] 开始处理 {N} 个点，分 {num_batches} 批，阈值={self.repulsion_threshold:.4f}")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, N)
                batch_indices = torch.arange(start_idx, end_idx, device=points_tensor.device)
                batch_points = points_tensor[batch_indices]
                
                # 计算这批点与所有点的距离（向量化，分批计算以节省内存）
                # 对于大批次，需要分块计算避免OOM
                chunk_size = 10000  # 每次与1万个目标点计算距离
                batch_adjustments = torch.zeros_like(batch_points)
                
                for chunk_start in range(0, N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, N)
                    chunk_points = points_tensor[chunk_start:chunk_end]
                    
                    # diff: [batch_size, chunk_size, 3]
                    diff = batch_points[:, None, :] - chunk_points[None, :, :]
                    dists = torch.norm(diff, dim=2)  # [batch_size, chunk_size]
                    
                    # 找到距离小于阈值的点（排除自己）
                    chunk_indices_tensor = torch.arange(chunk_start, chunk_end, device=points_tensor.device)
                    close_mask = (dists < self.repulsion_threshold) & (dists > 1e-6)
                    
                    for i, idx in enumerate(batch_indices):
                        close_mask_i = close_mask[i]
                        if not close_mask_i.any():
                            continue
                        
                        close_chunk_indices = chunk_indices_tensor[close_mask_i]
                        close_dists = dists[i, close_mask_i]
                        directions = diff[i, close_mask_i]
                        
                        # 限制处理的近邻数量（取最近的k个）
                        if len(close_chunk_indices) > self.repulsion_knn:
                            _, topk_indices = torch.topk(close_dists, self.repulsion_knn, largest=False)
                            close_chunk_indices = close_chunk_indices[topk_indices]
                            close_dists = close_dists[topk_indices]
                            directions = directions[topk_indices]
                        
                        # 归一化方向
                        directions = directions / (close_dists[:, None] + 1e-20)
                        
                        force_magnitude = (self.repulsion_threshold - close_dists) / self.repulsion_threshold
                        forces = directions * force_magnitude[:, None] * self.repulsion_strength
                        total_force = forces.sum(dim=0)
                        
                        batch_adjustments[i] += total_force
                        total_repulsions += len(close_chunk_indices)
                
                # 应用批次调整
                adjustments[batch_indices] = batch_adjustments
                
                # 清理GPU内存
                torch.cuda.empty_cache()
                
                if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
                    print(f"[Gaussian排斥] 已处理 {batch_idx + 1}/{num_batches} 批，累计排斥: {total_repulsions}...")
            
            # 应用调整（可以添加平滑避免过度移动）
            adjusted_points_tensor = points_tensor + adjustments
            
            # 转回CPU numpy数组
            adjusted_points = adjusted_points_tensor.detach().cpu().numpy()
            
            return adjusted_points, {
                "repulsions_applied": total_repulsions,
                "method": "gpu_knn",
                "points_processed": N
            }
            
        except Exception as e:
            print(f"[警告] GPU排斥计算失败，回退到CPU简化版本: {e}")
            # 回退到简化版本：只对压缩区域调整
            return self._apply_gaussian_repulsion_simple(points)
    
    def _apply_gaussian_repulsion_simple(self, points):
        """简化版本的排斥（CPU，只对压缩区域）"""
        N = len(points)
        if N == 0 or self.original_mesh_vertices is None:
            return points, {"repulsions_applied": 0}
        
        # 计算压缩程度
        deformation_factor = self._compute_local_deformation(self.mapping_face)
        
        # 对压缩区域应用额外缩放
        additional_scale = np.ones(N)
        compress_mask = deformation_factor < 0.8
        if compress_mask.any():
            additional_scale[compress_mask] = 0.7 + 0.3 * deformation_factor[compress_mask]
        
        # 重新计算位置
        faces = self.faces[self.mapping_face]
        v0 = self.vertices[faces[:, 0]]
        v1 = self.vertices[faces[:, 1]]
        v2 = self.vertices[faces[:, 2]]
        
        cpoints = (
            v0 * self.mapping_uvw[:, [0]]
            + v1 * self.mapping_uvw[:, [1]]
            + v2 * self.mapping_uvw[:, [2]]
        )
        
        if self.use_vertex_normals:
            vertex_normals = self._compute_vertex_normals(self.vertices, self.faces)
            n0 = vertex_normals[faces[:, 0]]
            n1 = vertex_normals[faces[:, 1]]
            n2 = vertex_normals[faces[:, 2]]
            fnormals = (
                n0 * self.mapping_uvw[:, [0]]
                + n1 * self.mapping_uvw[:, [1]]
                + n2 * self.mapping_uvw[:, [2]]
            )
            fnormals = fnormals / (np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20)
        else:
            fnormals = np.cross(v1 - v0, v2 - v0)
            fnormals = fnormals / (np.linalg.norm(fnormals, axis=1, keepdims=True) + 1e-20)
        
        adjusted_points = cpoints + self.mapping_dist[:, None] * fnormals * additional_scale[:, None]
        
        return adjusted_points, {"repulsions_applied": compress_mask.sum(), "method": "simple"}

    def _compute_local_deformation(self, face_indices):
        """计算局部形变程度（基于顶点位置变化）
        
        Args:
            face_indices: [N] 每个Gaussian点对应的面索引
            
        Returns:
            deformation_factor: [N] 形变因子（1.0表示无形变，<1.0表示压缩）
        """
        if self.original_mesh_vertices is None or not self.adaptive_distance:
            return np.ones(len(face_indices))
        
        # 获取每个面对应的顶点
        faces = self.faces[face_indices]
        v0_orig = self.original_mesh_vertices[faces[:, 0]]
        v1_orig = self.original_mesh_vertices[faces[:, 1]]
        v2_orig = self.original_mesh_vertices[faces[:, 2]]
        v0_curr = self.vertices[faces[:, 0]]
        v1_curr = self.vertices[faces[:, 1]]
        v2_curr = self.vertices[faces[:, 2]]
        
        # 计算原始和当前面的面积
        edge1_orig = v1_orig - v0_orig
        edge2_orig = v2_orig - v0_orig
        edge1_curr = v1_curr - v0_curr
        edge2_curr = v2_curr - v0_curr
        
        area_orig = np.linalg.norm(np.cross(edge1_orig, edge2_orig), axis=1)
        area_curr = np.linalg.norm(np.cross(edge1_curr, edge2_curr), axis=1)
        
        # 计算面积变化率（形变因子）
        area_ratio = area_curr / (area_orig + 1e-20)
        
        # 对于压缩的区域（面积变小），减小距离以避免穿模
        # 使用平滑函数：当压缩超过阈值时，逐渐减小距离
        deformation_factor = np.ones_like(area_ratio)
        compress_mask = area_ratio < (1.0 - self.deformation_threshold)
        
        if compress_mask.any():
            # 压缩越严重，距离缩放越小（但不能小于0.5）
            compress_ratio = area_ratio[compress_mask]
            deformation_factor[compress_mask] = np.maximum(0.5, compress_ratio)
        
        return deformation_factor

    def scale(self, delta):
        self.points3D[:, :3] *= 1.1 ** (-delta)
        if self.vertices is not None:
            self.vertices *= 1.1 ** (-delta)

    def pan(self, rot, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        delta = 0.0005 * rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        self.points3D[:, :3] += delta
        if self.vertices is not None:
            self.vertices += delta

    def draw(self, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = self.points3D @ mvp.T  # [18, 4]
        points = points[:, :3] / points[:, 3:]  # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H  # [18]
        ys = (points[:, 1] + 1) / 2 * W  # [18]
        mask = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)

        # hide certain keypoints based on empirical occlusion
        if enable_occlusion:
            # decide view by the position of nose between two ears
            if points[0, 2] > points[-1, 2] and points[0, 2] < points[-2, 2]:
                # left view
                mask[-2] = False  # no right ear
                if xs[-4] > xs[-3]:
                    mask[-4] = False  # no right eye if it's "righter" than left eye
            elif points[0, 2] < points[-1, 2] and points[0, 2] > points[-2, 2]:
                # right view
                mask[-1] = False
                if xs[-3] < xs[-4]:
                    mask[-3] = False
            elif points[0, 2] > points[-1, 2] and points[0, 2] > points[-2, 2]:
                # back view
                mask[0] = False  # no nose
                mask[-3] = False  # no eyes
                mask[-4] = False

        # 18 points
        for i in range(18):
            if not mask[i]:
                continue
            cv2.circle(
                canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1
            )

        # 17 lines
        for i in range(17):
            cur_canvas = canvas.copy()
            if not mask[self.lines[i]].all():
                continue
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1
            )

            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])

            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)

    def render_gs(self, cam, H, W):
        cur_cam = MiniCam(cam.pose, H, W, cam.fovy, cam.fovx, cam.near, cam.far)

        out = self.gs.render(cur_cam)

        image = out["image"].permute(1, 2, 0).contiguous()  # [H, W, 3] in [0, 1]

        return image.detach().cpu().numpy()


class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.gui = opt.gui

        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.skel = Skeleton(opt)
        self.glctx = dr.RasterizeCudaContext()

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.mode = "gs"

        self.playing = False
        self.seq_id = 0

        self.save_image_path = "pose.png"
        self.save_json_path = "pose.json"
        self.mouse_loc = np.array([0, 0])
        self.points2D = None  # [18, 2]
        self.point_idx = 0
        self.drag_sensitivity = 0.0001
        self.pan_scale_skel = False
        self.enable_occlusion = True

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def step(self):
        if self.need_update:
            # mvp
            mv = self.cam.view  # [4, 4]
            proj = self.cam.perspective  # [4, 4]
            mvp = proj @ mv

            if self.mode == "skel":
                # render our openpose image, somehow
                self.render_buffer, self.points2D = self.skel.draw(
                    mvp, self.H, self.W, enable_occlusion=self.enable_occlusion
                )

            # if with smplx, overlay normal of mesh
            elif self.mode == "mesh":
                self.render_buffer = self.render_mesh_normal(
                    mvp, self.H, self.W, self.skel.vertices, self.skel.faces
                )

            # overlay gaussian splattings...
            elif self.mode == "gs":
                self.render_buffer = self.skel.render_gs(self.cam, self.H, self.W)

            self.need_update = False

            if self.gui:
                dpg.set_value("_texture", self.render_buffer)

        if self.playing:
            self.skel.body_pose = np.array(self.skel.motion_seq[self.seq_id % len(self.skel.motion_seq)])
            self.seq_id += 1
            self.skel.load_smplx(self.opt.smplx_path)
            self.need_update = True

    def render_mesh_normal(self, mvp, H, W, vertices, faces):
        mvp = torch.from_numpy(mvp.astype(np.float32)).cuda()
        vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
        faces = torch.from_numpy(faces.astype(np.int32)).cuda()

        vertices_clip = (
            torch.matmul(
                F.pad(vertices, pad=(0, 1), mode="constant", value=1.0),
                torch.transpose(mvp, 0, 1),
            )
            .float()
            .unsqueeze(0)
        )  # [1, N, 4]
        rast, _ = dr.rasterize(self.glctx, vertices_clip, faces, (H, W))

        i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
        v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(vertices)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = safe_normalize(normal)

        normal_image = (normal[0] + 1) / 2
        normal_image = torch.where(
            rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)
        )  # remove background
        buffer = normal_image.detach().cpu().numpy()

        return buffer

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            label="Viewer",
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # save image
            def callback_save_image(sender, app_data):
                image = (self.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_image_path, image)
                print(f"[INFO] write image to {self.save_image_path}")

            def callback_set_save_image_path(sender, app_data):
                self.save_image_path = app_data

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="save image",
                    tag="_button_save_image",
                    callback=callback_save_image,
                )
                dpg.bind_item_theme("_button_save_image", theme_button)

                dpg.add_input_text(
                    label="",
                    default_value=self.save_image_path,
                    callback=callback_set_save_image_path,
                )

            # save json
            def callback_save_json(sender, app_data):
                self.skel.write_json(self.save_json_path)
                print(f"[INFO] write json to {self.save_json_path}")

            def callback_set_save_json_path(sender, app_data):
                self.save_json_path = app_data

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="save json",
                    tag="_button_save_json",
                    callback=callback_save_json,
                )
                dpg.bind_item_theme("_button_save_json", theme_button)

                dpg.add_input_text(
                    label="",
                    default_value=self.save_json_path,
                    callback=callback_set_save_json_path,
                )

            # pan/scale mode
            def callback_set_pan_scale_mode(sender, app_data):
                self.pan_scale_skel = not self.pan_scale_skel

            dpg.add_checkbox(
                label="pan/scale skeleton",
                default_value=self.pan_scale_skel,
                callback=callback_set_pan_scale_mode,
            )

            # backview mode
            def callback_set_occlusion_mode(sender, app_data):
                self.enable_occlusion = not self.enable_occlusion
                self.need_update = True

            dpg.add_checkbox(
                label="use occlusion",
                default_value=self.enable_occlusion,
                callback=callback_set_occlusion_mode,
            )

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_int(
                label="FoV (vertical)",
                min_value=1,
                max_value=120,
                format="%d deg",
                default_value=self.cam.fovy,
                callback=callback_set_fovy,
            )

            # mode combo
            def callback_change_mode(sender, app_data):
                self.mode = app_data
                self.need_update = True

            dpg.add_combo(
                ("gs", "mesh", "skel"),
                label="mode",
                default_value=self.mode,
                callback=callback_change_mode,
            )

            # play the sequence
            def callback_play(sender, app_data):
                if self.playing:
                    self.playing = False
                    dpg.configure_item("_button_play", label="start")
                else:
                    self.playing = True
                    dpg.configure_item("_button_play", label="stop")

            dpg.add_button(label="start", tag="_button_play", callback=callback_play)
            dpg.bind_item_theme("_button_play", theme_button)

            # SMPLX pose editing
            with dpg.collapsing_header(label="SMPLX body_pose", default_open=False):

                def callback_update_body_pose(sender, app_data, user_data):
                    self.skel.body_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.body_pose.shape[0]):
                    dpg.add_input_floatx(
                        default_value=self.skel.body_pose[i],
                        size=3,
                        width=200,
                        format="%.3f",
                        on_enter=False,
                        callback=callback_update_body_pose,
                        user_data=i,
                    )

            with dpg.collapsing_header(
                label="SMPLX left_hand_pose", default_open=False
            ):

                def callback_update_left_hand_pose(sender, app_data, user_data):
                    self.skel.left_hand_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.left_hand_pose.shape[0]):
                    dpg.add_input_floatx(
                        default_value=self.skel.left_hand_pose[i],
                        size=3,
                        width=200,
                        format="%.3f",
                        on_enter=False,
                        callback=callback_update_left_hand_pose,
                        user_data=i,
                    )

            with dpg.collapsing_header(
                label="SMPLX right_hand_pose", default_open=False
            ):

                def callback_update_right_hand_pose(sender, app_data, user_data):
                    self.skel.right_hand_pose[user_data] = app_data[:3]
                    self.skel.load_smplx(self.opt.smplx_path)
                    self.need_update = True

                for i in range(self.skel.right_hand_pose.shape[0]):
                    dpg.add_input_floatx(
                        default_value=self.skel.right_hand_pose[i],
                        size=3,
                        width=200,
                        format="%.3f",
                        on_enter=False,
                        callback=callback_update_right_hand_pose,
                        user_data=i,
                    )

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            if self.pan_scale_skel:
                self.skel.scale(delta)
            else:
                self.cam.scale(delta)

            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            if self.pan_scale_skel:
                self.skel.pan(self.cam.rot, dx, dy)
            else:
                self.cam.pan(dx, dy)

            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        def callback_skel_select(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # determine the selected keypoint from mouse_loc
            if self.points2D is None:
                return  # not prepared

            dist = np.linalg.norm(self.points2D - self.mouse_loc, axis=1)  # [18]
            self.point_idx = np.argmin(dist)

        def callback_skel_drag(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]

            self.skel.points3D[self.point_idx, :3] += (
                self.drag_sensitivity
                * self.cam.rot.as_matrix()[:3, :3]
                @ np.array([dx, -dy, 0])
            )

            self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            # for skeleton editing
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right, callback=callback_skel_select
            )
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Right, callback=callback_skel_drag
            )

        dpg.create_viewport(
            title="pose viewer", resizable=False, width=self.W + 600, height=self.H
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.focus_item("_primary_window")

        dpg.setup_dearpygui()

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, help="path to gaussians ply", default="/home/dengxh/projects/HumanGaussian/content/save_ply/A_boy_with_a_beanie_wearing_black_leather_shoes-T-pose.ply")
    parser.add_argument("--motion", type=str, help="path to mition file", default="/home/dengxh/projects/HumanGaussian/content/smpl_walk_motion.npz")
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="/home/dengxh/projects/HumanGaussian/models/models_smplx_v1_1/models",
        help="path to models folder (contains smplx/)",
    )
    parser.add_argument(
        "--save", type=str, default="/home/dengxh/projects/HumanGaussian/saved/animated", help="path to render and save video"
    )

    parser.add_argument("--rotate", action="store_true", help="rotate during rendering")
    parser.add_argument(
        "--play", action="store_true", help="play the motion during rendering"
    )

    parser.add_argument("--W", type=int, default=800, help="GUI width")
    parser.add_argument("--H", type=int, default=800, help="GUI height")
    parser.add_argument("--gui", action="store_true", help="enable GUI")

    parser.add_argument(
        "--radius", type=float, default=2, help="default GUI camera radius from center"
    )
    parser.add_argument(
        "--fovy", type=float, default=50, help="default GUI camera fovy"
    )

    opt = parser.parse_args()

    # name = ( os.path.splitext(os.path.basename(opt.ply))[0] + "_" + os.path.splitext(os.path.basename(opt.motion))[0] )
    
    #取HG模型的导数第三字段，含有较多信息
    experiment_name = os.path.basename(os.path.dirname(os.path.dirname(opt.ply)))
    # motion 文件名
    motion_name = os.path.splitext(os.path.basename(opt.motion))[0]

    name = f"{experiment_name}_{motion_name}"


    gui = GUI(opt)

    print(f"[INFO] load smplx from {opt.smplx_path}")
    gui.skel.load_smplx(opt.smplx_path)
    gui.need_update = True

    if not opt.gui:
        os.makedirs(opt.save, exist_ok=True)

        import imageio

        images = []

        elevation = 0
        azimuth = np.arange(0, 360, 1, dtype=np.int32)
        rotation_len = len(azimuth)

        gui.playing = opt.play
        motion_len = len(gui.skel.motion_seq)

        total_len = min(motion_len, rotation_len)

        for i in tqdm.trange(total_len):
            if opt.rotate:
                gui.cam.from_angle(elevation, azimuth[i % rotation_len])
                gui.need_update = True

            gui.step()

            if opt.gui:
                dpg.render_dearpygui_frame()

            image = (gui.render_buffer * 255).astype(np.uint8)
            images.append(image)

        images = np.stack(images, axis=0)
        # ~6 seconds, 180 frames at 30 fps
        os.makedirs(opt.save, exist_ok=True)
        imageio.mimwrite(os.path.join(opt.save, f"{name}.mp4"), images, fps=30)

    else:
        gui.render()
