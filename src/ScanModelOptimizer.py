from typing import List, Literal
from pathlib import Path
import datetime
import argparse
import os
import sys
import bpy
import re
from mathutils import Vector


def get_time_stamp() -> str:
    now = datetime.datetime.now()
    # return now.strftime("%Y%m%d%H%M%S")
    return now.strftime("%H:%M:%S")


def resource_path(relative_path_str: str) -> Path:
    """
    PyInstallerの実行時か否かで場合分けしてリソースファイルのパスを取得する
    """
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / Path(relative_path_str)
    return Path(__file__).absolute().parent.parent / relative_path_str


def get_all_children(parent_obj: bpy.types.Object) -> List[bpy.types.Object]:
    """すべての子オブジェクトを取得する"""
    children = []
    for child in parent_obj.children:
        children.append(child)
        children.extend(get_all_children(child))  # 再帰処理
    return children


# def get_sibling_objects(obj: bpy.types.Object) -> List[bpy.types.Object]:
#     """ヒエラルキー的に兄弟であるオブジェクトを返す"""
#     if obj.parent is None:
#         return []
#     parent = obj.parent
#     siblings = [child for child in parent.children if child != obj]
#     return siblings


def get_mesh_objects_in_hierarchy(all_objects: List[bpy.types.Object]) -> List[bpy.types.Object]:
    """指定オブジェクトの階層以下のメッシュオブジェクトを再帰的に取得する"""
    mesh_objects = []
    for obj in all_objects:
        if obj.type == 'MESH' and obj not in mesh_objects:
            mesh_objects.append(obj)
        if obj.children:
            mesh_objects.extend(get_mesh_objects_in_hierarchy(obj.children))
    mesh_objects = list(set(mesh_objects))  # 重複を念のため削除
    return mesh_objects


def join_mesh_objects(
        mesh_objects: List[bpy.types.Object], context: bpy.context, new_name: str = "") -> bpy.types.Object:
    """メッシュオブジェクトを一つに結合する"""
    if len(mesh_objects) == 0:
        raise ValueError("No mesh objects to merge.")
    # アクティブオブジェクトとして1つを設定し、それを基準に結合
    context.view_layer.objects.active = mesh_objects[0]
    # すべてのメッシュオブジェクトを選択状態にする
    if len(mesh_objects) == 1:
        print(f"only one object to merge: {mesh_objects[0].name}")
    else:
        for obj in mesh_objects:
            print(f"merge object: {obj.name}")
            obj.select_set(True)
        # メッシュオブジェクトを結合
        bpy.ops.object.join()
    merged = context.active_object
    if new_name != "":
        merged.name = new_name
    return merged


def scale_to_unit_box(obj: bpy.types.Object, scale: float = 1.0) -> float:
    """
    オブジェクトを指定の大きさにスケーリングする。
    内部処理で、スケールを確定しているので注意。
    :return サイズ変更率
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # オブジェクトのバウンディングボックスを取得
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    # 各軸のバウンディングボックスサイズを計算
    bbox_min = Vector((min(corner[i] for corner in bbox_corners) for i in range(3)))
    bbox_max = Vector((max(corner[i] for corner in bbox_corners) for i in range(3)))
    bbox_size = bbox_max - bbox_min
    # 最大サイズを基準にスケールを計算
    max_bbox_dim = max(bbox_size.x, bbox_size.y, bbox_size.z)
    if max_bbox_dim == 0:
        raise RuntimeError("バウンディングボックスのサイズが0です。")
    current_scale = obj.scale.copy()
    scale_factor = scale / max_bbox_dim
    obj.scale = current_scale * scale_factor
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return scale_factor


def remove_object_tree(obj_name: str, use_re: bool = True) -> None:
    """
    指定オブジェクトを下位ヒエラルキーも含め削除する
    obj_name: 削除対象オブジェクト名
    use_re: obj_name に正規表現を使うか否か
    """
    objects: List[bpy.types.Object] = []
    if use_re:
        objects = [o for o in bpy.data.objects if re.match(obj_name, o.name)]
    else:
        obj: bpy.types.Object = bpy.data.objects.get(obj_name)
        if obj is not None:
            objects.append(obj)
    if len(objects) == 0:
        print(f"{obj_name} not found")
        return

    object_to_remove: List[bpy.types.Object] = []
    for obj in objects:
        object_to_remove.append(obj)
        object_to_remove += get_all_children(obj)
    for obj in object_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


def move_to_root_keep_rotation(obj: bpy.types.Object) -> None:
    """
    最終的なトランスフォームを維持してオブジェクトをルートに移動する
    """
    world_matrix = obj.matrix_world.copy()
    # bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    obj.parent = None
    obj.matrix_world = world_matrix


def get_root_objects(objects: List[bpy.types.Object]) -> List[bpy.types.Object]:
    """ルートオブジェクトを取得する"""
    root_objects = []
    for obj in objects:
        if obj.parent is None:
            root_objects.append(obj)
    return root_objects


def extrude_region_shrink_fatten(obj: bpy.types.Object, value: float = 0.01) -> None:
    """
    全ての面を法線方向に押し出す
    :param obj: 対象オブジェクト
    :param value: 押し出し量
    """
    print(f"{get_time_stamp()} | Extruding faces")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_shrink_fatten(
        MESH_OT_extrude_region={"use_normal_flip": False, "use_dissolve_ortho_edges": False, "mirror": False},
        TRANSFORM_OT_shrink_fatten={"value": value, "use_even_offset": False, "mirror": False,
                                    "use_proportional_edit": False, "proportional_edit_falloff": 'SMOOTH',
                                    "proportional_size": 1, "use_proportional_connected": False,
                                    "use_proportional_projected": False, "snap": False, "release_confirm": False,
                                    "use_accurate": False})
    bpy.ops.object.mode_set(mode='OBJECT')


def add_voxel_remesh_modifier(
        obj: bpy.types.Object, voxel_size: float = 0.001, adaptivity: float = 0.0,
        apply_immediate: bool = False) -> bpy.types.RemeshModifier:
    """
    ボクセルリメッシュモディファイアを追加する
    """
    print(f"{get_time_stamp()} | Adding Voxel Remesh modifier")
    remesh_mod = obj.modifiers.new(name="Voxel Remesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = voxel_size
    # メッシュのディテールを調整するアダプティビティ（0 = 無効, 1 = 最大）
    remesh_mod.adaptivity = adaptivity
    # スムージングを自動で行うか
    remesh_mod.use_smooth_shade = False
    if apply_immediate:
        bpy.ops.object.modifier_apply(modifier=remesh_mod.name)
    return remesh_mod


def merge_by_distance(obj: bpy.types.Object, merge_distance: float = 0.002) -> None:
    """
    指定されたオブジェクトをエディットモードに切り替え、「Merge by Distance」を実行する。
    近い距離の頂点を結合する UVは崩れない
    """
    if merge_distance <= 0:
        return
    print(f"{get_time_stamp()} | Merging by distance")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_decimate_modifier(
        obj: bpy.types.Object, ratio: float = 1.0, apply_immediate: bool = False) -> None:
    """
    DecimateモディファイアをCOLLAPSEで適用する。
    """
    if ratio >= 1.0:
        return
    print(f"{get_time_stamp()} | Adding Decimate(collapse) modifier")
    decimate_mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    decimate_mod.decimate_type = 'COLLAPSE'
    decimate_mod.ratio = ratio

    if apply_immediate:
        bpy.ops.object.modifier_apply(modifier=decimate_mod.name)


def decimate_geometry(obj: bpy.types.Object, ratio: float) -> None:
    """
    指定されたオブジェクトをエディットモードに切り替え、デシメイト（Geometryの簡略化）を実行する。
    """
    if ratio >= 1.0:
        return
    print(f"{get_time_stamp()} | Decimating geometry")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.decimate(ratio=ratio)
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_smooth_modifier(
        obj: bpy.types.Object, factor: float = 2.0, repeat: int = 4, apply_immediate: bool = False) -> None:
    smooth_mod = obj.modifiers.new(name="Smooth", type='SMOOTH')
    print(f"{get_time_stamp()} | Adding Smooth modifier")
    smooth_mod.factor = factor
    smooth_mod.iterations = repeat
    if apply_immediate:
        bpy.ops.object.modifier_apply(modifier=smooth_mod.name)


def assign_material(obj: bpy.types.Object, mat_name: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(mat_name)
    # オブジェクトにマテリアルがすでに割り当てられていなければ、スロットを追加して割り当て
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    return mat


def auto_uv_unwrap(
        obj: bpy.types.Object, angle_limit: float = 66.0, island_margin: float = 0.00) -> None:
    """
    自動UV展開
    """
    print(f"{get_time_stamp()} | auto UV unwrap")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # 自動UV展開（スマートUVプロジェクト）を実行
    bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
    # オブジェクトモードに戻る
    bpy.ops.object.mode_set(mode='OBJECT')


def select_image_texture_node(material: bpy.types.Material, node_name: str) -> bpy.types.ImageTexture:
    """指定したマテリアルの中で、特定のImage Textureノードを選択状態にする"""
    if not material.use_nodes:
        sys.exit("マテリアルがノードを使用していません。")

    nodes = material.node_tree.nodes
    image_texture_node = nodes.get(node_name)

    if image_texture_node is None:
        sys.exit(f"ノード '{node_name}' が見つかりません。")

    # 全てのノードを選択解除
    for node in nodes:
        node.select = False
    # 指定されたImage Textureノードを選択
    image_texture_node.select = True
    # そのノードをアクティブに設定
    material.node_tree.nodes.active = image_texture_node
    return image_texture_node


def remove_images(image_name: str, use_re: bool = True) -> None:
    images_to_remove = []
    for image in bpy.data.images:
        if use_re:
            if re.match(image_name, image.name):
                images_to_remove.append(image)
        else:
            if image.name == image_name:
                images_to_remove.append(image)
    for image in images_to_remove:
        bpy.data.images.remove(image)


def disconnect_input_from_principled_bsdf(
        material: bpy.types.Material, disconnect_normal: bool = False, disconnect_color: bool= False) -> None:
    """
    Principled BSDFノードの入力を切断する
    :param material: マテリアル
    :param disconnect_normal: ノーマル入力を切断するか
    :param disconnect_color: カラー入力を切断するか
    """
    node_tree = material.node_tree
    principled_bsdf = None
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break

    if disconnect_normal:
        normal_input = principled_bsdf.inputs['Normal']
        if normal_input.is_linked:
            link = normal_input.links[0]
            node_tree.links.remove(link)
            print(f"'Normal' 入力が切断されました。")

    if disconnect_color:
        color_input = principled_bsdf.inputs['Base Color']
        if color_input.is_linked:
            link = color_input.links[0]
            node_tree.links.remove(link)
            print(f"'Base Color' 入力が切断されました。")


def set_material_metallic_and_roughness(
        material: bpy.types.Material, metallic_value: float, roughness_value: float) -> None:
    """
    指定されたマテリアルのPrincipled BSDFノードに対して、メタリックとラフネスを設定する。
    :param material: 対象のマテリアル
    :param metallic_value: メタリックの値（0.0〜1.0）
    :param roughness_value: ラフネスの値（0.0〜1.0）
    """
    node_tree = material.node_tree
    principled_bsdf = None
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break
    principled_bsdf.inputs['Metallic'].default_value = metallic_value
    principled_bsdf.inputs['Roughness'].default_value = roughness_value


def bake_texture(
        objects_from: List[bpy.types.Object],
        obj_to: bpy.types.Object,
        texture_size: int,
        mat_name: str,
        target_node_name: str,
        cage_extrusion: float,
        bake_type: Literal['DIFFUSE', 'NORMAL'],  # NOTE: 他にも増やせそう
        bake_diffuse_from: Literal['DIFFUSE', 'EMIT', 'COMBINED'] | None= 'DIFFUSE'
    ) -> None:
    """
    テクスチャをベイクする
    :param objects_from: ベイク元オブジェクトの配列
    :param obj_to: ベイク先オブジェクト
    :param texture_size: テクスチャサイズ
    :param mat_name: ベイク先オブジェクトのマテリアル名
    :param target_node_name: ベイク先マテリアル内のImage Textureノード名
    :param bake_type: ベイクタイプ（DIFFUSE, NORMAL）
    :param bake_diffuse_from: diffuseベイクの場合のベイク元タイプ（DIFFUSE, EMIT, COMBINED）
    """
    print(f"{get_time_stamp()} | Baking texture({bake_type})")
    material = bpy.data.materials.get(mat_name)

    # テクスチャがアクティブになるように設定
    bpy.context.view_layer.objects.active = obj_to
    obj_to.active_material_index = 0
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.render.engine = 'CYCLES'  # Cyclesレンダーエンジンを使用
    bpy.context.scene.cycles.bake_type = bake_type  # Cyclesのベイクタイプを指定

    bpy.context.scene.render.bake.use_selected_to_active = True  # 選択したオブジェクト間でベイク
    if bake_type == 'DIFFUSE':
        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_color = True
        bpy.context.scene.render.bake.use_pass_indirect = False
    elif bake_type == 'NORMAL':
        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_color = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        bpy.context.scene.render.bake.normal_space = 'TANGENT'
        bpy.context.scene.render.bake.normal_r = 'POS_X'
        bpy.context.scene.render.bake.normal_g = 'POS_Y'
        bpy.context.scene.render.bake.normal_b = 'POS_Z'

    bpy.context.scene.render.bake.use_cage = True
    bpy.context.scene.render.bake.cage_extrusion = cage_extrusion
    bpy.context.scene.render.bake.max_ray_distance = 0.00
    bpy.context.scene.render.bake.margin = 16

    # イメージオブジェクトの作成
    bpy.context.scene.render.bake.target = 'IMAGE_TEXTURES'
    bpy.context.scene.render.bake.use_clear = True
    image = bpy.data.images.new(name=target_node_name, width=texture_size, height=texture_size)
    if bake_type == 'NORMAL':
        image.colorspace_settings.name = 'Non-Color'
    else:
        image.colorspace_settings.name = 'sRGB'
    image_texture_node = select_image_texture_node(material, target_node_name)
    image_texture_node.image = image

    print("--------------------------------")
    print(f"render.engine: {bpy.context.scene.render.engine}")
    print(f"cycles.bake_type: {bpy.context.scene.cycles.bake_type}")
    print(f"render.bake.use_pass_direct: {bpy.context.scene.render.bake.use_pass_direct}")
    print(f"render.bake.use_pass_color: {bpy.context.scene.render.bake.use_pass_color}")
    print(f"render.bake.use_pass_indirect: {bpy.context.scene.render.bake.use_pass_indirect}")
    print(f"render.bake.normal_space: {bpy.context.scene.render.bake.normal_space}")
    print(f"render.bake.normal_r: {bpy.context.scene.render.bake.normal_r}")
    print(f"render.bake.normal_g: {bpy.context.scene.render.bake.normal_g}")
    print(f"render.bake.normal_b: {bpy.context.scene.render.bake.normal_b}")
    print(f"render.bake.use_cage: {bpy.context.scene.render.bake.use_cage}")
    print(f"render.bake.cage_extrusion: {bpy.context.scene.render.bake.cage_extrusion}")
    print(f"render.bake.max_ray_distance: {bpy.context.scene.render.bake.max_ray_distance}")
    print(f"render.bake.margin: {bpy.context.scene.render.bake.margin}")
    print(f"render.bake.target: {bpy.context.scene.render.bake.target}")
    print(f"render.bake.use_selected_to_active: {bpy.context.scene.render.bake.use_selected_to_active}")
    print(f"render.bake.use_clear: {bpy.context.scene.render.bake.use_clear}")
    print("--------------------------------")

    # ベイク対象のノードを選択状態にする
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects_from:
        try:
            obj.select_set(True)
        except Exception as e:
            print(f"{obj} Error: {e}")
            raise
    obj_to.select_set(True)
    bpy.context.view_layer.objects.active = obj_to

    if bake_type == 'DIFFUSE':
        if bake_diffuse_from is None:
            bpy.ops.object.bake()
        else:
            bpy.ops.object.bake(type=bake_diffuse_from)
    elif bake_type == 'NORMAL':
        bpy.ops.object.bake(type='NORMAL')
    else:
        sys.exit("Invalid bake type")

    print(f"{get_time_stamp()} | Baking finished")

    # # 法線マップのBチャンネルを修正する場合
    # if bake_type == 'NORMAL':
    #     pixels = list(image.pixels)
    #     for i in range(2, len(pixels), 4):  # Bチャンネルは2番目
    #         pixels[i] = 1.0
    #     # ピクセルデータを更新
    #     image.pixels = pixels

    # 外部ファイル保存する場合
    # image.filepath_raw = f"//{MERGED_TEXTURE_NAME}.png"
    # image.file_format = 'PNG'

    # ベイク結果をBlender内に保存
    if image.packed_file is None:
        image.pack()


def get_unique_filepath(filepath: Path) -> Path:
    """
    ファイル名がすでに存在する場合、連番を付与して新しいファイル名を返す。
    """
    if not filepath.exists():
        return filepath

    # 親ディレクトリ、ファイル名、拡張子に分解
    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix
    counter = 1

    while True:
        new_filepath = parent / f"{stem}_{counter}{suffix}"
        if not new_filepath.exists():
            return new_filepath
        counter += 1


def get_current_scene_vertex_count() -> int:
    """
    シーン内の頂点数を取得する
    """
    vertex_count = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            vertex_count += len(obj.data.vertices)
    return vertex_count


def export_model(obj: bpy.types.Object, file_path: Path, extensions: List[str]) -> None:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    for extension in extensions:
        file_path = file_path.with_suffix(extension)

        if file_path.suffix == ".glb":
            bpy.ops.export_scene.gltf(filepath=file_path.as_posix(), export_format='GLB', use_selection=True)
        elif file_path.suffix == ".usdc":
            bpy.ops.wm.usd_export(
                filepath=file_path.as_posix(),  # エクスポート先のパス
                check_existing=False,  # 既存ファイルの上書きを確認
                selected_objects_only=True,  # 選択されたオブジェクトのみをエクスポート
                export_animation=False,  # アニメーションを含めない場合は False
                export_materials=True,  # マテリアルを含める
                export_textures=True,  # テクスチャをエクスポート
                export_meshes=True,  # メッシュオブジェクトをエクスポート
                export_cameras=False,  # カメラを含める場合は True
                export_lights=False,  # ライトを含める場合は True
            )
        elif file_path.suffix == ".fbx":
            raise NotImplementedError("FBX export is not implemented yet.")
            # TODO fbx自体はバイナリファイルだが、テクスチャは外部ファイルとして保存しないといけない模様
            # bpy.ops.export_scene.fbx(
            #     filepath=file_path.as_posix(),  # エクスポート先のパス
            #     use_selection=True,  # 選択されたオブジェクトのみエクスポート
            #     global_scale=1.0,  # スケールの調整（1.0がデフォルト）
            #     apply_unit_scale=True,  # 単位スケールの適用
            #     bake_space_transform=False,  # 空間変換をベイクしない
            #     object_types={'MESH', 'EMPTY'},  # エクスポートするオブジェクトタイプ
            #     use_mesh_modifiers=True,  # モディファイアを適用
            #     use_mesh_modifiers_render=True,  # レンダリングモディファイアを適用
            #     mesh_smooth_type='OFF',  # スムージングの種類
            #     embed_textures=True,  # テクスチャを埋め込む
            #     path_mode='COPY',  # テクスチャのパスモード  # 真横にテクスチャを置けばいいらしい？
            # )
        elif file_path.suffix == ".obj":
            raise NotImplementedError("OBJ export is not implemented yet.")
            # デフォルトではobjエクスポートはなくなった模様
            # bpy.ops.preferences.addon_enable(module="export_obj")  # アドオンを有効化 現状Error
            # bpy.ops.export_scene.obj(
            #     filepath=str(file_path.as_posix()),  # 出力ファイルのパス
            #     use_selection=True,  # 選択されたオブジェクトのみをエクスポート
            #     use_materials=True,  # MTLファイルを出力
            #     use_normals=True,  # 法線情報を出力
            #     use_uvs=True,  # UVマップを出力
            #     axis_forward='-Z',  # 前方向
            #     axis_up='Y',  # 上方向
            #     path_mode='AUTO',  # パスのモード
            #     global_scale=1.0  # スケールの調整（1.0でデフォルト）
            # )


def load_model(input_file_path: Path, remove_object_names: List[str] | None) -> List[bpy.types.Object]:
    """
    モデルを読み込む
    :param input_file_path: 読み込むファイルのパス
    :param remove_object_names: 読み込み後に削除したいオブジェクト名称のリスト
    :return: 新規に読み込まれたオブジェクトのリスト
    """
    objects_before = set(bpy.data.objects)
    if input_file_path.suffix == ".glb" or input_file_path.suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=input_file_path.as_posix())
    else:
        sys.exit(f"Unsupported file format :{input_file_path}")
    # 読み込んだモデルからレンダリング前に削除したいものがあれば削除
    for remove_name in remove_object_names or []:
        remove_object_tree(remove_name, use_re=True)
    objects_after = set(bpy.data.objects)
    new_objects = objects_after - objects_before
    return list(new_objects)


def save_blend_file(file_path: Path) -> None:
    """
    ブレンドファイルを保存する
    """
    print(f"{get_time_stamp()} | Saving blend file: {file_path.as_posix()}")
    bpy.ops.wm.save_as_mainfile(filepath=file_path.as_posix())


###################################


WORK_OBJ_NAME: str = "WorkObj"  # シーンに配置されている作業用オブジェクトの名称 マテリアルが割り当てられている
MERGED_OBJ_NAME: str = "MergedObject"
MERGED_MATERIAL_NAME: str = "MergedMaterial"
MERGED_TEXTURE_NODE_NAME: str = "MergedTexture"
NORMAL_TEXTURE_NODE_NAME: str = "NormalTexture"

ALLOWED_INPUT_EXTENSIONS: List[str] = [".glb", ".gltf"]
ALLOWED_OUTPUT_EXTENSIONS: List[str] = [".glb", ".usdc"]  # ".fbx", ".obj"

OUTPUT_BLEND_TIMING: List[str] = ["BEFORE_BAKE", "AFTER_BAKE", "AFTER_EXPORT"]
BAKE_DIFFUSE_FROM: List[str] = ["DIFFUSE", "EMIT", "COMBINED"]

def main():
    print(f"bpy version: {bpy.app.version_string}")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", type=str,
        help=f"入力ファイル ({', '.join(['*' + ext for ext in ALLOWED_INPUT_EXTENSIONS])})")
    parser.add_argument(
        "-o", "--output", type=str, default="",
        help=f"出力ファイルパス ({', '.join(['*' + ext for ext in ALLOWED_OUTPUT_EXTENSIONS])})")
    # parser.add_argument(
    #     "--export_formats", type=str, nargs="*", default=[],
    #     help=f"出力ファイルフォーマット一覧 ({', '.join([ext[1:] for ext in ALLOWED_OUTPUT_EXTENSIONS])})")
    parser.add_argument("--output_blend", action="store_true",
                        help="詳細確認用のblendファイルも出力する場合に指定")
    parser.add_argument("--output_blend_timing", type=str,
                        choices=OUTPUT_BLEND_TIMING, default="AFTER_EXPORT",
                        help=f"blendファイルも出力するタイミング "
                             f"{' / '.join(OUTPUT_BLEND_TIMING)} (デフォルト: AFTER_EXPORT)")
    parser.add_argument("--quality", type=float, default=1.0,
                        help="出力クオリティ  通常1.0で大きいほど高品質になる")
    parser.add_argument("--decimate_ratio", type=float, default=0.5,
                        help="ポリゴン削減率 1.0で削除しない デフォルト0.5")
    parser.add_argument("--merge_distance", type=float, default=0.001,
                        help="頂点結合距離　0で結合しない デフォルト0.001")
    parser.add_argument("--do_only_remesh", action="store_true",
                        help="メッシュの統合のみ実行し、UV・テクスチャ関連の処理はしない")
    parser.add_argument("--texture_size", type=int, default=2048,
                        help="出力テクスチャーサイズ デフォルト 2048")
    parser.add_argument(
        "-tex", "--bake_texture", type=int, default=1, choices=[0, 1],
        help="テクスチャをベイクするか(1 or 0) デフォルト1")
    parser.add_argument(
        "-nrm", "--bake_normal", type=int, default=0, choices=[0, 1],
        help="法線マップをベイクするか(1 or 0) デフォルト0")
    parser.add_argument(
        "--cage_extrusion", type=float, default=0.01,
        help="ベイク時のケージの押し出し量 デフォルト0.01")
    parser.add_argument(
        "--bake_diffuse_from", type=str, default="DIFFUSE", choices=BAKE_DIFFUSE_FROM,
        help="DIFFUSEベイク元の指定 "
             "'DIFFUSE':ディフューズカラー, 'EMIT':エミッションカラー 'COMBINED':最終合成 デフォルト: DIFFUSE")
    parser.add_argument("--mat_metallic", type=float,  default=0.0,
                        help="マテリアルのメタリック値 デフォルト 0.0")
    parser.add_argument("--mat_roughness", type=float, default=0.5,
                        help="マテリアルのラフネス値 デフォルト 0.5")
    parser.add_argument("--prevent_overwrite", action="store_true",
                        help="ファイル上書きを許可しない場合に指定")
    parser.add_argument("-ro", "--remove_object_names", type=str, nargs="*", default=[],
                        help="レンダリング時に削除したいオブジェクト名称（正規表現）の指定")
    parser.add_argument("--confirm", action="store_true",
                        help="処理前に確認メッセージを表示する場合に指定")
    args = parser.parse_args()
    args.quality = max(0.1, args.quality) * 1.0
    args.texture_size = max(256, args.texture_size)

    assert os.path.exists(args.file_path), f"File not found: {args.file_path}"
    assert Path(args.file_path).suffix  in ALLOWED_INPUT_EXTENSIONS, \
        f"Unsupported input file format: {Path(args.file_path).suffix}"
    assert args.output == "" or (Path(args.output).suffix in ALLOWED_OUTPUT_EXTENSIONS), \
        f"Unsupported output file format: {Path(args.output).suffix}"

    input_file_path: Path = Path(args.file_path)
    output_path: Path
    if args.output == "":
        output_path = Path(args.file_path).with_name(Path(args.file_path).stem + "_out.glb")
    else:
        output_path = Path(args.output)
    output_path_blend = output_path.with_suffix(".blend")
    if args.prevent_overwrite:
        output_path = get_unique_filepath(output_path)
        output_path_blend = get_unique_filepath(output_path_blend)

    args.export_formats = []
    args.export_formats.append(output_path.suffix)
    args.export_formats = [ext if ext[0] == "." else "." + ext for ext in args.export_formats]
    args.export_formats = list(set(args.export_formats))
    for ext in args.export_formats:
        assert ext in ALLOWED_OUTPUT_EXTENSIONS, f"Unsupported output file format: {ext}"

    print("--------------------------------")
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_path}")
    print(f"Output format: {args.export_formats}")
    if args.output_blend:
        print(f"Output blend file: {output_path_blend}")
        print(f"Output blend timing: {args.output_blend_timing}")
    print(f"Quality: {args.quality}")
    print(f"Decimate ratio: {args.decimate_ratio}")
    print(f"Merge distance: {args.merge_distance}")
    print(f"Texture size: {args.texture_size}")
    # print(f"Prevent overwrite: {args.prevent_overwrite}")
    print(f"Bake cage extrusion: {args.cage_extrusion}")
    print(f"Bake texture: {args.bake_texture}")
    print(f"Bake normal: {args.bake_normal}")
    print(f"Metallic: {args.mat_metallic}")
    print(f"Roughness: {args.mat_roughness}")
    print(f"Roughness: {args.remove_object_names}")
    print("--------------------------------")

    if args.confirm:
        prompt = input("Continue? [y/n]: ")
        if prompt.lower() != "y":
            sys.exit("Aborted.")

    bpy.ops.wm.open_mainfile(filepath=resource_path("blend/template.blend").as_posix())
    # 不必要なworkオブジェクトを削除(マテリアルのわかりやすい保持のため入っている）
    print(f"work object : {bpy.data.objects.get(WORK_OBJ_NAME)}")
    bpy.data.objects.remove(bpy.data.objects[WORK_OBJ_NAME], do_unlink=True)

    new_objects = load_model(input_file_path, args.remove_object_names)
    # print(f"Loaded objects: {new_objects}")

    start_vertex_count = get_current_scene_vertex_count()

    # メッシュオブジェクトを結合する
    mesh_objects: List[bpy.types.Object] = get_mesh_objects_in_hierarchy(new_objects)
    if not mesh_objects:
        sys.exit("Mesh object not found")
    for obj in mesh_objects:
        print(f"Mesh object: {obj.name}")

    merged: bpy.types.Object = join_mesh_objects(mesh_objects, bpy.context, MERGED_OBJ_NAME)
    move_to_root_keep_rotation(merged)  # ヒエラルキーの最上位に移動
    # 統合メッシュ以外を削除
    for new_object in new_objects:
        if new_object != merged:  # メッシュ統合がなければ == になる
            try:
                delete_obj_name = new_object.name  # すでに削除されているとここでエラーになる メッシュ統合時なら正常動作
                bpy.data.objects.remove(new_object, do_unlink=True)
            except Exception as e:
                pass
            else:
                print(f"Removed: {delete_obj_name}")

    # サイズ1のBBOXに入るよう一時的にスケールを変更する ベイク処理のパラメータを簡易に設定できるように
    # quality ：一度サイズ1から拡大して処理して戻すことで、処理制度をあげるための引数
    scale_factor: float = scale_to_unit_box(merged, args.quality)
    scake_factor_inv = 1.0 / scale_factor
    print(f"Scale factor: {scale_factor}")

    # ########## リダクション処理 タイプA ##########
    # # 細かい部分がつぶれないようにメッシュを押し出す
    # extrude_value = 0.002 * args.quality
    # extrude_region_shrink_fatten(merged, extrude_value)
    # ボクセル変換
    # add_voxel_remesh_modifier(merged, apply_immediate=True)
    # スムージング ここで少し小さくなる
    # apply_smooth_modifier(merged, factor=2.0, repeat=4, apply_immediate=True)
    # # デシメイトモディファイヤを適用してポリゴン数を削減
    # apply_decimate_modifier(merged, ratio=args.decimate_ratio, apply_immediate=True)

    # ########## リダクション処理 タイプB こちらの方が圧倒的に速い ##########
    # 近い頂点を結合
    merge_by_distance(merged, merge_distance=args.merge_distance)
    # デシメイトモディファイヤを利用せず直接ポリゴン数を削減
    decimate_geometry(merged, ratio=args.decimate_ratio)

    final_vertex_count = get_current_scene_vertex_count()

    if not args.do_only_remesh:

        # 自動UV展開
        auto_uv_unwrap(merged)
        # あらかじめ用意したマテリアルを割り当てる
        mat = assign_material(merged, MERGED_MATERIAL_NAME)
        # 利用しないシェーダーインプットを切断
        disconnect_input_from_principled_bsdf(
            mat, disconnect_normal=args.bake_normal == 0, disconnect_color= args.bake_texture == 0)

        # メタリックとラフネスを設定
        set_material_metallic_and_roughness(
            mat, metallic_value=args.mat_metallic, roughness_value=args.mat_roughness)

        # テクスチャ転写用にもう一度モデルを読み込む
        new_objects = load_model(input_file_path, args.remove_object_names)
        print(f"Loaded objects: {new_objects}")
        new_root_objects = get_root_objects(new_objects)
        print(f"Root objects: {new_root_objects}")
        # ロードしたものがルートである場合がある 後の処理で消してはいけない
        keep_root_objects = []
        for new_root_object in new_root_objects:
            for new_object in new_objects:
                if new_root_object == new_object:
                    keep_root_objects.append(new_root_object)

        # テクスチャの転写
        mesh_objects = get_mesh_objects_in_hierarchy(new_root_objects)
        mesh_object: bpy.types.Object = (
            join_mesh_objects(mesh_objects, bpy.context))  # 統合するとベイクが1回になって大幅に速い
        move_to_root_keep_rotation(mesh_object)  # ヒエラルキーの最上位に移動
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # 現在のスケールがあるので確定
        mesh_object.scale = Vector((scale_factor, scale_factor, scale_factor))  # ベイク先と同じ大きさに

        # ベイク元モデルのゴミを削除
        for new_root_object in new_root_objects:
            if new_root_object not in keep_root_objects:
                try:
                    print(f"removing unnecessary object : {new_root_object.name}")
                    remove_object_tree(new_root_object.name)
                except Exception as e:
                    pass

        if args.output_blend and args.output_blend_timing == "BEFORE_BAKE":
            save_blend_file(output_path_blend)

        if args.bake_texture:
            try:
                bake_texture(
                    [mesh_object], merged,
                    bake_type='DIFFUSE',
                    texture_size=args.texture_size,
                    mat_name=MERGED_MATERIAL_NAME,
                    cage_extrusion=args.cage_extrusion,
                    target_node_name=MERGED_TEXTURE_NODE_NAME,
                    bake_diffuse_from=args.bake_diffuse_from  # 元オブジェクトはエミッションに色がある
                )
            except Exception as e:
                print(f"Error in bake_texture: {e}")
                print(f"テクスチャのベイクに失敗しました")
        if args.bake_normal:
            try:
                bake_texture(
                    [mesh_object], merged,
                    bake_type='NORMAL',
                    texture_size=args.texture_size,
                    mat_name=MERGED_MATERIAL_NAME,
                    cage_extrusion=args.cage_extrusion,
                    target_node_name=NORMAL_TEXTURE_NODE_NAME
                )
            except Exception as e:
                print(f"Error in bake_normal: {e}")
                print(f"ノーマルのベイクに失敗しました")

        if args.output_blend and args.output_blend_timing == "AFTER_BAKE":
            save_blend_file(output_path_blend)

        # 不必要なイメージを削除
        try:
            remove_images("Image_.+")
            bpy.data.objects.remove(bpy.data.objects[mesh_object.name], do_unlink=True)
        except Exception as e:
            print(f"Error in remove_images: {e}")
            print(f"不必要なイメージの削除に失敗しました")

    # ベイク用に調整していたスケールを元に戻す
    merged.scale = Vector((scake_factor_inv, scake_factor_inv, scake_factor_inv))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 出力
    print(f"{get_time_stamp()} | Exporting model")
    export_model(merged, file_path=output_path, extensions=args.export_formats)
    if args.output_blend and args.output_blend_timing == "AFTER_EXPORT":
        save_blend_file(output_path_blend)

    print(f"Vertex count: {start_vertex_count} -> {final_vertex_count}")


if __name__ == "__main__":
    main()

