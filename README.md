# ScanModelOptimizer

3Dモデルのメッシュとテクスチャを自動で最適化するツールです。
デフォルトでの動作は、スキャンモデルを想定していますが、そうではないモデルにも使えます。
出力結果が綺麗にならない場合は、引数を調整してください。

bpyモジュールの力を借りて処理を行います。 
現状、入出力はGLBファイルのみ対応しています。FBXファイルの入出力はBlenderとの相性が悪いので現状考えていません。


# 動作環境

Windows11 + Python 3.11.4 で作成されています。bpy・PyInstallerなどのモジュールを利用します。
Macでも動作すると思われます。

動作環境構築までの手順例を下記に記します。

```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 処理内容

* 分割されているメッシュを統合
* 頂点数を削減
* 自動UV展開と新規UVへのテクスチャの転写

※ 薄いパーツのあるモデル等、デフォルトの引数設定ではうまくテクスチャが転写できない場合があります。


### 使い方

pythonを直接実行します。入力glbファイル以外の引数はすべてオプションです。

```
usage: ScanModelOptimizer.py [-h] [-o OUTPUT] [--output_blend]
                             [--output_blend_timing {BEFORE_BAKE,AFTER_BAKE,AFTER_EXPORT}]
                             [--quality QUALITY]
                             [--decimate_ratio DECIMATE_RATIO]
                             [--merge_distance MERGE_DISTANCE]
                             [--do_only_remesh] [--texture_size TEXTURE_SIZE]
                             [-tex {0,1}] [-nrm {0,1}]
                             [--cage_extrusion CAGE_EXTRUSION]
                             [--bake_diffuse_from {DIFFUSE,EMIT,COMBINED}]
                             [--mat_metallic MAT_METALLIC]
                             [--mat_roughness MAT_ROUGHNESS]
                             [--prevent_overwrite]
                             [-ro [REMOVE_OBJECT_NAMES ...]] [--confirm]
                             file_path

positional arguments:
  file_path             入力ファイル (*.glb, *.gltf)

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        出力ファイルパス (*.glb, *.usdc)
  --output_blend        詳細確認用のblendファイルも出力する場合に指定
  --output_blend_timing {BEFORE_BAKE,AFTER_BAKE,AFTER_EXPORT}
                        blendファイルも出力するタイミング BEFORE_BAKE / AFTER_BAKE /
                        AFTER_EXPORT (デフォルト: AFTER_EXPORT)
  --quality QUALITY     出力クオリティ 通常1.0で大きいほど高品質になる
  --decimate_ratio DECIMATE_RATIO
                        ポリゴン削減率 1.0で削除しない デフォルト0.5
  --merge_distance MERGE_DISTANCE
                        頂点結合距離　0で結合しない デフォルト0.001
  --do_only_remesh      メッシュの統合のみ実行し、UV・テクスチャ関連の処理はしない
  --texture_size TEXTURE_SIZE
                        出力テクスチャーサイズ デフォルト 2048
  -tex {0,1}, --bake_texture {0,1}
                        テクスチャをベイクするか(1 or 0) デフォルト1
  -nrm {0,1}, --bake_normal {0,1}
                        法線マップをベイクするか(1 or 0) デフォルト0
  --cage_extrusion CAGE_EXTRUSION
                        ベイク時のケージの押し出し量 デフォルト0.01
  --bake_diffuse_from {DIFFUSE,EMIT,COMBINED}
                        DIFFUSEベイク元の指定 'DIFFUSE':ディフューズカラー, 'EMIT':エミッションカラー
                        'COMBINED':最終合成 デフォルト: DIFFUSE
  --mat_metallic MAT_METALLIC
                        マテリアルのメタリック値 デフォルト 0.0
  --mat_roughness MAT_ROUGHNESS
                        マテリアルのラフネス値 デフォルト 0.5
  --prevent_overwrite   ファイル上書きを許可しない場合に指定
  -ro [REMOVE_OBJECT_NAMES ...], --remove_object_names [REMOVE_OBJECT_NAMES ...]
                        レンダリング時に削除したいオブジェクト名称（正規表現）の指定
  --confirm             処理前に確認メッセージを表示する場合に指定
```

### 実行例

OUTPUT引数はファイル指定です。
`python src\ScanModelOptimizer.py -o "temp\output.glb" input.glb`

## 特別な引数

### do_only_remesh

指定するとメッシュの結合・リダクションのみの対応を行います。 
qualityやcage_extrusion設定など、どう引数を調整してもテクスチャの転写がうまく行かない場合に、
このオプションを指定してください。（UVとテクスチャは元のモデルの物が保持されます。）

### bake_diffuse_from

DIFFUSE（カラー）テクスチャは通常元モデルのDIFFUSEからベイクされますが、
スキャンモデルによってはEMIT（エミッション）にカラーがある場合があるので、これを切り替えられるようにしています。

## ツールのアプリ化

### アプリ化の手順

PyInstallerでspecファイルを引数指定してください。distフォルダにアプリが出力されます。

```pyinstaller ScanModelOptimizer.spec```

もしくは

``` python -m PyInstaller .\ScanModelOptimizer.spec```

アプリの引数はpythonコードを直接実行した場合と同じです。

```ScanModelOptimizer.exe -o "temp\output.glb" input.glb```


### アプリへモデルファイルをドラッグアンドロップ

Windowsの場合、アプリにglbファイルをドラッグ＆ドロップすることで、そのファイルを最適化できます。
その際全てのオプション引数がデフォルトとなるため、オプション引数を指定したい場合は、コマンドラインから実行するか
一度下記のような好きな引数を設定したバッチファイルを作成し、そこへglbファイルをドラッグ＆ドロップしてください。

```
@echo off
ScanModelOptimizer.exe %1 --texture_size 1024 --decimate_ratio 0.5
```


# ライセンス

bpyモジュール利用のため、ライセンスがGPL-3.0 Licenseとなります。ご注意ください。
[LICENSE](LICENSE) 