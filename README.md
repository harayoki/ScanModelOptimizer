# ScanModelOptimizer

3Dモデルのメッシュとテクスチャを自動で最適化するツールです。
デフォルトでの動作は、スキャンモデルを想定していますが、そうではないモデルにも使えます。その場合は引数を調整してください。

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

※ 薄いパーツのあるモデルではデフォルトの引数設定ではうまくテクスチャが転写できないようです。


### 使い方

pythonを直接実行します。入力glbファイル以外の引数はすべてオプションです。

```
usage: ScanModelOptimizer.py [-h] [-o OUTPUT] [--output_blend] [--quality QUALITY] [--decimate_ratio DECIMATE_RATIO] [--merge_distance MERGE_DISTANCE] [--do_only_remesh] [--texture_size TEXTURE_SIZE] [-tex {0,1}] [-nrm {0,1}] [--cage_extrusion CAGE_EXTRUSION] [--bake_diffuse_from {DIFFUSE,EMIT}]
                             [--mat_metallic MAT_METALLIC] [--mat_roughness MAT_ROUGHNESS] [--prevent_overwrite] [-ro [REMOVE_OBJECT_NAMES ...]] [--confirm]
                             file_path

positional arguments:
  file_path             入力ファイル (*.glb, *.gltf)

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        出力ファイルパス (*.glb, *.usdc)
  --output_blend        blendファイルも出力するか
  --quality QUALITY     出力クオリティ 通常1.0で大きいほど高品質になる
  --decimate_ratio DECIMATE_RATIO
                        ポリゴン削減率 1.0で削除しない
  --merge_distance MERGE_DISTANCE
                        頂点結合距離　0で結合しない
  --do_only_remesh      メッシュの統合のみ実行し、UV・テクスチャ関連の処理はしない
  --texture_size TEXTURE_SIZE
                        出力テクスチャーサイズ
  -tex {0,1}, --bake_texture {0,1}
                        テクスチャをベイクするか(1 or 0)
  -nrm {0,1}, --bake_normal {0,1}
                        法線マップをベイクするか
  --cage_extrusion CAGE_EXTRUSION
                        ベイク時のケージの押し出し量
  --bake_diffuse_from {DIFFUSE,EMIT}
                        DIFFUSEベイク元の指定　DIFFUSE: ディフューズカラー, EMIT: エミッションカラー
  --mat_metallic MAT_METALLIC
                        マテリアルのメタリック値
  --mat_roughness MAT_ROUGHNESS
                        マテリアルのラフネス値
  --prevent_overwrite   ファイル上書きを許可しない
  -ro [REMOVE_OBJECT_NAMES ...], --remove_object_names [REMOVE_OBJECT_NAMES ...]
                        レンダリング時に削除したいオブジェクト名称（正規表現）の指定
  --confirm             処理前に確認メッセージを表示する
```

### 実行例

OUTPUT引数はファイル指定です。
`python src\ScanModelOptimizer.py -o "temp\output.glb" input.glb`

## 特別な引数

### do_only_remesh

指定するとメッシュの結合・リダクションのみの対応を行います。 テクスチャの転写がうまく行かない場合など、
0かつqualityやcage_extrusion設定もうまく行かないような場合に、このオプションを指定してください。

### bake_diffuse_from

DIFFUSE（カラー）テクスチャは通常元モデルのDIFFUSEからベイクされますが、
スキャンモデルによってはEMIT（エミッション）にカラーがある場合があるのでこれを切り替えられるようにしています。

## ツールのアプリ化

PyInstallerでspecファイルを引数指定してください。distフォルダにアプリが出力されます。

```pyinstaller ScanModelOptimizer.spec```

アプリの引数はpythonコードを直接実行した場合と同じです。

```ScanModelOptimizer.exe -o "temp\output.glb" input.glb```


## アプリへのドラッグアンドロップ

Windowsの場合、アプリにglbファイルをドラッグ＆ドロップすることで、そのファイルを最適化できます。
その際全てのオプション引数がデフォルトとなるため、オプション引数を指定したい場合は、コマンドラインから実行するか
一度下記のような好きな引数を設定したバッチファイルを作成し、そこへglbファイルをドラッグ＆ドロップしてください。

```
@echo off
ScanModelOptimizer.exe %1 --texture_size 1024 --decimate_ratio 0.5
```


# ライセンス

GPL-3.0 Licenseです。ご注意ください。
[LICENSE](LICENSE) 