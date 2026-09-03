# ModelScope 使用说明

本项目默认通过 [ModelScope](https://modelscope.cn)（阿里云魔搭）下载模型权重，替代 Hugging Face。

## 为什么用 ModelScope

本项目用到的 DINOv3 模型（`facebook/dinov3-vitl16-pretrain-lvd1689m`）在 Hugging Face 上是 **gated 仓库**，未授权会返回 403：`GatedRepoError` / `Cannot access gated repo`。

因此统一改用 ModelScope 下载。ModelScope 上对应的仓库均为**公开可下载**，无需授权。

## 依赖与安装

```bash
pip install modelscope
```

`setup.sh` 的 `--basic` 已包含该依赖：

```bash
bash setup.sh --basic
```

## 模型 ID 对照

| 用途 | Hugging Face | ModelScope |
|---|---|---|
| 3D 管线 | `microsoft/TRELLIS.2-4B` | `microsoft/TRELLIS.2-4B` |
| 图像特征 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| 抠图 | `ZhengPeng7/BiRefNet` | `modelscope/BiRefNet` |

绝大多数仓库在 ModelScope 上的 ID 与 HF 相同；仅个别需要映射（见下文 `HF_TO_MODELSCOPE`）。

## 下载工具模块

新增统一的模型下载入口：`trellis2/utils/model_hub.py`

- **优先使用 ModelScope** 下载，失败时**自动回退到 Hugging Face**。
- 提供三个函数：
  - `snapshot_download(repo_id, allow_file_pattern=None, ...)` — 下载整个或部分仓库
  - `get_repo_dir(repo_id)` — 整仓下载一次并缓存，返回本地目录（推荐，避免重复拉取）
  - `hf_hub_download(repo_id, filename, ...)` — 下载单个文件
- ID 映射通过 `HF_TO_MODELSCOPE` 字典配置：

```python
HF_TO_MODELSCOPE = {
    'ZhengPeng7/BiRefNet': 'modelscope/BiRefNet',
}
```

### 已接入的调用点

| 文件 | 用途 |
|---|---|
| `trellis2/pipelines/base.py` | 下载 `pipeline.json` |
| `trellis2/pipelines/__init__.py` | 下载 `pipeline.json` |
| `trellis2/models/__init__.py` | 下载各流模型 `.json` / `.safetensors` |
| `trellis2/modules/image_feature_extractor.py` | DINOv3 加载 |
| `trellis2/pipelines/rembg/BiRefNet.py` | 抠图模型加载 |
| `trellis2/trainers/flow_matching/mixins/image_conditioned.py` | 训练侧 DINOv3 |
| `trellis2/trainers/flow_matching/mixins/text_conditioned.py` | 训练侧 CLIP |

> 应用层无需改动，`app.py` / `app_texturing.py` 的 `from_pretrained('microsoft/TRELLIS.2-4B')` 会自动走 ModelScope。

## 下载慢？—— 核心参数

ModelScope 默认对**单个大文件只开 1 条连接**下载（`MODELSCOPE_DOWNLOAD_PARALLELS=1`），大权重文件只有几百 KB/s；而 Hugging Face 用多段并行能到几百 MB/s。

`model_hub.py` 已在 `modelscope` 首次 import 前设好默认值：

```python
os.environ.setdefault('MODELSCOPE_DOWNLOAD_PARALLELS', '16')              # 大文件并行连接数
os.environ.setdefault('MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB', '128') # >128MB 走并行
```

`max_workers`（跨文件并发）默认为 8。

> ⚠️ 这些环境变量**只在 `modelscope` 第一次 import 时读取一次**。若 `modelscope` 在此之前已被 import，再设置不会生效，必须通过 shell / 容器环境变量注入。

## 环境变量一览

| 变量 | 说明 | 建议值 |
|---|---|---|
| `MODELSCOPE_DOWNLOAD_PARALLELS` | 单个大文件的并行连接数 | `16`（带宽大可到 `32`） |
| `MODELSCOPE_MAX_WORKERS` | 同时下载的文件数 | `8` |
| `MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB` | 高于该大小走并行下载 | `128` ~ `500` |
| `MODELSCOPE_CACHE` | 模型缓存目录 | 指向大容量、高速磁盘/卷 |
| `MODELSCOPE_API_TOKEN` | 账号访问令牌（仅授权模型需要） | 无 |

## 缓存目录（MODELSCOPE_CACHE）

默认缓存位于 `~/.cache/modelscope/hub`。把它指到大容量、高速的磁盘/卷上（K8s 建议挂载持久卷），命中缓存后二次运行不再下载。

**命令行：**
```bash
export MODELSCOPE_CACHE=/data/modelscope
```

**Python（需在首次下载前设置）：**
```python
import os
os.environ['MODELSCOPE_CACHE'] = '/data/modelscope'
```

**K8s Pod：**
```yaml
spec:
  containers:
    - name: trellis2
      env:
        - name: MODELSCOPE_CACHE
          value: /data/modelscope
      volumeMounts:
        - name: model-cache
          mountPath: /data/modelscope
  volumes:
    - name: model-cache
      persistentVolumeClaim:
        claimName: trellis2-cache-pvc
```

> 切换缓存目录会重新下载；建议**第一次下载前**就设置好。TRELLIS.2-4B 整仓较大（预估 30GB+），请预留足够空间。

## Token 与权限

- ModelScope 的 `MODELSCOPE_API_TOKEN` 对应 HF 的 `HF_TOKEN`，**只负责权限，不负责加速**。
- 本项目的模型均为公开模型，**不需要 token**。
- 仅当访问「受限/授权」或「私有」模型时，才需要 token，并且你的账号须先被该模型授权。
- 只需要「下载」的话，**只读（Read）令牌即可**，无需读写（Write）权限。

获取方式：
```bash
modelscope login          # 交互式登录，令牌写入 ~/.modelscope/ 下
# 或
export MODELSCOPE_API_TOKEN=<你的token>
```

## 常见问题

**1. 只有几百 KB/s**
先确认并发生效：在 pod 里
```bash
python -c "from modelscope.hub.constants import MODELSCOPE_DOWNLOAD_PARALLELS, MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB; print(MODELSCOPE_DOWNLOAD_PARALLELS, MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB)"
```
应输出 `16 128`。若仍是 `1 500`，说明环境变量未在 import 前注入，请用 shell / 容器 env 设置。
若并发已调大仍慢，通常是 **pod 出口带宽低** 或 **跨地域链路**（海外 region 访问国内 OSS），建议预先下载到本地镜像/持久卷，运行时用 `local_files_only`。

**2. 某仓库 ModelScope 没有**
代码会自动回退到 Hugging Face（例如部分未镜像的仓库）。

**3. 403 GatedRepoError**
旧版 `modelscope` 没有对应仓库时可能回退到 HF 仍遇 gated 仓库。请升级：
```bash
pip install -U modelscope
```
并确认该仓库在 ModelScope 存在且公开。

**4. 看不到实时下载速度和进度**
ModelScope 默认用 `tqdm` 显示下载进度，但 tqdm 在**无 TTY 的容器/日志环境（如 K8s pod、后台运行）会自动禁用**，所以日志里看不到进度和速度。

`model_hub.py` 已内置一个控制台进度回调（`_ConsoleProgressCallback`），每次下载会打印：
```
[ModelScope] 开始下载 xxx.safetensors (1234.5 MB)
[ModelScope] xxx.safetensors: 512.0/1234.5 MB (41.5%)  45.20 MB/s
[ModelScope] 完成 xxx.safetensors (1234.5 MB, 用时 27.4s)
```
- 在有 TTY 的终端里，仍由 tqdm 显示美观进度条；在无 TTY 环境则自动切换到控制台输出。
- 若仍不显示，请确认 stdout 未被重定向丢弃，或临时用 `python -u app.py` 关闭缓冲。

## 附：首次运行提速思路

- 预先在构建镜像或初始化容器时用 `snapshot_download` 把模型下载到持久卷。
- 运行时把 `MODELSCOPE_CACHE` 指向该卷，并开启 `HF_HUB_OFFLINE` / `local_files_only` 模式，避免每次联网。
