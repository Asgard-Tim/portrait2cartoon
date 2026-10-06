# 人脸素描系统

打开摄像头或导入一张照片，在画面合格时去掉背景，并用卷积网络生成白底人物线稿。

项目内容已上传至Github：https://github.com/Asgard-Tim/portrait2cartoon

## 项目思路

系统分两条路径，素描网络相同。

1. **摄像头。** 左侧实时显示画面，大约每 0.2 秒检测一次。画面需要清晰、亮度适中、只有一张足够大的人脸，并且眼睛睁开、带有微笑。条件连续满足约 0.6 秒后定格。左侧停在这一帧，便于和右侧线稿对照。点击「重新拍摄」后才继续捕捉。
2. **导入图片。** 不检查表情和画面质量，直接生成线稿。

定格或导入之后会做三件事：把原图存到 `input/`，用人物分割去掉背景并裁到人像附近，再调用素描网络，结果存到 `output/`。界面上的「另存为」可以把线稿存到其他位置。

## 目录

| 路径 | 作用 |
| --- | --- |
| `gui.py` | 摄像头、检测、去背景和界面 |
| `sketch_net.py` | 素描卷积网络 |
| `models/` | 人脸关键点、人物分割和线稿生成器权重 |
| `input/` | 摄像头原图、导入原图，以及示例 `demo.png` |
| `output/` | 生成的线稿 |
| `reference/` | Chan 等人 CVPR 2022 论文原文 |

线稿权重是 `models/sk_model.pth`。人脸检测使用 `models/face_landmarker.task`，去背景使用 `models/selfie_segmenter.tflite`。

## 安装

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在项目目录执行：

```bash
uv sync
```

依赖写在 `pyproject.toml` 里：PyTorch，以及固定版本的 MediaPipe 0.10.35。OpenCV 会随 MediaPipe 安装。不要再另装一份 OpenCV，否则两个包会冲突。

本机有 NVIDIA GPU 时使用 CUDA，Apple 芯片使用 MPS，否则使用 CPU。

## 使用

启动界面：

```bash
uv run python gui.py
```

macOS 若提示无法打开摄像头，到「系统设置 → 隐私与安全性 → 摄像头」允许当前应用，然后重新打开程序。

界面按钮：

- **导入图片**：选择一张本地照片，跳过微笑等检测，直接生成线稿。
- **重新拍摄**：清除当前结果，左侧恢复实时画面。
- **另存为**：把当前线稿保存到指定路径。生成成功后才会启用。

只对示例图跑素描网络、不做检测和去背景：

```bash
uv run python sketch_net.py
```

它读取 `input/demo.png`，写出 `output/demo_sketch.png`。

## 素描图像是怎么生成的

素描部分遵循 Caroline Chan、Frédo Durand、Phillip Isola 的 CVPR 2022 论文 *Learning to Generate Line Drawings that Convey Geometry and Semantics*（Informative Drawings）。原文在 `reference/Chan_Learning_to_Generate_Line_Drawings_CVPR2022.pdf`。

论文在训练生成器时用两类损失，让线条既顺着几何结构，也保留五官、头发、衣服这些语义。几何分支约束深度和法线，语义分支让生成图仍能被识别成原来的内容。这些损失只用于训练。测试时不再计算它们，只把照片送进已经训练好的生成器。

本项目的测试步骤与论文实现一致：

1. 用双三次插值把图像短边缩放到 256，并让宽高都能被 4 整除。生成器有两次步长为 2 的下采样，输入尺寸需要是 4 的倍数。
2. 像素从 0–255 归一化到 0–1，作为三通道张量输入。
3. 生成器是全卷积网络：7×7 卷积升到 64 通道，两次下采样到 256 通道，3 个残差块，两次转置卷积恢复分辨率，最后用 Sigmoid 得到单通道线稿。输出接近 1 的地方是白纸，接近 0 的地方是线条。
4. 界面再把单通道结果存成普通图片。分辨率保持生成器的输出尺寸，不再放大回原图，以免线条发虚。
