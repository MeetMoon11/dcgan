# DCGAN V2 Step1：fashion-pattern-images 自检与预览

本步骤仅做：**可访问性验证、标签统计、可视化预览、19->11映射模板生成**。  
不做训练、不做 patch 裁切、不引入 OpenCV/分割模型。

## 1) 安装依赖

在仓库根目录执行（Windows + 指定解释器示例）：

```powershell
C:/ProgramData/anaconda3/envs/ml/python.exe -m pip install -r requirements_v2.txt
```

## 2) Hugging Face gated 数据集访问前置条件（非常重要）

数据集 `yainage90/fashion-pattern-images` 是 gated dataset。若未同意访问条件，会报 403。

请先完成：

1. 打开数据集页面并点击 **Agree / Access**：  
   https://huggingface.co/datasets/yainage90/fashion-pattern-images
2. 使用本机登录 Hugging Face（Read token 即可）：

```powershell
C:/ProgramData/anaconda3/envs/ml/python.exe -m huggingface_hub login
# 或
hf auth login
```

参考文档：
- https://huggingface.co/docs/hub/en/security-tokens
- https://huggingface.co/docs/hub/en/datasets-usage

## 3) 运行脚本

默认命令（仓库根目录）：

```powershell
C:/ProgramData/anaconda3/envs/ml/python.exe scripts_v2/step1_inspect_fashion_pattern_images.py
```

或绝对路径示例：

```powershell
& C:/ProgramData/anaconda3/envs/ml/python.exe d:/work/Aurora/dcgan/scripts_v2/step1_inspect_fashion_pattern_images.py
```

可选参数：

```powershell
C:/ProgramData/anaconda3/envs/ml/python.exe scripts_v2/step1_inspect_fashion_pattern_images.py `
  --dataset yainage90/fashion-pattern-images `
  --split train `
  --out_dir outputs_v2 `
  --max_preview_per_label 12 `
  --seed 42
```

## 4) 运行成功验收标准

执行后应生成：

1. `outputs_v2/fashion_pattern_images_label_stats.json`
2. `outputs_v2/pattern19_to_style11_template.json`
3. `outputs_v2/preview_fashion_patterns/grid_by_label.png`
4. `outputs_v2/preview_fashion_patterns/<label_name>/sample_*.png`（每类至少1张）
5. 终端打印 19 类 label 列表及 count

如果出现 403 / 权限错误，请先检查是否已完成网页同意 + 本机 token 登录。
