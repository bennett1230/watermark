# 数字水印系统实现总结

## 功能概述
本系统实现了基于DCT(离散余弦变换)和DWT(离散小波变换)的数字水印嵌入与提取算法，并提供了一套完整的鲁棒性测试框架。

## 核心功能

### 1. 水印嵌入
- **DCT嵌入**：将水印图像通过DCT变换嵌入到宿主图像的频域
- **DWT嵌入**：将水印图像通过小波变换嵌入到宿主图像的高频子带

### 2. 水印提取
- **DCT提取**：从含水印图像中提取DCT域水印
- **DWT提取**：从含水印图像中提取DWT域水印

### 3. 鲁棒性测试
系统支持多种攻击测试：
- **噪声攻击**：高斯、盐、胡椒、盐和胡椒、斑点噪声
- **几何攻击**：旋转、缩放、平移
- **图像处理攻击**：对比度调整、亮度调整、裁剪

## 使用方法

```python
# 初始化水印器
wm = ImageWatermarker()

# 加载图像
wm.load_images("original.png", "watermark.png")

# DCT嵌入
watermarked = wm.embed_dct(alpha=0.1)

# DWT嵌入
watermarked = wm.embed_dwt(alpha=0.1)

# 提取水印(DCT)
extracted = wm.extract_dct(watermarked_img)

# 提取水印(DWT)
extracted = wm.extract_dwt(watermarked_img)

# 鲁棒性测试
test_results = wm.robustness_test(watermarked_img, "test_output")
