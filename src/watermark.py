import warnings
import cv2
import numpy as np
import pywt
import os
from skimage import util, exposure, metrics
from skimage.transform import rotate, resize, AffineTransform, warp
import matplotlib.pyplot as plt

class ImageWatermarker:
    def __init__(self):
        self.original_img = None
        self.watermark_img = None
        self.watermarked_img = None

    def load_images(self, original_path, watermark_path, grayscale=True):
        if grayscale:
            self.original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            self.watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.original_img = cv2.imread(original_path)
            self.watermark_img = cv2.imread(watermark_path)

        if self.original_img is not None and self.watermark_img is not None:
            self.watermark_img = cv2.resize(self.watermark_img,
                                            (self.original_img.shape[1], self.original_img.shape[0]))
        return self.original_img is not None and self.watermark_img is not None

    def embed_dct(self, alpha=0.1):
        if self.original_img is None or self.watermark_img is None:
            raise ValueError("请先加载图像")

        original_float = np.float32(self.original_img) / 255.0
        watermark_float = np.float32(self.watermark_img) / 255.0

        original_dct = cv2.dct(original_float)
        watermark_dct = cv2.dct(watermark_float)

        watermarked_dct = original_dct + alpha * watermark_dct
        watermarked = cv2.idct(watermarked_dct)
        self.watermarked_img = np.uint8(np.clip(watermarked * 255, 0, 255))
        return self.watermarked_img

    def embed_dwt(self, alpha=0.1, wavelet='haar'):
        if self.original_img is None or self.watermark_img is None:
            raise ValueError("请先加载图像")

        coeffs_original = pywt.dwt2(self.original_img, wavelet)
        cA, (cH, cV, cD) = coeffs_original

        # 将水印缩放到 cH 的尺寸（注意顺序，cv2.resize的参数是宽,高）
        watermark_resized = cv2.resize(self.watermark_img, (cH.shape[1], cH.shape[0]))

        # 这里不再对水印做DWT，直接用缩放后的水印作为细节调整
        cH_new = cH + alpha * watermark_resized
        cV_new = cV + alpha * watermark_resized

        watermarked = pywt.idwt2((cA, (cH_new, cV_new, cD)), wavelet)
        self.watermarked_img = np.uint8(np.clip(watermarked, 0, 255))
        return self.watermarked_img

    def extract_dct(self, watermarked_img, original_img=None, alpha=0.1):
        original = original_img if original_img is not None else self.original_img
        if original is None:
            raise ValueError("请先加载原始图像")

        if original.shape != watermarked_img.shape:
            original = cv2.resize(original, (watermarked_img.shape[1], watermarked_img.shape[0]))

        original_float = np.float32(original) / 255.0
        watermarked_float = np.float32(watermarked_img) / 255.0

        original_dct = cv2.dct(original_float)
        watermarked_dct = cv2.dct(watermarked_float)

        watermark_dct = (watermarked_dct - original_dct) / alpha
        watermark = cv2.idct(watermark_dct)
        watermark = np.uint8(np.clip(watermark * 255, 0, 255))
        return watermark

    def extract_dwt(self, watermarked_img, original_img=None, alpha=0.1, wavelet='haar'):
        original = original_img if original_img is not None else self.original_img
        if original is None:
            raise ValueError("缺少原始图像")

        if original.shape != watermarked_img.shape:
            original = cv2.resize(original, (watermarked_img.shape[1], watermarked_img.shape[0]))

        coeffs_original = pywt.dwt2(original, wavelet)
        _, (cH_o, cV_o, _) = coeffs_original

        coeffs_watermarked = pywt.dwt2(watermarked_img, wavelet)
        _, (cH_w, cV_w, _) = coeffs_watermarked

        cH_watermark = (cH_w - cH_o) / alpha
        cV_watermark = (cV_w - cV_o) / alpha

        dummy_cA = np.zeros_like(cH_watermark)
        dummy_cD = np.zeros_like(cH_watermark)

        watermark = pywt.idwt2((dummy_cA, (cH_watermark, cV_watermark, dummy_cD)), wavelet)
        watermark = np.uint8(np.clip(watermark, 0, 255))
        return watermark

    def robustness_test(self, watermarked_img, output_dir, tests_to_run='all'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        test_cases = {}
        height, width = watermarked_img.shape[:2]
        crop_size = min(50, height // 2 - 1, width // 2 - 1)

        if tests_to_run == 'all' or 'noise' in tests_to_run:
            test_cases.update({
                'noise_gaussian': util.random_noise(watermarked_img, mode='gaussian'),
                'noise_salt': util.random_noise(watermarked_img, mode='salt'),
                'noise_pepper': util.random_noise(watermarked_img, mode='pepper'),
                'noise_s&p': util.random_noise(watermarked_img, mode='s&p'),
                'noise_speckle': util.random_noise(watermarked_img, mode='speckle')
            })

        if tests_to_run == 'all' or 'geometric' in tests_to_run:
            test_cases.update({
                'rotated_5': rotate(watermarked_img, angle=5, preserve_range=True).astype(np.uint8),
                'rotated_30': rotate(watermarked_img, angle=30, preserve_range=True).astype(np.uint8),
                'scaled_down': resize(watermarked_img, (height // 2, width // 2), preserve_range=True).astype(np.uint8),
                'scaled_up': resize(watermarked_img, (height * 2, width * 2), preserve_range=True).astype(np.uint8),
                'translated': warp(watermarked_img, AffineTransform(translation=(10, 20)), preserve_range=True).astype(np.uint8)
            })

        if tests_to_run == 'all' or 'processing' in tests_to_run:
            test_cases.update({
                'contrast_low': exposure.adjust_gamma(watermarked_img, gamma=2.0),
                'contrast_high': exposure.adjust_gamma(watermarked_img, gamma=0.5),
                'bright_low': exposure.adjust_log(watermarked_img, gain=0.5),
                'bright_high': exposure.adjust_log(watermarked_img, gain=1.5),
                'cropped': watermarked_img[crop_size:-crop_size, crop_size:-crop_size] if height > 2 * crop_size and width > 2 * crop_size else watermarked_img[10:-10, 10:-10]
            })

        test_paths = {}
        for name, img in test_cases.items():
            try:
                if isinstance(img, np.ndarray) and img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                test_path = os.path.join(output_dir, f"{name}.png")
                cv2.imwrite(test_path, img)
                test_paths[name] = test_path
            except Exception as e:
                print(f"无法保存测试图像 {name}: {str(e)}")
                continue

        return test_paths

    @staticmethod
    def evaluate_extraction(original_watermark, extracted_watermark):
        if extracted_watermark is None:
            return {'ssim': 0, 'mse': float('inf'), 'correlation': 0, 'psnr': 0, 'success': False}

        original_resized = cv2.resize(original_watermark, (extracted_watermark.shape[1], extracted_watermark.shape[0]))

        # 确保输入为 uint8 类型
        original_resized = original_resized.astype(np.uint8)
        extracted_watermark = extracted_watermark.astype(np.uint8)

        try:
            ssim_value = metrics.structural_similarity(
                original_resized, extracted_watermark,
                data_range=255,
                channel_axis=None
            )
            if np.isnan(ssim_value):
                ssim_value = 0
        except Exception:
            ssim_value = 0

        mse_value = metrics.mean_squared_error(original_resized, extracted_watermark)

        # 计算相关系数，防止异常
        try:
            correlation = np.corrcoef(original_resized.flatten(), extracted_watermark.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0
        except Exception:
            correlation = 0

        psnr = 20 * np.log10(255.0 / np.sqrt(mse_value)) if mse_value != 0 else float('inf')

        return {'ssim': ssim_value, 'mse': mse_value, 'correlation': correlation, 'psnr': psnr, 'success': True}

    @staticmethod
    def display_images(images, titles=None, cols=2, figsize=(15, 10)):
        plt.figure(figsize=figsize)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        num_images = len(images)
        rows = (num_images + cols - 1) // cols

        for i, (key, img) in enumerate(images.items()):
            plt.subplot(rows, cols, i + 1)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            title = titles[i] if titles and i < len(titles) else key
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def simple_rotation_correction(img, angle):
        # 中心点旋转回去
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated


def interactive_test():
    wm = ImageWatermarker()

    print("=== 图片水印系统交互式测试 ===")

    base_dir = r"E:\Project\project2\examples"  # 默认路径

    original_path = input(f"请输入原始图像路径(默认为 {os.path.join(base_dir, 'original.png')}): ") or os.path.join(base_dir, "original.png")
    watermark_path = input(f"请输入水印图像路径(默认为 {os.path.join(base_dir, 'watermark.png')}): ") or os.path.join(base_dir, "watermark.png")

    if not wm.load_images(original_path, watermark_path):
        print("无法加载图像，请检查路径")
        return

    wm.display_images({
        "原图": wm.original_img,
        "水印图": wm.watermark_img
    }, ["原始图像", "水印图像"])

    method = input("选择嵌入方式 (1=DCT, 2=DWT，默认1): ") or "1"
    alpha = float(input("请输入嵌入强度 alpha (默认0.1): ") or 0.1)

    if method == "2":
        print("正在使用 DWT 嵌入...")
        wm.embed_dwt(alpha=alpha)
        output_filename = os.path.join(base_dir, "watermarked_output_dwt.png")
    else:
        print("正在使用 DCT 嵌入...")
        wm.embed_dct(alpha=alpha)
        output_filename = os.path.join(base_dir, "watermarked_output_dct.png")

    wm.display_images({
        "含水印图": wm.watermarked_img
    }, ["水印图像"])

    save = input("是否保存水印图像？(y/n，默认y): ") or "y"
    if save.lower() == "y":
        cv2.imwrite(output_filename, wm.watermarked_img)
        print(f"已保存为 {output_filename}")

    test = input("是否进行鲁棒性测试？(y/n，默认n): ") or "n"
    if test.lower() == "y":
        warnings.filterwarnings("ignore")  # 忽略所有警告

        test_results = wm.robustness_test(wm.watermarked_img, os.path.join(base_dir, "robustness_tests"))


        name_map = {
            'noise_gaussian': '噪声（高斯）',
            'noise_salt': '噪声（盐）',
            'noise_pepper': '噪声（胡椒）',
            'noise_s&p': '噪声（盐和胡椒）',
            'noise_speckle': '噪声（斑点）',
            'rotated_5': '旋转 5 度',
            'rotated_30': '旋转 30 度',
            'scaled_down': '缩小',
            'scaled_up': '放大',
            'translated': '平移',
            'contrast_low': '对比度降低',
            'contrast_high': '对比度增强',
            'bright_low': '亮度降低',
            'bright_high': '亮度增强',
            'cropped': '裁剪'
        }

        with np.errstate(divide='ignore', invalid='ignore'):
            for name, path in test_results.items():
                attacked_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if attacked_img is None:
                    continue
                if method == "2":
                    extracted = wm.extract_dwt(attacked_img, alpha=alpha)
                else:
                    extracted = wm.extract_dct(attacked_img, alpha=alpha)

                eval_res = wm.evaluate_extraction(wm.watermark_img, extracted)

                display_name = name_map.get(name, name)
                print(f"[{display_name}] SSIM: {eval_res['ssim']:.4f}, PSNR: {eval_res['psnr']:.2f}, 相关性: {eval_res['correlation']:.4f}")

if __name__ == "__main__":
    interactive_test()