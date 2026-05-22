import cv2
import numpy as np
import argparse
from pathlib import Path


def odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def remove_small_components(mask, min_area=20):
    """
    移除太小的白色雜點。
    mask: 白線=255，背景=0
    """
    if min_area <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def enhance_white_line_image(
    input_path,
    output_path,
    scale=4,
    denoise=7,
    top_hat_size=17,
    threshold_mode="otsu",
    manual_threshold=35,
    close_iter=1,
    stroke_adjust=0,
    remove_small=20,
    invert_output=False
):
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    # 1. 灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 先放大，避免二值化後再放大產生大鋸齒
    if scale > 1:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LANCZOS4
        )

    # 3. 降噪，避免 JPG 顆粒被當成線條
    if denoise > 0:
        gray = cv2.fastNlMeansDenoising(gray, None, h=denoise)

    # 4. 強化白色細線
    # top-hat 會把黑底上的亮細線抓出來
    kernel_size = odd(top_hat_size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    enhanced = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # 5. 拉開對比
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    # 6. 二值化
    if threshold_mode == "otsu":
        _, mask = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold_mode == "manual":
        _, mask = cv2.threshold(
            enhanced,
            manual_threshold,
            255,
            cv2.THRESH_BINARY
        )
    else:
        raise ValueError("threshold_mode must be 'otsu' or 'manual'")

    # 7. 接合微小斷線
    if close_iter > 0:
        close_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iter
        )

    # 8. 移除小雜點
    mask = remove_small_components(mask, min_area=remove_small)

    # 9. 調整線寬
    # 正數：線變粗
    # 負數：線變細
    if stroke_adjust != 0:
        k = np.ones((3, 3), np.uint8)

        if stroke_adjust > 0:
            mask = cv2.dilate(mask, k, iterations=stroke_adjust)
        else:
            mask = cv2.erode(mask, k, iterations=abs(stroke_adjust))

    # 10. 輸出
    # 預設：黑底白線
    # invert_output=True：白底黑線
    if invert_output:
        result = 255 - mask
    else:
        result = mask

    cv2.imwrite(str(output_path), result)
    print(f"Saved: {output_path}")
    print(f"Output size: {result.shape[1]} x {result.shape[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance blurry white line art on black background."
    )

    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", default="enhanced.png", help="Output image path")

    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--denoise", type=int, default=7, help="Denoise strength")
    parser.add_argument("--top-hat-size", type=int, default=17, help="White line extraction kernel size")

    parser.add_argument(
        "--threshold-mode",
        choices=["otsu", "manual"],
        default="otsu"
    )

    parser.add_argument("--manual-threshold", type=int, default=35)
    parser.add_argument("--close", type=int, default=1, help="Connect broken lines")
    parser.add_argument("--stroke", type=int, default=0, help="Positive = thicker, negative = thinner")
    parser.add_argument("--remove-small", type=int, default=20, help="Remove small white noise")
    parser.add_argument("--invert-output", action="store_true", help="Output white background with black lines")

    args = parser.parse_args()

    enhance_white_line_image(
        input_path=Path(args.input),
        output_path=Path(args.output),
        scale=args.scale,
        denoise=args.denoise,
        top_hat_size=args.top_hat_size,
        threshold_mode=args.threshold_mode,
        manual_threshold=args.manual_threshold,
        close_iter=args.close,
        stroke_adjust=args.stroke,
        remove_small=args.remove_small,
        invert_output=args.invert_output
    )


if __name__ == "__main__":
    main()