import os
import json
import base64
from PIL import Image
from io import BytesIO

# 경로 설정 (절대 경로 유지)
METADATA_FILE = r"c:\Users\SEOP\Desktop\Embedding_Image\extracted_metadata.json"
OPTIMIZED_FILE = r"c:\Users\SEOP\Desktop\Embedding_Image\optimized_metadata.json"

def get_tiny_thumbnail_b64(image_path):
    """이미지를 아주 작은 사이즈로 리사이징하여 Base64로 변환"""
    if not os.path.exists(image_path):
        return ""
    try:
        img = Image.open(image_path)
        # 투명색 처리 및 RGB 변환
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # 600px 수준의 썸네일 (화질과 성능의 균형)
        img.thumbnail((600, 600))
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"File not found: {METADATA_FILE}")
        return

    print("Loading original metadata...")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting optimization for {len(data)} images...")
    
    for idx, item in enumerate(data):
        img_path = item.get("image_path", "")
        if img_path:
            # 썸네일 생성 및 데이터 추가
            item["thumbnail_b64"] = get_tiny_thumbnail_b64(img_path)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(data)}...")

    with open(OPTIMIZED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nOptimization Complete! Saved to: {OPTIMIZED_FILE}")

if __name__ == "__main__":
    main()
