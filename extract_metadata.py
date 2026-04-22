import os
import glob
import json
import time
from pathlib import Path
import google.generativeai as genai
from PIL import Image

API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)

# Using Gemini 3.1 Flash Image Preview for multimodal JSON output
model = genai.GenerativeModel("models/gemini-3.1-flash-image-preview", generation_config={"response_mime_type": "application/json"})

PROJECT_ROOT = r"c:\Users\SEOP\Desktop\Embedding_Image\project"
OUTPUT_FILE = r"c:\Users\SEOP\Desktop\Embedding_Image\extracted_metadata.json"

PROMPT = """
당신은 SEOP 건축사사무소의 시니어 건축 시각화 전문가입니다. 제공된 투시도/조감도 이미지를 분석하여 회사의 고유한 디자인 언어와 렌더링 스타일을 추출하세요.
반드시 아래의 JSON 형식으로만 답변해야 합니다.

{
  "project_name": "프로젝트 이름 (프로젝트 폴더명 기반 추출 혹은 추론)",
  "project_usage": "프로젝트 용도 (프로젝트 폴더명 기반 추출 혹은 추론)",
  "camera_angle": "카메라 구도 (예: 아이레벨 투시도, 버드아이 조감도, 파사드 정면뷰 등)",
  "massing_and_form": "건물의 형태적 특징 (예: 수직적 상승감, 곡선형 매스 분절, 필로티 구조 등)",
  "materiality": ["사용된 주된 마감재 3개 이하 배열 (예: 노출콘크리트, 롱브릭, 우드루버 등)"],
  "lighting_and_atmosphere": "빛의 방향과 전반적인 무드 (예: 매직아워의 따뜻함, 한낮의 강렬한 대비, 안개 낀 차분함 등)",
  "surroundings": "주변 컨텍스트 (예: 도심지 밀집 지역, 숲 속, 수변 공간 등)",
  "style_keywords": ["디자인 톤앤매너를 나타내는 형용사 3~5개 배열"],
  "embedding_text": "위의 모든 요소를 종합하여 3~4문장으로 서술한 디테일한 묘사 (이 텍스트가 모델에 의해 임베딩에 사용됩니다.)"
}
"""

def get_image_files(root_dir):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return files

def extract_metadata(image_path):
    print(f"Processing {image_path}...")
    try:
        img = Image.open(image_path)
        # Resize safely to maintain aspect ratio and save bandwidth
        img.thumbnail((1024, 1024))
        
        rel_dir = os.path.dirname(os.path.relpath(image_path, PROJECT_ROOT))
        folder_name = rel_dir.split(os.sep)[0] if rel_dir else ""
        
        context_prompt = f"이 이미지는 '{folder_name}' 관련 프로젝트에 속해 있습니다.\n" + PROMPT
        
        response = model.generate_content([context_prompt, img])
        
        # Parse output
        result = json.loads(response.text)
        # Store metadata
        result['image_path'] = image_path
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    image_files = get_image_files(PROJECT_ROOT)
    print(f"Found {len(image_files)} images.")
    
    metadata_list = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
        except:
            pass
            
    processed_paths = {m.get('image_path') for m in metadata_list}
    
    count = 0
    max_process = 10000  # Process all remaining images
    
    for img_path in image_files:
        if img_path in processed_paths:
            continue
            
        metadata = extract_metadata(img_path)
        if metadata:
            metadata_list.append(metadata)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            processed_paths.add(img_path)
            
        count += 1
        if count >= max_process:
            print(f"Reached maximum process limit ({max_process}) for this test run. Run again or increase max_process to process more.")
            break
            
        time.sleep(2) # Prevent API rate limit
        
    print(f"Total processed metadata items saved: {len(metadata_list)}")

if __name__ == "__main__":
    main()
