import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from face_extractor import FaceExtractor

def prepare_region_dataset(input_dir="data", output_base_dir="data_regions", region="skin"):
    """
    Priprema dataset sa izdvojenim regionima lica
    """
    extractor = FaceExtractor()
    splits = ['train', 'val', 'test']
    seasons = ['spring', 'summer', 'autumn', 'winter']
    
    for split in splits:
        for season in seasons:
            input_path = os.path.join(input_dir, split, season)
            if not os.path.exists(input_path):
                continue
            
            # Za 'all' opciju, čuvamo u više foldera
            if region == 'all':
                output_paths = {
                    'skin': os.path.join(output_base_dir, 'skin', split, season),
                    'hair': os.path.join(output_base_dir, 'hair', split, season),
                    'left_eye': os.path.join(output_base_dir, 'left_eye', split, season),
                    'right_eye': os.path.join(output_base_dir, 'right_eye', split, season)
                }
                for path in output_paths.values():
                    os.makedirs(path, exist_ok=True)
            else:
                output_path = os.path.join(output_base_dir, region, split, season)
                os.makedirs(output_path, exist_ok=True)
            
            images = [f for f in os.listdir(input_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\nObrada: {split}/{season} - {len(images)} slika")
            
            for img_name in tqdm(images):
                img_path = os.path.join(input_path, img_name)
                
                result = extractor.extract_features(img_path, visualize=False)
                
                if result is None:
                    print(f"  ❌ Lice nije detektovano: {img_name}")
                    continue
                
                if region == 'all':
                    base_name = os.path.splitext(img_name)[0]
                    
                    # Sačuvaj kožu
                    skin_only = cv2.bitwise_and(result['original_image'], 
                                               result['original_image'],
                                               mask=result['skin_mask'])
                    cv2.imwrite(os.path.join(output_paths['skin'], 
                                           f"{base_name}.jpg"), skin_only)
                    
                    # Sačuvaj kosu
                    hair_only = cv2.bitwise_and(result['original_image'],
                                               result['original_image'],
                                               mask=result['hair_mask'])
                    cv2.imwrite(os.path.join(output_paths['hair'],
                                           f"{base_name}.jpg"), hair_only)
                    
                    # Sačuvaj oči
                    if result['left_eye'].size > 0:
                        cv2.imwrite(os.path.join(output_paths['left_eye'],
                                               f"{base_name}.jpg"), result['left_eye'])
                    if result['right_eye'].size > 0:
                        cv2.imwrite(os.path.join(output_paths['right_eye'],
                                               f"{base_name}.jpg"), result['right_eye'])
                else:
                    if region == 'skin':
                        output_img = cv2.bitwise_and(result['original_image'], 
                                                    result['original_image'],
                                                    mask=result['skin_mask'])
                    elif region == 'hair':
                        output_img = cv2.bitwise_and(result['original_image'],
                                                    result['original_image'],
                                                    mask=result['hair_mask'])
                    elif region == 'left_eye':
                        output_img = result['left_eye']
                    elif region == 'right_eye':
                        output_img = result['right_eye']
                    else:
                        continue
                    
                    if output_img.size > 0:
                        cv2.imwrite(os.path.join(output_path, img_name), output_img)
    
    extractor.close()
    print(f"\n✅ Dataset za region '{region}' je spreman!")

if __name__ == "__main__":
    # Pripremi sve regione odjednom
    prepare_region_dataset("data", "data_regions_all", region="all")
    
    print("\n📁 Struktura foldera:")
    print("data_regions_all/")
    print("├── skin/")
    print("│   ├── train/spring/")
    print("│   ├── train/summer/")
    print("│   ├── ...")
    print("├── hair/")
    print("├── left_eye/")
    print("└── right_eye/")