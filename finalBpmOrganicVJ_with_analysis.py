#!/usr/bin/env python
"""最終版 BPM対応オーガニックレイアウトVJ - 動作解析機能付き"""

import cv2
import numpy as np
import random
import glob
import time
from typing import List, Dict, Tuple
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import platform

class FinalBpmOrganicVJWithAnalysis:
    """最終版 BPM同期オーガニックVJ with 動作解析"""
    
    def __init__(self, bpm: int = 120):
        self.video_caps = []
        self.bpm = bpm
        self.beat_interval = 60.0 / bpm
        self.last_update = 0
        self.current_layout = None
        self.beat_multiplier = 4  # 何ビートごとに更新するか（デフォルト4拍=1小節）
        
        # TAP BPM用
        self.tap_times = deque(maxlen=8)
        self.last_tap_time = 0
        
        # 動画再生管理用
        self.video_positions = []  # 各動画の現在位置
        self.crop_configs = []  # 現在のクロップ設定
        self.layout_regions = []  # 現在のレイアウト領域
        self.fixed_sizes = []  # 固定された枠サイズ
        self.frame_effects = []  # 各枠のエフェクト設定
        self.playback_speed = 2.0  # 再生速度（2.0 = 2倍速）
        
        # 動作解析用
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # 各動画ごとに背景減算器を用意
        self.bg_subtractors = []
        self.motion_area_threshold = 300  # 初期値を300に変更
        self.frame_count = 0
        self.prev_faces = []
        
        # 背景色設定
        self.background_colors = {
            'black': (0, 0, 0),           # 黒（デフォルト）
            'red': (0, 0, 255),           # 赤（光の三原色）
            'green': (0, 255, 0),         # 緑（光の三原色）
            'blue': (255, 0, 0),          # 青（光の三原色）
            'cyan': (255, 255, 0),        # シアン（色の三原色）
            'magenta': (255, 0, 255),     # マゼンタ（色の三原色）
            'yellow': (0, 255, 255)       # イエロー（色の三原色）
        }
        self.current_bg_color = 'black'  # デフォルトは黒
        
        # キーワードリスト（sakugabooruのタグを参考に拡充）
        self.action_keywords = {
            'face': ['顔認識', '人物検出', 'フェイス', 'ヒューマン', '表情解析', 
                    'character_acting', 'crying', 'emotion', 'performance'],
            'motion': ['動作検出', 'モーション', '移動体', 'ムーブメント', '動体追跡',
                      'smears', 'impact_frames', 'rotation', 'morphing'],
            'large_motion': ['激しい動き', 'ダイナミック', '高速移動', 'アクション', 'エネルギー',
                           'fighting', 'explosions', 'debris', 'beams', 'lightning'],
            'small_motion': ['微細な動き', '繊細', 'デリケート', '小刻み', 'ソフト',
                           'hair', 'fabric', 'liquid', 'wind', 'sparks'],
            'effects': ['effects', 'smoke', 'fire', 'water', 'particles',
                       'エフェクト', '煙', '炎', '水', 'パーティクル'],
            'action': ['running', 'walk_cycle', 'flying', 'falling', 'dancing',
                      'ランニング', '歩行', '飛行', '落下', 'ダンス'],
            'technique': ['background_animation', 'cgi', '3d_background', 'kanada_light_flare',
                         '背景動画', 'CGI', '3D背景', 'カナダ光'],
            'creature': ['creatures', 'mecha', 'animals', 'vehicle', 'missiles',
                        'クリーチャー', 'メカ', '動物', '乗り物', 'ミサイル']
        }
        
        # フォント設定
        self.font_path = self._find_japanese_font()
        self.font_size = 36  # 大きくした
        self.small_font_size = 28  # 大きくした
        
    def _find_japanese_font(self):
        """利用可能な日本語フォントを検索"""
        system = platform.system()
        
        font_candidates = []
        if system == 'Darwin':  # macOS
            font_candidates = [
                "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",  # 明朝体
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # ゴシック体
                "/System/Library/Fonts/ヒラギノ角ゴシック.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
        elif system == 'Windows':
            font_candidates = [
                "C:/Windows/Fonts/msgothic.ttc",
                "C:/Windows/Fonts/meiryo.ttc",
                "C:/Windows/Fonts/YuGothic.ttc"
            ]
        else:  # Linux
            font_candidates = [
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf"
            ]
        
        for font_path in font_candidates:
            try:
                ImageFont.truetype(font_path, 20)
                print(f"フォントを使用: {font_path}")
                return font_path
            except:
                continue
        
        print("日本語フォントが見つかりません。英語表示になります。")
        return None
        
    def set_bpm(self, bpm: int):
        """BPMを設定"""
        self.bpm = max(60, min(200, bpm))
        self.beat_interval = 60.0 / self.bpm
        print(f"BPM: {self.bpm}")
        
    def tap_bpm(self):
        """TAPによるBPM検出"""
        current_time = time.time()
        
        if current_time - self.last_tap_time > 2.0:
            self.tap_times.clear()
        
        self.tap_times.append(current_time)
        self.last_tap_time = current_time
        
        if len(self.tap_times) >= 2:
            intervals = []
            for i in range(1, len(self.tap_times)):
                intervals.append(self.tap_times[i] - self.tap_times[i-1])
            
            avg_interval = sum(intervals) / len(intervals)
            calculated_bpm = int(60.0 / avg_interval)
            
            if 60 <= calculated_bpm <= 200:
                self.set_bpm(calculated_bpm)
    
    def setup_videos(self, video_paths: List[str], num_videos: int = 4):
        """動画をセットアップ"""
        for path in video_paths[:num_videos]:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.video_caps.append(cap)
                self.video_positions.append(0)
                # 各動画用の背景減算器を作成
                self.bg_subtractors.append(cv2.createBackgroundSubtractorMOG2(
                    history=500, 
                    varThreshold=16, 
                    detectShadows=False
                ))
                print(f"Loaded: {path}")
        return len(self.video_caps) > 0
    
    def detect_features(self, frame: np.ndarray, video_idx: int = 0) -> Dict:
        """フレームから特徴を検出"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 顔検出（3フレームごとに更新）
        faces = []
        if self.frame_count % 3 == 0:
            # 顔検出の最小サイズを調整可能に
            min_face_size = int(30 * (self.motion_area_threshold / 100))
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(min_face_size, min_face_size))
        
        # 動き検出（各動画用の背景減算器を使用）
        if video_idx < len(self.bg_subtractors):
            fg_mask = self.bg_subtractors[video_idx].apply(gray)
        else:
            # フォールバック
            fg_mask = np.zeros_like(gray)
        
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 動きの領域をフィルタリング
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append({
                    'rect': (x, y, w, h),
                    'area': area,
                    'type': 'large_motion' if area > 1000 else 'small_motion'
                })
        
        return {
            'faces': faces,
            'motion_regions': motion_regions
        }
    
    def draw_text_with_style(self, img: np.ndarray, text: str, pos: Tuple[int, int], 
                           font_size: int = None, color: Tuple[int, int, int] = (255, 255, 255),
                           style: str = 'gothic') -> np.ndarray:
        """スタイリッシュなテキストを描画"""
        if self.font_path is None:
            # フォントがない場合は通常のOpenCV描画
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            return img
        
        # PILで描画
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        if font_size is None:
            font_size = self.font_size
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        # 影付きテキスト
        shadow_offset = 2
        draw.text((pos[0] + shadow_offset, pos[1] + shadow_offset), text, 
                 fill=(0, 0, 0), font=font)
        draw.text(pos, text, fill=color[::-1], font=font)  # BGRからRGBに変換
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def apply_frame_effect(self, frame: np.ndarray, effect_type: str) -> np.ndarray:
        """フレームにエフェクトを適用"""
        if effect_type == 'none':
            return frame
        elif effect_type == 'highlight':
            # ハイライト効果
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.add(l, 30)  # 明度を上げる
            l[l > 255] = 255
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif effect_type == 'edge':
            # 輪郭抽出
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # 元の画像と合成
            return cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        elif effect_type == 'blur':
            # ぼかし効果
            return cv2.GaussianBlur(frame, (9, 9), 0)
        elif effect_type == 'contrast':
            # コントラスト強調
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif effect_type == 'posterize':
            # ポスタライズ効果
            return cv2.convertScaleAbs(frame // 32 * 32)
        elif effect_type == 'sketch':
            # 線画エフェクト（ペンシルスケッチ）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ガウシアンブラー適用
            blur = cv2.GaussianBlur(gray, (21, 21), 0, 0)
            # 除算で鉛筆画風に
            sketch = cv2.divide(gray, blur, scale=256)
            # 3チャンネルに変換
            return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        elif effect_type == 'pencil':
            # よりアーティスティックな鉛筆画
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # エッジ保持フィルタ
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 10)
            # 反転して線画に
            edges = cv2.bitwise_not(edges)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # 元画像と合成
            return cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
        elif effect_type == 'mask_circle':
            # 円形マスクエフェクト
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            # ぼかしを適用してソフトエッジに
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            mask = mask / 255.0
            mask = np.stack([mask] * 3, axis=-1)
            # 背景を暗くして適用
            result = frame * mask
            return result.astype(np.uint8)
        elif effect_type == 'mask_ellipse':
            # 楕円形マスクエフェクト
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            axes = (int(w * 0.4), int(h * 0.3))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            # ぼかしを適用
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
            mask = mask / 255.0
            mask = np.stack([mask] * 3, axis=-1)
            result = frame * mask
            return result.astype(np.uint8)
        elif effect_type == 'mask_vignette':
            # ビネット（周辺減光）エフェクト
            h, w = frame.shape[:2]
            # 放射状グラデーションマスクを作成
            Y, X = np.ogrid[:h, :w]
            center = (h // 2, w // 2)
            dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            mask = 1 - (dist_from_center / max_dist) ** 1.5
            mask = np.clip(mask, 0.3, 1.0)  # 最小値を0.3に
            mask = np.stack([mask] * 3, axis=-1)
            result = frame * mask
            return result.astype(np.uint8)
        else:
            return frame
    
    def draw_analysis_overlay(self, frame: np.ndarray, features: Dict) -> np.ndarray:
        """解析結果をオーバーレイ表示"""
        overlay = frame.copy()
        
        # 白色で統一
        color = (255, 255, 255)
        thickness = 1
        
        # 顔検出の表示
        for i, (x, y, w, h) in enumerate(features['faces']):
            # シンプルな実線の矩形
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            # キーワード表示
            keyword = random.choice(self.action_keywords['face'])
            overlay = self.draw_text_with_style(overlay, keyword, (x, y - 10), 
                                              self.small_font_size, color)
        
        # 動き検出の表示
        for motion in features['motion_regions']:
            x, y, w, h = motion['rect']
            motion_type = motion['type']
            area = motion['area']
            
            # 動きの大きさに応じてキーワードを選択
            if motion_type == 'large_motion':
                # 大きな動きの場合、複数のカテゴリから選択
                category = random.choice(['large_motion', 'effects', 'action'])
                keywords = self.action_keywords[category]
            else:
                # 小さな動きの場合も、複数のカテゴリから選択
                category = random.choice(['small_motion', 'technique'])
                keywords = self.action_keywords[category]
            
            # シンプルな実線の矩形
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            # キーワード表示
            keyword = random.choice(keywords)
            overlay = self.draw_text_with_style(overlay, keyword, (x, y - 10), 
                                              self.small_font_size, color)
        
        return overlay
    
    def _draw_dashed_rect(self, img: np.ndarray, pt1: Tuple[int, int], 
                         pt2: Tuple[int, int], color: Tuple[int, int, int], 
                         thickness: int, dash_length: int = 5):
        """点線の矩形を描画"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # 上辺
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        # 下辺
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        # 左辺
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        # 右辺
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)
    
    def generate_organic_layout(self, width: int, height: int, 
                              num_regions: int = 6) -> List[Dict]:
        """有機的なレイアウトを生成（矩形のみ）"""
        regions = []
        
        # サイズバリエーション（より大きく）
        size_variations = [
            (0.25, 0.35),   # 中
            (0.3, 0.4),     # 大
            (0.35, 0.45),   # 特大
            (0.4, 0.5),     # 超特大
            (0.25, 0.5),    # 縦長
            (0.5, 0.25),    # 横長
        ]
        
        for _ in range(num_regions):
            # ランダムなサイズ
            size_var = random.choice(size_variations)
            w = int(width * random.uniform(size_var[0], size_var[1]))
            h = int(height * random.uniform(size_var[0], size_var[1]))
            
            # ランダムな位置（画面外も許可してより有機的に）
            x = random.randint(-w//3, width - 2*w//3)
            y = random.randint(-h//3, height - 2*h//3)
            
            regions.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })
        
        return regions
    
    def extract_random_crops(self, frames: List[Tuple[np.ndarray, Dict]]) -> List[Dict]:
        """各フレームからランダムクロップ（特徴情報付き）"""
        crops = []
        
        for video_idx, (frame, features) in enumerate(frames):
            h, w = frame.shape[:2]
            
            # 各動画から1-2個のクロップ（軽量化）
            num_crops = random.randint(1, 2)
            
            for _ in range(num_crops):
                # ランダムなサイズ（元画像の20%〜70%）
                crop_w = random.randint(int(w * 0.2), int(w * 0.7))
                crop_h = random.randint(int(h * 0.2), int(h * 0.7))
                
                # ランダムな位置
                if crop_w < w and crop_h < h:
                    x = random.randint(0, w - crop_w)
                    y = random.randint(0, h - crop_h)
                    
                    # クロップ設定を保存（動画再生用）
                    crop_config = {
                        'video_idx': video_idx,
                        'crop_x': x,
                        'crop_y': y,
                        'crop_w': crop_w,
                        'crop_h': crop_h,
                        'features': self._get_features_in_region(features, x, y, crop_w, crop_h)
                    }
                    
                    crops.append(crop_config)
        
        return crops
    
    def _get_features_in_region(self, features: Dict, x: int, y: int, w: int, h: int) -> Dict:
        """指定領域内の特徴を取得"""
        region_features = {'faces': [], 'motion_regions': []}
        
        # 顔検出
        for face in features['faces']:
            fx, fy, fw, fh = face
            # 領域内に顔が含まれているか
            if (fx >= x and fy >= y and fx + fw <= x + w and fy + fh <= y + h):
                # 相対座標に変換
                region_features['faces'].append((fx - x, fy - y, fw, fh))
        
        # 動き検出
        for motion in features['motion_regions']:
            mx, my, mw, mh = motion['rect']
            # 領域内に動きが含まれているか
            if (mx >= x and my >= y and mx + mw <= x + w and my + mh <= y + h):
                region_features['motion_regions'].append({
                    'rect': (mx - x, my - y, mw, mh),
                    'area': motion['area'],
                    'type': motion['type']
                })
        
        return region_features
    
    def create_organic_composition(self, frames: List[np.ndarray], show_analysis: bool, 
                                 width: int = 1920, height: int = 1080) -> np.ndarray:
        """有機的なコンポジションを作成（動画再生＋解析オーバーレイ付き）"""
        # 背景（選択された色）
        bg_color = self.background_colors[self.current_bg_color]
        canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # 黒背景の場合のみ微妙なグラデーションを追加
        if self.current_bg_color == 'black':
            for y in range(height):
                gradient = int(20 * (y / height))  # より見やすく調整
                canvas[y, :] = np.clip(canvas[y, :] + gradient, 0, 255)
        
        # 既存のレイアウト領域を使用（なければ生成）
        if not self.layout_regions:
            self.layout_regions = self.generate_organic_layout(width, height, num_regions=6)
            # 初回は固定サイズも初期化
            self.fixed_sizes = []
            for region in self.layout_regions:
                self.fixed_sizes.append({
                    'width': region['width'],
                    'height': region['height']
                })
            
            # エフェクト設定を初期化（必ず1つはnoneを含む）
            effect_types = ['none', 'highlight', 'edge', 'blur', 'contrast', 'posterize', 
                           'sketch', 'pencil', 'mask_circle', 'mask_ellipse', 'mask_vignette']
            self.frame_effects = ['none']  # 最初は必ずnone
            
            # 残りの枠にランダムにエフェクトを割り当て
            for i in range(1, len(self.layout_regions)):
                self.frame_effects.append(random.choice(effect_types))
            
        regions = self.layout_regions
        
        # 解析が有効な場合は、現在のフレームで特徴を検出
        current_features = []
        if show_analysis:
            for idx, frame in enumerate(frames):
                features = self.detect_features(frame, video_idx=idx)
                current_features.append(features)
        
        # 配置
        for i, region in enumerate(regions):
            if i >= len(self.crop_configs):
                break
            
            crop_config = self.crop_configs[i]
            video_idx = crop_config['video_idx']
            
            if video_idx >= len(frames):
                continue
                
            frame = frames[video_idx]
            
            # クロップ領域を取得
            x = crop_config['crop_x']
            y = crop_config['crop_y']
            w = crop_config['crop_w']
            h = crop_config['crop_h']
            
            # フレームの範囲内に収める
            frame_h, frame_w = frame.shape[:2]
            x = min(x, frame_w - w)
            y = min(y, frame_h - h)
            
            if x >= 0 and y >= 0 and x + w <= frame_w and y + h <= frame_h:
                crop_img = frame[y:y+h, x:x+w]
                
                # 固定されたターゲットサイズを使用
                if i < len(self.fixed_sizes):
                    target_w = self.fixed_sizes[i]['width']
                    target_h = self.fixed_sizes[i]['height']
                else:
                    # フォールバック
                    target_w = region['width']
                    target_h = region['height']
                
                if crop_img.shape[1] > 0 and crop_img.shape[0] > 0:
                    # アスペクト比を保持してリサイズ
                    src_h, src_w = crop_img.shape[:2]
                    src_aspect = src_w / src_h
                    dst_aspect = target_w / target_h
                    
                    if src_aspect > dst_aspect:
                        # 元画像の方が横長
                        new_w = target_w
                        new_h = int(target_w / src_aspect)
                    else:
                        # 元画像の方が縦長
                        new_h = target_h
                        new_w = int(target_h * src_aspect)
                    
                    # リサイズ
                    resized = cv2.resize(crop_img, (new_w, new_h))
                    
                    # 枠に合わせて中央に配置するための黒い背景を作成
                    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    y_offset = (target_h - new_h) // 2
                    x_offset = (target_w - new_w) // 2
                    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    resized = padded
                    
                    # エフェクトを適用
                    if i < len(self.frame_effects):
                        resized = self.apply_frame_effect(resized, self.frame_effects[i])
                    
                    # 解析オーバーレイを適用
                    if show_analysis and current_features:
                        # 現在のフレームからクロップ領域の特徴を再計算
                        current_crop_features = self._get_features_in_region(
                            current_features[video_idx], 
                            x, y, w, h
                        )
                        
                        # 特徴もリサイズに合わせてスケーリング
                        scale_x = target_w / crop_img.shape[1]
                        scale_y = target_h / crop_img.shape[0]
                        scaled_features = self._scale_features(current_crop_features, scale_x, scale_y)
                        
                        if scaled_features['faces'] or scaled_features['motion_regions']:
                            resized = self.draw_analysis_overlay(resized, scaled_features)
                    
                    # 追加のぼかしは削除（エフェクトとして統合済み）
                    
                    # ソフトシャドウ
                    shadow_offset = 4
                    shadow_x = region['x'] + shadow_offset
                    shadow_y = region['y'] + shadow_offset
                    
                    # 影を描画（境界チェック付き）
                    if (0 <= shadow_x < width - target_w and 
                        0 <= shadow_y < height - target_h):
                        shadow_area = canvas[shadow_y:shadow_y+target_h,
                                           shadow_x:shadow_x+target_w]
                        shadow_area[:] = (shadow_area * 0.85).astype(np.uint8)
                    
                    # 画像を配置（境界チェック）
                    rx, ry = region['x'], region['y']
                    
                    # 画面内に収まる部分を計算
                    x1 = max(0, rx)
                    y1 = max(0, ry)
                    x2 = min(width, rx + target_w)
                    y2 = min(height, ry + target_h)
                    
                    if x2 > x1 and y2 > y1:
                        # 画像の対応部分
                        img_x1 = x1 - rx
                        img_y1 = y1 - ry
                        img_x2 = img_x1 + (x2 - x1)
                        img_y2 = img_y1 + (y2 - y1)
                        
                        canvas[y1:y2, x1:x2] = resized[img_y1:img_y2, img_x1:img_x2]
        
        return canvas
    
    def _scale_features(self, features: Dict, scale_x: float, scale_y: float) -> Dict:
        """特徴をスケーリング"""
        scaled_features = {'faces': [], 'motion_regions': []}
        
        # 顔をスケーリング
        for face in features['faces']:
            x, y, w, h = face
            scaled_features['faces'].append((
                int(x * scale_x), int(y * scale_y),
                int(w * scale_x), int(h * scale_y)
            ))
        
        # 動きをスケーリング
        for motion in features['motion_regions']:
            x, y, w, h = motion['rect']
            scaled_features['motion_regions'].append({
                'rect': (int(x * scale_x), int(y * scale_y),
                        int(w * scale_x), int(h * scale_y)),
                'area': motion['area'] * scale_x * scale_y,
                'type': motion['type']
            })
        
        return scaled_features
    
    def run(self):
        """メインループ"""
        print("\n=== BPM Organic Layout VJ with Analysis ===")
        print(f"Initial BPM: {self.bpm}")
        print("\nControls:")
        print("  q: Quit")
        print("  t: Tap BPM (tap multiple times)")
        print("  +/-: BPM ±10")
        print("  [/]: BPM ±1")
        print("  1-4: Beat multiplier (1=every beat, 4=every bar)")
        print("  Space: Force update (layout + effects)")
        print("  a: Toggle analysis overlay (OFF for better performance)")
        print("  m/n: Motion detection threshold -/+")
        print("  s/d: Playback speed -/+ (default: 2x)")
        print("  f/g: Face detection size -/+")
        print("  b: Cycle background color (Black/RGB/CMY)")
        print("\n")
        
        show_analysis = False  # デフォルトでOFFにしてパフォーマンス向上
        
        while True:
            self.frame_count += 1
            
            # 各動画から現在のフレームを取得（速度調整付き）
            current_frames = []
            for i, cap in enumerate(self.video_caps):
                # 再生速度に応じてフレームをスキップ
                skip_frames = int(self.playback_speed)
                for _ in range(skip_frames):
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_positions[i] = 0
                        ret, frame = cap.read()
                        break
                if ret:
                    current_frames.append(frame)
                    self.video_positions[i] = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if not current_frames:
                break
            
            # BPMに基づいてレイアウト更新
            current_time = time.time()
            update_interval = self.beat_interval * self.beat_multiplier
            
            if (self.crop_configs == [] or 
                current_time - self.last_update >= update_interval):
                
                # 特徴検出（レイアウト更新時のみ）
                frames_with_features = []
                for idx, frame in enumerate(current_frames):
                    features = self.detect_features(frame, video_idx=idx) if show_analysis else {'faces': [], 'motion_regions': []}
                    frames_with_features.append((frame, features))
                
                # クロップ設定を生成
                self.crop_configs = self.extract_random_crops(frames_with_features)
                
                # レイアウト領域も更新（BPMに合わせて切り替え）
                self.layout_regions = self.generate_organic_layout(1920, 1080, num_regions=6)
                
                # 固定サイズもリセット
                self.fixed_sizes = []
                for region in self.layout_regions:
                    self.fixed_sizes.append({
                        'width': region['width'],
                        'height': region['height']
                    })
                
                # エフェクトもリセット（必ず1つはnoneを含む）
                effect_types = ['none', 'highlight', 'edge', 'blur', 'contrast', 'posterize', 
                               'sketch', 'pencil', 'mask_circle', 'mask_ellipse', 'mask_vignette']
                self.frame_effects = ['none']  # 最初は必ずnone
                
                # 残りの枠にランダムにエフェクトを割り当て
                for i in range(1, len(self.layout_regions)):
                    self.frame_effects.append(random.choice(effect_types))
                
                self.last_update = current_time
            
            # 現在のフレームでコンポジションを作成（動画再生）
            display = self.create_organic_composition(current_frames, show_analysis)
            
            # 情報表示
            display_h, display_w = display.shape[:2]
            info_text = f"BPM: {self.bpm} | Update: {self.beat_multiplier} beat(s) | Speed: {self.playback_speed}x | Analysis: {'ON' if show_analysis else 'OFF'} | BG: {self.current_bg_color.upper()}"
            # 背景色に応じてテキスト色を調整
            text_color = (200, 200, 200) if self.current_bg_color == 'black' else (50, 50, 50)
            cv2.putText(display, info_text, (10, display_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
            # ビートインジケーター（小さな点）
            beat_progress = ((current_time - self.last_update) / update_interval)
            indicator_size = int(5 + 10 * (1 - beat_progress))
            cv2.circle(display, (display_w - 30, display_h - 30), 
                      indicator_size, (100, 100, 100), -1)
            
            # 表示
            cv2.imshow('BPM Organic VJ with Analysis', cv2.resize(display, (1280, 720)))
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.tap_bpm()
            elif key == ord('+') or key == ord('='):
                self.set_bpm(self.bpm + 10)
            elif key == ord('-'):
                self.set_bpm(self.bpm - 10)
            elif key == ord('['):
                self.set_bpm(self.bpm - 1)
            elif key == ord(']'):
                self.set_bpm(self.bpm + 1)
            elif key >= ord('1') and key <= ord('4'):
                self.beat_multiplier = int(chr(key))
                print(f"Beat multiplier: {self.beat_multiplier}")
            elif key == ord(' '):
                self.last_update = 0  # 強制更新
                # エフェクトも再生成
                effect_types = ['none', 'highlight', 'edge', 'blur', 'contrast', 'posterize', 
                               'sketch', 'pencil', 'mask_circle', 'mask_ellipse', 'mask_vignette']
                self.frame_effects = ['none']  # 最初は必ずnone
                for i in range(1, len(self.layout_regions)):
                    self.frame_effects.append(random.choice(effect_types))
            elif key == ord('a'):
                show_analysis = not show_analysis
                print(f"Analysis overlay: {'ON' if show_analysis else 'OFF'}")
            elif key == ord('m'):
                # 動き検出の閾値を下げる（より敏感に）
                self.motion_area_threshold = max(20, self.motion_area_threshold - 20)
                print(f"Motion detection threshold: {self.motion_area_threshold}")
            elif key == ord('n'):
                # 動き検出の閾値を上げる（より鈍感に）
                self.motion_area_threshold = min(2000, self.motion_area_threshold + 50)
                print(f"Motion detection threshold: {self.motion_area_threshold}")
            elif key == ord('f'):
                # フォントサイズを小さくする
                self.font_size = max(20, self.font_size - 4)
                self.small_font_size = max(14, self.small_font_size - 4)
                print(f"Font size: {self.font_size}/{self.small_font_size}")
            elif key == ord('g'):
                # フォントサイズを大きくする
                self.font_size = min(60, self.font_size + 4)
                self.small_font_size = min(48, self.small_font_size + 4)
                print(f"Font size: {self.font_size}/{self.small_font_size}")
            elif key == ord('s'):
                # 再生速度を下げる
                self.playback_speed = max(0.5, self.playback_speed - 0.5)
                print(f"Playback speed: {self.playback_speed}x")
            elif key == ord('d'):
                # 再生速度を上げる
                self.playback_speed = min(4.0, self.playback_speed + 0.5)
                print(f"Playback speed: {self.playback_speed}x")
            elif key == ord('b'):
                # 背景色を切り替える
                color_list = list(self.background_colors.keys())
                current_index = color_list.index(self.current_bg_color)
                next_index = (current_index + 1) % len(color_list)
                self.current_bg_color = color_list[next_index]
                print(f"Background color: {self.current_bg_color.upper()}")
        
        # クリーンアップ
        for cap in self.video_caps:
            cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BPM Organic Layout VJ with Analysis')
    parser.add_argument('--bpm', type=int, default=120, 
                       help='Initial BPM (60-200, default: 120)')
    parser.add_argument('--videos', type=int, default=4,
                       help='Number of videos to use (3-5, default: 4)')
    args = parser.parse_args()
    
    # 動画を検索（現在のフォルダとサブフォルダ内の全MP4ファイル）
    all_videos = glob.glob("**/*.mp4", recursive=True)
    
    if len(all_videos) < 3:
        print(f"Error: Need at least 3 videos, found {len(all_videos)}")
        return
    
    # 指定数の動画をランダム選択（全動画を使用）
    num_videos = min(args.videos, len(all_videos))
    selected = all_videos[:num_videos] if num_videos == len(all_videos) else random.sample(all_videos, num_videos)
    
    print(f"Loading {num_videos} videos...")
    
    # VJ起動
    vj = FinalBpmOrganicVJWithAnalysis(bpm=args.bpm)
    if vj.setup_videos(selected, num_videos):
        vj.run()
    else:
        print("Error: Failed to load videos")

if __name__ == "__main__":
    main()