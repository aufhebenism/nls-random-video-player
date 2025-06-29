# BPM Organic VJ with Motion Analysis

BPM同期型のオーガニックレイアウトVJシステムです。動画ファイルを音楽のBPMに合わせて自動的に切り替え、動作解析によってエフェクトを動的に適用します。

## 特徴

- **BPM同期**: 音楽のテンポに合わせて映像レイアウトを自動切り替え
- **オーガニックレイアウト**: ランダムなグリッドレイアウトで映像を配置
- **動作解析**: 顔認識と動き検出により適切なエフェクトを自動選択
- **豊富なエフェクト**: ブラー、エッジ検出、グリッチ、カレイドスコープなど
- **リアルタイムコントロール**: キーボードでBPMやエフェクトを調整可能

## 動作環境

- Python 3.11 推奨
- macOS, Windows, Linux

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/aufhebenism/bpm-organic-vj.git
cd bpm-organic-vj

# Python 3.11で仮想環境を作成（推奨）
python3.11 -m venv .venv311
source .venv311/bin/activate  # macOS/Linux
# または
.venv311\Scripts\activate  # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

## 使い方

### 基本的な使い方

```bash
python finalBpmOrganicVJ_with_analysis.py /path/to/video/folder
```

### 設定ファイルを使用

```bash
python finalBpmOrganicVJ_with_analysis.py /path/to/video/folder --config config.json
```

## キーボードコントロール

- `q`: 終了
- `+/-`: BPMを増減
- `t`: TAP BPM（タップしてBPMを設定）
- `1-9`: ビートマルチプライヤーを変更
- `r`: レイアウトをリセット
- `f`: フルスクリーン切り替え
- `[/]`: 再生速度を調整
- `m`: 動作解析のON/OFF
- `p/o`: 動き検出の閾値を調整

### エフェクトコントロール
- `g`: ガウシアンブラー
- `b`: モーションブラー
- `e`: エッジ検出
- `x`: ピクセレート
- `z`: グリッチ
- `w`: ウェーブ
- `k`: カレイドスコープ
- `c`: カラーシフト
- `v`: ポスタライズ
- `s`: スケッチ
- `d`: 鉛筆画
- `n`: マスク

## 設定

`config.json`で詳細な設定が可能です：

```json
{
  "window": {
    "width": 800,
    "height": 600,
    "fullscreen": false
  },
  "video": {
    "folder_path": "/path/to/video/folder",
    "extensions": [".mp4", ".avi", ".mov", ".mkv"]
  },
  "bpm": {
    "default": 120,
    "multiplier": 4
  },
  // ... その他の設定
}
```

### 主な設定項目

- **window**: ウィンドウサイズとフルスクリーン設定
- **video**: 動画フォルダと対応拡張子
- **bpm**: デフォルトBPMとビートマルチプライヤー
- **layout**: グリッドレイアウトの設定
- **effects**: 各エフェクトの有効/無効と詳細設定
- **motion_analysis**: 動作解析の設定とキーワードマッピング
- **face_detection**: 顔認識の設定（アニメ顔対応）
- **playback**: 再生速度の範囲
- **osc**: OSC通信設定（外部コントロール用）

## 動作解析について

システムは以下の解析を行い、適切なエフェクトを選択します：

1. **顔認識**: 人物やキャラクターの顔を検出
2. **動き検出**: フレーム間の動きの大きさを解析
3. **キーワードマッピング**: 検出結果に基づいて適切なエフェクトを選択

### エフェクトマッピング例

- **aggressive** (激しい動き): グリッチ、エッジ検出、ポスタライズ
- **smooth** (なめらかな動き): ガウシアンブラー、ウェーブ、カラーシフト
- **dynamic** (動的): モーションブラー、カレイドスコープ、ピクセレート
- **static** (静的): スケッチ、鉛筆画、マスク

## アニメ顔認識について

アニメキャラクターの顔認識を使用する場合は、`lbpcascade_animeface.xml`をプロジェクトディレクトリに配置し、config.jsonで設定してください：

```json
"face_detection": {
  "enabled": true,
  "cascade_path": "lbpcascade_animeface.xml",
  // ...
}
```

## トラブルシューティング

### 動画が読み込まれない
- 動画フォルダのパスが正しいか確認
- 対応している動画形式か確認（mp4, avi, mov, mkv）

### パフォーマンスが悪い
- ウィンドウサイズを小さくする
- 動作解析を無効にする（`m`キー）
- エフェクトを減らす

### エフェクトが適用されない
- 動作解析が有効になっているか確認
- 動き検出の閾値を調整（`p/o`キー）

## ライセンス

MIT License

## 作者

aufhebenism

## 貢献

プルリクエストを歓迎します。大きな変更を行う場合は、まずissueで議論してください。