## 文章中の潜在要素を考慮した対話システム（すべてのファイル）

## 説明
（「」内は論文内のモジュールの名前と対応）
 * chatbot：対話システム本体の内容物のフォルダ
 * emotion_classifier：対話システムの「感情検出」モデルの訓練時に使うフォルダ（再訓練する必要は今のところなし）
 * intent_calculator：対話システムの「会話文分類」モデルの訓練時に使うフォルダ（データが増える度に再訓練する必要あり）

### Chatbot/chatbotの内容物
 * chatbot/models：使用する機械学習モデルが入っているフォルダー（ex. lstm.pyはLSTMのモデルが含まれているフォルダー）
 * chatbot/output_module：「潜在空間マッパー」の学習ファイル
 * chatbot/params：「自己組織化モジュール」と「潜在空間マッパー」の学習ファイル
 * chatbot/sentence_dimentionalizer：emotion_classifierとintent_calculatorの学習ファイルが保存されているフォルダ
（
 * chatbot/emotion_classifier：「感情検出」モデルの学習ファイル
 * chatbot/intent_calculator：「会話文分類」モデルの学習ファイル
）
 * chatbot/chatbot.py：対話システムの本体のコード。すべてのモデルのロード、自己組織化モジュールの学習、出力の計算などすべてここで行われる。
 * chatbot/constants.py：使用する”定数”のためのファイル
 * chatbot/functions.py：「単語→ベクトル」や「ベクトル→単語」などといったGloVeの処理（チャットボットには依存しない）処理を行う関数を集めたファイル
 * chatbot/test.py：対話システムと対話を行うファイル。
 * chatbot/train_lstm.py：「出力文の潜在空間表示」モジュールの訓練の為に使うファイル。
 * chatbot/train_nn.py：「潜在空間マッパー」モジュールの訓練の為に使うファイル。

## 実行
 学習完了時点で、chatbot/test.pyを実行。
（現段階でアップロードされている学習ファイルは、簡易的なデータセットを用いた学習ファイルのため、不完全である。）
