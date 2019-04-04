## 文章中の潜在要素を考慮した対話システム
このプロジェクトは、従来の対話システムのような文章の文法的処理に付け加え、表面では検知できない潜在的な変数をも吟味した対話システムである．

## 動機
現在普及に及んでいる人との会話を目的とした対話システムの殆どは会話文の文法の構成を着目点として処理を行っている．しかし，人間同士で会話を行うとき，人は無意識に感情を働かせたり過去の会話や自分の知識に基づいて相手に対する返答を行う．今回は人が会話を行うときに使用すると推測される感情の変化やその人個人の経験などを会話の一つの潜在要素として使用したときの対話システムの話し方の変化について検証を行った．今回の対話システムはある程度の会話の継続性に意識をし、会話を続けることができた．


## ビルドステータス

![Build Status](https://travis-ci.org/akashnimare/foco.svg?branch=master)


## 環境
CPU: Intel® Core™ i7-4770K CPU @ 3.50GHz × 8

GPU: GeForce GTX 1050 Ti/PCIe/SSE2

メモリー: 15.6 GiB (16GiB以上を推奨)

## 依存ライブラリ
このコードは Python3においてKeras(TensorFlowバックエンド), Scipy, RakeとNumpyに依存する．

インストール方法(Ubuntu 16.04以上)：
```
pip install numpy scipy python-rake keras tensorflow
```

## Code style

![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)
 
## Screenshots
![](https://github.com/IronEdward/chatbot/blob/master/starting.png)
![](https://github.com/IronEdward/chatbot/blob/master/started.png)
![](https://github.com/IronEdward/chatbot/blob/master/waiting_input.png)


## 使用方法
```
python chatbot/test.py
```
## License
MIT © [Narutatsu Li]()
