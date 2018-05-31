# AAAIに向けたモデル

## モデルの概要
双方向翻訳モデルの同時学習
* RNNの隠れ状態を近づけることで翻訳制度を向上させる

## 提案手法
### sentence projection
cross lingual word embedingのprejectionを文単位で行う。  

|| hs - W ht ||  
|| U hs - ht ||  
W U = E  

W,Uの二つの行列を用いる方法をsoft、  
U = W^{-1}とする方法をhardとする  

soft,hardの他にprocrustes解析により上の制約を達成する方法が考えれらる。  
したがって、提案する手法は3つとなる。  

ここでhtはデコーダの時刻tにおける隠れ状態  
hsはアテンションにより得られるエンコーダの状態の加重平均とする。  
アテンションはhard, softをそれぞれ試す。  

### residual connection for attention with cross lingual word embedding
上の制約を学習する上で、アテンションの性質により類似している  
隠れ状態が文脈ベクトルとして抽出されやすい。  
そこで、単語embeddingをcross lingual word embeddingにより初期化、  
さらに、RNNの最上層へとresidual connectionする。  
これにより、アテンションにより対訳関係の単語が選ばれやすくできると考えられる。

## Todo
* 単語の埋め込み表現獲得(FastText)
* 双方向モデルの作成
* 制約の作成(soft, hard, procrustes)
