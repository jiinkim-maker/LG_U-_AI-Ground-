# LG_U-_AI-Ground-
LG U+ 콘텐츠 추천 모델 제작

## About the project

- **LG U+ 콘텐츠 추천 모델 제작**
- **LG유플러스의 아이들나라 서비스 데이터를 활용하여, 고객이 다음에 시청할 콘텐츠를 추천하는 Task.**
    
    컨텐츠 기반(content-based) 방법과 더불어 추천시스템의 한가지 큰 줄기인 **협업 필터링(Collaborative Filtering)**은 특정 아이템에 대한 유저의 선호도를 평점과 방문기록 등의 과거 상호관계(interaction)에 기반하여 모델링.
    
- 과거의 모든 상호작용들을 바탕으로 새로운 데이터에 대한 개인화된 상호작용을 예측하여 주는 추천시스템
- 기존의 선형적인 관계(내적 계산)만을 고려하는 방식 외에 심층 신경망을 사용하여 데이터 사이의 상호관계를 직접 모델링

## Role

- Data Feature Engineering
- NCF 모델링 과정을 전담.

## Results
최종 : 658개의 팀 중 37위 기록.
![Untitled___](https://github.com/user-attachments/assets/9a199d77-fa47-4063-944e-f1a95dbaa6a7)


## Paper Review
# 1. **Backgrounds : Recommender system**

[기존의 협업 필터링]

협업 필터링에서 가장 대중적으로 사용되는 기술은 **Matrix Factorization**. 'Netflix Prize'를 통하여 유명해진 Matrix factorization은 **latent space**를 공유하는 **User vector**와 **Item vector**의 **내적**을 통해 **Interation**을 모델링하는 방법입니다.

MF는 또한 User, Item 간의 **Sparse**
한 Interaction Matrix를 채워주는 **Matrix completion.** 

MF(**Matrix Factorization**)의 한계점

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2306b31-878a-44d1-bd86-c4870de309f8/Untitled.png)

1. 상호작용을 모델링하기 위해 사용하는 내적(inner-product)은 latent feature를 선형적으로 곱하는 단순한 방법으로 매우 효율적이다. 그러나, 이는 내적이 **선형적인 관계만을 모델링**할 수 있음을 의미.
2. MF는 user와 item을 같은 latent space에 위치시키기 때문에, user 사이의 유사도 또한 내적으로 나타낼 수 있습니다. 또한 latent vector가 단위 벡터라는 가정 하에는 코사인 유사도 등의 방법도 가능해 집니다.
3. 더 높은 차원의 latent space를 도입(하나의 축 추가) → 더 복잡한 대소 관계(유사도 관계) 표현이 가능해짐 → 그러나, 모델의 일반화 성능(generalization)을 저해할 수 있는 방법

즉, 내적은 user와 item 사이의 복잡한 **비선형적 상호관계를 모델링 하기에는 불충분**할 수 있다.

*# latent space : Latent vector는 한 이미지가 가지고 있는 잠재적인 벡터 형태의 변수 → 데이터가 가지고 있는 잠재적인 변수, 특정 차원에서 원하는 정보들이 모여있는 공간*

[**심층신경망(Deep Neural Network)를 이용한** 비선형적 관계를 모델링]

- 협업필터링을 위해 사용되는 데이터 두 종류

**1) Explicit feedback**

Item에 대한 User의 모든 직접적인 평가를 포함하는 데이터로 구하기는 어렵지만 바로 사용해도 좋을만큼 데이터의 품질이 좋은 경우가 많음.

ex) 평점, 취향, 후기, …

**2) Implicit feedback**

직접적으로 Item과 User간의 선호 관계를 나타내지는 않지만, 간접적인 정보를 포함하고 있는 데이터이다. 다양한 방법을 통하여 구할 수 있지만(수집 용이) 의미 파악이 어려움.

ex) 방문기록, 검색기록, 구매여부, 장바구니, ....

- 본 논문은 **Implicit feedback을 활용해 신경망 학습.**
1. Implicit feedback은 관계 파악이 어렵지만, 많은 양의 데이터를 구할 수 있기 때문에 신경망 구조의 유효성을 검증하기에 알맞다고 할 수 있음.
2. Implicit feedback에 의해 본 논문의 User u와 item j간의 interaction을 binary로 표현. 이때 1값은 관계가 관찰됨을 의미할 뿐, 특별히 선호도를 나타내지는 않음.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b813a01e-3cdb-459a-9d56-e17673c3bdd1/Untitled.png)

즉, **NCF가 학습하는 문제**는 User와 item 사이의 interaction 유무를 예측하는 **binary classification**이 됩니다.

# 2. **Neural Collaborative Filtering**

![General Framework](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1b821eb-a794-4cc6-a3a8-a9a91800a8c7/Untitled.png)

General Framework

NCF는 4개의 layer로 구성. (layer들을 각각 분석해보자.)

- **1) Input Layer(sparse)**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3d5c6730-7102-4045-886c-2259ff2bbdc7/Untitled.png)
    
    - Input Layer는 각각 user, item을 나타내는 **feature vector**
    - 각각의 벡터는 **one-hot encoding** 표현 → 매우 **sparse한 상태**
    
- **2) Embedding Layer**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cc507aaa-5ee9-41c7-bfea-740bbeca20dc/Untitled.png)
    
    - **sparse(희소)**한 **input vector**를 **Dense**하게 바꿔주는 역할을 하는 layer. 일반적인 임베딩 방법과 동일하게 fully-connected layer가 사용
    - **Embedding Matrix(가중치 매트릭스)**를 통하여 M,N 차원의 user,item vector를 K차원의 latent space에 표현한 것으로도 생각해 볼 수 있음.
    
    #**가중치 matrix**와 One-hot-encoding된 **Matrix**와 **내적**된 것이 최종 **Embedding matrix**
    
- **3) Neural CF Layers**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b46e96e6-bbd3-43f2-84be-537a6e8dbd6d/Untitled.png)
    
    - 임베딩이 완료된 User, Item latent vector는 여러 층의 신경망을 거침. 이 다층 신경망 구조 → **Neural CF Layers**.
    - User latent vector와 Item latent vector를 **concatenate**한 벡터를 시작으로 각각의 층을 거치며 인공신경망을 통해 복잡한 **비선형의 데이터 관계**를 학습.
    
    # 3. Model Training
    
    - NCF는 binary 형태의 implicit data를 사용하여 학습하는 구조 → bernoulli distribution을 가정
    - 목적 함수로 사용하기 위한 Negative log likelihood의 형태
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9495ab7a-88b3-4963-af60-2f6935af3832/Untitled.png)
        
    - Optimizer = **Stochastic Gradient Descent(SGD)**
    
    Loss Function을 계산할 때, 전체 데이터(Batch) 대신 일부 데이터의 모음(Mini-Batch)를 사용하여 Loss Function을 계산
    
    → 계산 속도가 훨씬 빠르기 때문에 같은 시간에 더 많은 step을 갈 수 있으며, 여러 번 반복할 경우 Batch 처리한 결과로 수렴 + Local Minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성도 높음.
    
    # 4. **Fusion of GMF and MLP**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d61ef7be-3221-4393-ab94-a1bf7c2f2c07/Untitled.png)
    
- **4) Output Layer**
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/150d4ddd-c3a6-4a85-851e-ee2f663eee41/Untitled.png)
    
    - Output layer에서는 NCF layer의 hidden vector를 input으로 받아 predictive score y_hat을 예측→ 이를 target y와의 비교를 통해 학습 진행
    - P와 Q = embedding layer의 matrix
    - 0f = 모델 파라미터
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bddeee4a-a676-4b2c-ace7-7eeb238baa09/Untitled.png)
    
    model f는 multi-layer neural network 정의되어있으므로, 다음과 같이 차원 상으로 표현
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2c92721a-feec-41f1-b19a-04600ca7bad9/Untitled.png)
    

# 3. **Fusion of GMF and MLP**

본 논문에서는 NCF의 두 가지 예시인 GMF와 MLP를 융합한 모델 제안

→ **제안된 fusion 모델은 선형, 비선형의 Interaction을 모두 포착할 수 있는 모델** = **Neural Matrix Factorization**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/58143f3e-0274-40d8-b0e1-cf651989f8ac/Untitled.png)

최종 score = MLP와 GMF의 output을 concatenate.

ex) final_score = torch.cat((MLP, GMP), dim=1)
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5ddd6d34-b199-4137-8c5b-973235058979/Untitled.png)
