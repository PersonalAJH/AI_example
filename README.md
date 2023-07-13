# AI_example
Various applications using AI


에이전트 입장에서 루프
1. 현재상황 st 에서 어떤 액션을 해야할지 at를 결정
2. 결정된 행동 at 를 환경으로 보냄
3. 환경으로부터 그에 따른 보상과 다음상태의 정보를 받음

Markov Process
-미리 정의된 어떠한 확률분포를 따라 상태와 상태를 전이해 가는 과정
-상태집합 S(에이전트가 될 수 있는 상황들) 와 전이 확률 행렬 P(상태에서 다른상태로 이동하는 확률분포표, 하나의 상태에서 뻗어나가는 모든 확률의 합은 1이다)가 필요하다.
-Markov Property : 이전의 상태는 현재의 상태의 확률분포에 영향을 줄 수 없으며 오직 현재상태에 의해서만 전이확률분포가 결정된다.

마르코프한 예시 : 바둑, 오목, 체스와 같이 현재상태가 이전상태에 의해 영향을 받지 않는 경우


Markov Reward Process
-Markov Process 에서 Reward 와 gamma(감쇠비)가 추가된다
-MRP = (S,P,R,r)(S:상태, P:상태전이확률, R:보상, r: 감쇠비)
-감쇠비의 필요성 : 수학적 편리성, 사람의 선호 반영, 미래에 대한 불확실성 반영

*Monte-Carlo 접근법 : 큰수의 법칙-> 여러번 샘플링을 거친후에 그의 확률을 확인


Markov Decision Process
-MDP는 MRP에서 에이전트가 더해진 것이다
-MDP = (S,A,P,R,r)
-MDP의 전이확률분포(P)는 MRP와는 약간다르다. MDP에서 P는 t시점, s상태에서 에이전트가 a를 하였을 때 s'에 도달할 확률을 의미한다(바둑에서 내가 A를 둔다고 상대방이 항상 같은것을 두는것은 아니기 때문에), Reward(R) 또한 어떤 액션을 했을 때 얻는 보상을 의미한다


정책함수(policy function/pi)
-MDP속 상태에서 s0에서 선택할 수 있는 액션에 대해서 얼만큼의 확률을 부여할 지 정책함수(pi)가 결정함
-정책함수는 에이전트에 속에 있으며 더 큰보상을 얻기 위해 정책을 교정해 나가는 것이 강화학습->CNN의 feature function


=================================================================================================================================================
Chapter 3

벨만 방정식
- 재귀함수 : 현재 시점(t) 와 다음 시점(t+1) 사이의 재귀적 관계를 이용해 정의한다. 

S(t) 에서 S(t+1) 이 되기 위해서는 두가지의 확률적 변경이 존재함
1. 정책이 액션을 선택할때
2. 액션이 선택되고 전이 확률에 의해 그 상태가 되는 확률

벨만 방정식 0단계, 2단계(강화학습 ppt 참고)는 현재 상태와 다음 상태를 밸류 사이의 연결고리를 나타내며 2단계는 0단계의 기댓값 연산자를 모두 확률과 밸류의 곱으로 나타낸 것을 의미한다. 2단계 식을 계산하기 위해서는 보상함수(각 상태에서 액션을 선택하면 얻는 보상), 전이 확률(각 상태에서 액션을 선택하면 다음 상태가 어디가 될지에 관한 확률 분포) 를 반드시 알아야 한다. 

MDP 를 모르는 상황(확률 분포등)에서 학습하는 접근법을 Model-free 접근법 이라고 하며 반대로 보상과 확률분포를 안다면 실제로 경험해 보지 않고 머릿속에서 시뮬레이션 해보는 것 만으로도 강화학습을 할 수 있다 이런 종류의 접근법을 Model_based or Planning 이라고 한다.

MDP의 정보를 모르는 상황에서는 0단계 식을, 아는 상황에서는 2단계 식을 사용하여 강화학습을 한다. 


==================================================================================================================================================Chpater 4
MDP를 알 떄의 플래닝

반복적 정책 평가
-테이블의 값들을 초기화한 후 밸ㅁ나 기대 방정식을 반복적으로 사용 하여 테이블에 적어 놓은 값을 족므씩 업데이트 해가는 방법론, 전이확률과 보상함수를 모두 알 떄

최고의 정책 찾기
-Greedy policy(먼 미래까지는 보지 않고 눈앞의 이익이 최대화하는 선택을 취하는 방식)

과연 정책이 개선되는가? 
-Pi_greedy는 Pi보다 좋은 정책이며 t=0 일때 부터 성립하고 t, t+1일 때 모두 성립하므로 귀납적으로 ㅈ거용하였을 때 그리디 정책이 원래 정책 pi보다 좋음을 확인할 수 있다. 


==================================================================================================================================================Chpater 5
MDP를 모를 때 밸류 평가하기 

MDP를 모르는 경우 - 정확히는 보상함수와 전이함수를 모르는경우 ->이런경우는 "모델프리" 불린다.

모델 - 강화학습에서 환경의 모델의 줄임말로 에이전트의 액션에 대해 환경이 어떻게 응답할지 예측하기 위해 사용하는 모든 것을 가리킵니다. 에이전트의 액션에 대하여 환경이 어떻게 반응할 수 있다면 에이전트의 입장에서는 여러가지의 계획을 세울수 있다.

test test
test