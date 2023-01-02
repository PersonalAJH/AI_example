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
Markov Reward Process
Markov Decision Process