# federated-learning

3-2학기 지능형 IoT 수업 과제. Jetson 두 대(client1, client2)와 서버 간 연합학습(federated learning)을 구현했습니다.

## 구성

- `(baseline)client1.py`, `(baseline)client2.py`, `(baseline)server.py` — 기본 제공 베이스라인
- `client1.py`, `client2.py`, `server.py` — 베이스라인을 개선한 버전

각 클라이언트가 로컬 데이터로 학습한 모델을 서버에서 취합(aggregation)하는 구조입니다.
