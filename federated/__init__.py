"""
연합학습 모듈
- Server: 연합학습 서버 (가중치 집계)
- Client: 연합학습 클라이언트 (로컬 학습)
- Aggregator: 가중치 평균화 알고리즘
"""

from .server import FederatedServer
from .client import FederatedClient
from .aggregator import FederatedAveraging

__all__ = ['FederatedServer', 'FederatedClient', 'FederatedAveraging']

