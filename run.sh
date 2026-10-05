#!/bin/bash
# 선행연구 분석기 실행 스크립트

cd "$(dirname "$0")"

# API 키 확인
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "⚠️  ANTHROPIC_API_KEY가 설정되지 않았습니다."
  echo "   다음 명령어로 설정 후 다시 실행하세요:"
  echo ""
  echo "   export ANTHROPIC_API_KEY='sk-ant-...'"
  echo "   ./run.sh"
  echo ""
  exit 1
fi

# 패키지 설치 (없으면)
echo "📦 패키지 확인 중..."
pip3 install -q -r requirements.txt

# 실행
echo "🚀 서버 시작 중..."
python3 app.py
