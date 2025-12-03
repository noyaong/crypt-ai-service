"""CLI 명령어"""

import argparse
import sys

import torch


def check_system():
    """시스템 상태 확인"""
    print("🔍 시스템 체크")
    print("=" * 50)

    # Python
    print(f"Python: {sys.version}")

    # PyTorch
    print(f"PyTorch: {torch.__version__}")

    # MPS
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    print(f"MPS Available: {mps_available}")
    print(f"MPS Built: {mps_built}")

    if mps_available and mps_built:
        print("✅ Apple Silicon GPU 사용 가능!")

        # 간단한 벤치마크
        import time

        device = torch.device("mps")
        size = 2048
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warm-up
        _ = torch.matmul(a, b)
        torch.mps.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            _ = torch.matmul(a, b)
        torch.mps.synchronize()
        elapsed = time.time() - start

        print(f"\n📊 벤치마크 ({size}x{size} 행렬곱 x10)")
        print(f"   총 시간: {elapsed:.3f}s")
        print(f"   평균: {elapsed/10*1000:.2f}ms/연산")
    else:
        print("⚠️ MPS 사용 불가, CPU로 동작합니다.")


def get_price(symbols: list[str]):
    """시세 조회"""
    from crypto_ai import CryptoAIService

    try:
        service = CryptoAIService()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    for symbol in symbols:
        print(f"\n📊 {symbol.upper()}")
        print("-" * 40)
        result = service.get_price(symbol)
        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            print(f"  이름: {result['name']}")
            print(f"  가격: {result['price_usd']}")
            print(f"  1시간: {result['change_1h']}")
            print(f"  24시간: {result['change_24h']}")
            print(f"  7일: {result['change_7d']}")
            print(f"  거래량(24h): {result['volume_24h']}")
            print(f"  시가총액: {result['market_cap']}")


def get_market_insights():
    """시장 인사이트"""
    from crypto_ai import CryptoAIService

    try:
        service = CryptoAIService()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print("🌍 시장 인사이트 조회 중...")
    insights = service.get_market_insights(limit=50)

    print("\n📈 시장 개요")
    print("-" * 40)
    overview = insights["market_overview"]
    print(f"  총 시가총액: ${overview['total_market_cap']:,.0f}")
    print(f"  24h 거래량: ${overview['total_volume_24h']:,.0f}")
    print(f"  BTC 도미넌스: {overview['btc_dominance']:.1f}%")
    print(f"  ETH 도미넌스: {overview['eth_dominance']:.1f}%")

    print("\n🎭 센티먼트")
    print("-" * 40)
    sentiment = insights["sentiment_analysis"]
    print(f"  공포-탐욕 지수: {sentiment['fear_greed_index']:.1f}")
    print(f"  센티먼트: {sentiment['sentiment']}")
    print(f"  24h 평균 변동: {sentiment['avg_change_24h']:+.2f}%")

    print("\n🚀 Top 5 상승")
    print("-" * 40)
    for coin in insights["top_gainers"]:
        print(f"  {coin['symbol']}: {coin['change_24h']:+.1f}%")

    print("\n📉 Top 5 하락")
    print("-" * 40)
    for coin in insights["top_losers"]:
        print(f"  {coin['symbol']}: {coin['change_24h']:+.1f}%")


def get_coin_insight(symbol: str):
    """코인 인사이트"""
    from crypto_ai import CryptoAIService

    try:
        service = CryptoAIService()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"💡 {symbol.upper()} 인사이트")
    print("=" * 50)
    for insight in service.get_coin_insights(symbol):
        print(f"  {insight}")


def predict_price(symbol: str, model_type: str = "transformer", checkpoint_dir: str = "checkpoints", interval: str = "1h"):
    """학습된 모델로 가격 예측"""
    from pathlib import Path

    import numpy as np

    from crypto_ai.preprocessing import DataPipeline, DataConfig, FEATURE_COLUMNS, INPUT_SIZE
    from crypto_ai.analyzer import get_device

    symbol = symbol.upper()
    interval_label = {"1h": "1시간봉", "4h": "4시간봉", "1d": "일봉"}.get(interval, interval)
    print(f"🤖 {symbol} AI 예측 ({interval_label})")
    print("=" * 50)

    # 체크포인트 경로 결정 (코인별/타임프레임별 디렉토리)
    if model_type == "transformer":
        checkpoint_path = Path(checkpoint_dir) / "transformer" / symbol / interval / "best.pt"
    else:
        checkpoint_path = Path(checkpoint_dir) / "lstm" / symbol / interval / "best.pt"

    if not checkpoint_path.exists():
        print(f"❌ 체크포인트가 없습니다: {checkpoint_path}")
        print(f"   먼저 {symbol} ({interval}) 모델을 학습하세요:")
        if model_type == "transformer":
            print(f"   uv run python scripts/train_transformer.py --symbol {symbol} --interval {interval} --days 90 --epochs 20 --multi-task")
        else:
            print(f"   uv run python scripts/train.py --symbol {symbol} --interval {interval} --days 90 --epochs 20")
        sys.exit(1)

    print(f"📂 체크포인트: {checkpoint_path}")

    device = get_device()
    print(f"📱 Device: {device}")

    # 데이터 수집 (타임프레임별 적절한 설정)
    days_map = {"1h": 7, "4h": 30, "1d": 90}  # interval별 적절한 데이터 기간
    seq_len_map = {"1h": 60, "4h": 60, "1d": 30}  # interval별 시퀀스 길이
    days = days_map.get(interval, 7)
    seq_len = seq_len_map.get(interval, 60)
    print(f"\n📊 {symbol} 데이터 수집 중... ({interval_label}, {days}일, seq={seq_len})")
    config = DataConfig(symbol=symbol, interval=interval, days=days, sequence_length=seq_len)
    pipeline = DataPipeline(config)

    try:
        df = pipeline.fetch_data()
        df = pipeline.compute_features(df)
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        sys.exit(1)

    # 특성 준비 (13개 특성)
    features = pipeline.normalize_features(df, FEATURE_COLUMNS, fit=True)

    # 최근 60개 시퀀스
    seq_len = config.sequence_length
    if len(features) < seq_len:
        print(f"❌ 데이터가 부족합니다. 최소 {seq_len}개 필요, 현재 {len(features)}개")
        sys.exit(1)

    x = torch.tensor(features[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)

    # 모델 로드 및 예측
    print(f"🔮 모델 로드 중... ({model_type})")

    if model_type == "transformer":
        from crypto_ai.transformer import CryptoTransformer

        # 체크포인트에서 multi_task 여부 확인
        checkpoint = torch.load(checkpoint_path, map_location=device)
        multi_task = checkpoint.get("multi_task", False)

        model = CryptoTransformer(
            input_size=INPUT_SIZE,  # 13 features
            d_model=64,
            num_heads=4,
            num_layers=3,
            multi_task=multi_task,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # 예측
        with torch.no_grad():
            outputs = model(x, return_attention=False)
            probs = torch.softmax(outputs["direction"], dim=-1).cpu().numpy()[0]
            direction_idx = probs.argmax()

        direction_names = ["하락 📉", "횡보 ➡️", "상승 📈"]
        direction_colors = ["\033[91m", "\033[93m", "\033[92m"]  # 빨강, 노랑, 초록
        reset_color = "\033[0m"

        print(f"\n{'='*50}")
        print(f"🎯 예측 결과: {direction_colors[direction_idx]}{direction_names[direction_idx]}{reset_color}")
        print(f"   신뢰도: {probs[direction_idx]*100:.1f}%")
        print(f"{'='*50}")

        print(f"\n📊 확률 분포:")
        for i, (name, prob) in enumerate(zip(["하락", "횡보", "상승"], probs)):
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ◀" if i == direction_idx else ""
            print(f"   {name}: {bar} {prob*100:5.1f}%{marker}")

        if multi_task:
            volatility = outputs["volatility"].cpu().numpy()[0][0]
            volume = outputs["volume"].cpu().numpy()[0][0]
            print(f"\n📈 추가 예측:")
            print(f"   변동성: {'높음 ⚠️' if volatility > 0.5 else '보통' if volatility > -0.5 else '낮음'} ({volatility:+.2f})")
            print(f"   거래량: {'증가 🔺' if volume > 0.3 else '감소 🔻' if volume < -0.3 else '유지'} ({volume:+.2f})")

    else:
        from crypto_ai.analyzer import ChartAnalyzer

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ChartAnalyzer(input_size=INPUT_SIZE, hidden_size=64, num_layers=2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            probs = model(x).cpu().numpy()[0]
            direction_idx = probs.argmax()

        direction_names = ["하락 📉", "횡보 ➡️", "상승 📈"]
        print(f"\n🎯 예측 결과: {direction_names[direction_idx]}")
        print(f"   신뢰도: {probs[direction_idx]*100:.1f}%")

    # 현재 기술적 지표
    print(f"\n📉 현재 기술적 지표:")
    latest = df.iloc[-1]
    print(f"   RSI: {latest['rsi']:.1f} {'(과매수)' if latest['rsi'] > 70 else '(과매도)' if latest['rsi'] < 30 else ''}")
    print(f"   MACD: {latest['macd']:.4f} ({'양' if latest['macd'] > 0 else '음'})")
    print(f"   볼린저 위치: {latest['bb_position']*100:.1f}%")

    # 시장 심리 지표
    print(f"\n🎭 시장 심리:")
    fng = latest['fear_greed']
    fng_label = '극단적 공포' if fng < 25 else '공포' if fng < 45 else '중립' if fng < 55 else '탐욕' if fng < 75 else '극단적 탐욕'
    print(f"   Fear & Greed: {fng:.0f} ({fng_label})")
    print(f"   BTC 도미넌스: {latest['btc_dominance']:.1f}%")

    # 최근 가격 정보
    print(f"\n💰 최근 가격:")
    print(f"   현재가: ${latest['close']:,.2f}")
    # 24시간 전 대비 변동률 계산
    if len(df) >= 24:
        price_24h_ago = df.iloc[-24]['close']
        change_24h = (latest['close'] - price_24h_ago) / price_24h_ago * 100
    else:
        change_24h = latest['returns'] * 100
    print(f"   24h 변동: {change_24h:+.2f}%")

    print(f"\n⚠️  주의: AI 예측은 참고용이며, 투자 결정은 본인 책임입니다.")


def ai_insight(symbol: str, checkpoint_dir: str = "checkpoints"):
    """멀티 타임프레임 AI 인사이트 (LLM 분석)"""
    from dotenv import load_dotenv
    from crypto_ai.llm_insight import generate_ai_insight

    load_dotenv()  # 환경변수 로드 (CRYPTOPANIC_API_KEY 등)
    symbol = symbol.upper()
    print(f"🤖 {symbol} 멀티 타임프레임 AI 인사이트")
    print("=" * 60)

    print("📊 타임프레임별 예측 수집 중...")
    result = generate_ai_insight(symbol, checkpoint_dir)

    if not result["predictions"]:
        print(f"❌ {symbol} 학습된 모델이 없습니다.")
        print("   먼저 모델을 학습하세요:")
        print(f"   uv run python scripts/train_transformer.py --symbol {symbol} --interval 1h --days 90 --epochs 20 --multi-task")
        sys.exit(1)

    # 타임프레임별 예측 결과 출력
    timeframe_labels = {"1h": "단기 (1시간봉)", "4h": "중기 (4시간봉)", "1d": "장기 (일봉)"}
    direction_colors = {"하락": "\033[91m", "횡보": "\033[93m", "상승": "\033[92m"}
    reset_color = "\033[0m"

    print(f"\n📈 사용 가능한 타임프레임: {', '.join(result['available_timeframes'])}")
    print("-" * 60)

    for interval, pred in result["predictions"].items():
        label = timeframe_labels.get(interval, interval)
        direction = pred["direction"]
        color = direction_colors.get(direction, "")
        confidence = pred["confidence"] * 100

        print(f"\n【{label}】")
        print(f"   예측: {color}{direction}{reset_color} (신뢰도: {confidence:.1f}%)")
        print(f"   확률: 하락 {pred['probabilities']['하락']*100:.1f}% | 횡보 {pred['probabilities']['횡보']*100:.1f}% | 상승 {pred['probabilities']['상승']*100:.1f}%")
        print(f"   RSI: {pred['indicators']['rsi']:.1f} | MACD: {pred['indicators']['macd']:.2f} | BB: {pred['indicators']['bb_position']*100:.1f}%")

    # 시장 상황
    if result["predictions"]:
        first_pred = next(iter(result["predictions"].values()))
        fng = first_pred["market"]["fear_greed"]
        fng_label = "극단적 공포" if fng < 25 else "공포" if fng < 45 else "중립" if fng < 55 else "탐욕" if fng < 75 else "극단적 탐욕"

        print(f"\n🎭 시장 심리")
        print(f"   Fear & Greed: {fng:.0f} ({fng_label})")
        print(f"   BTC 도미넌스: {first_pred['market']['btc_dominance']:.1f}%")
        print(f"   현재가: ${first_pred['price']['current']:,.2f}")
        print(f"   24h 변동: {first_pred['price']['change_24h']:+.2f}%")

    # 센티멘트 데이터 출력
    if result.get("sentiment"):
        sent = result["sentiment"]
        print(f"\n📰 센티멘트 분석")

        # 뉴스 센티멘트
        if sent.get("news"):
            news = sent["news"]
            total_news = news.get("total_posts", 0)
            if total_news > 0:
                score = news.get("sentiment_score") or 0
                news_label = "긍정적" if score > 0.2 else "부정적" if score < -0.2 else "중립"
                print(f"   뉴스: {total_news}건 ({news_label})")
                # 최근 헤드라인 1개 표시
                headlines = news.get("recent_headlines", [])
                if headlines:
                    title = headlines[0].get("title", "")[:50]
                    print(f"   최근: {title}...")

        # 소셜 지표
        if sent.get("social"):
            social = sent["social"]
            social_label = "높음" if social["score"] > 70 else "보통" if social["score"] > 40 else "낮음"
            print(f"   소셜 활성도: {social_label} (점수: {social['score']:.1f}/100)")

        # 종합 센티멘트
        overall_emoji = "🟢" if sent["overall_sentiment"] == "bullish" else "🔴" if sent["overall_sentiment"] == "bearish" else "🟡"
        print(f"   종합: {overall_emoji} {sent['overall_sentiment'].upper()} (점수: {sent['overall_score']:+.3f})")

    # LLM 인사이트
    print("\n" + "=" * 60)
    print("💡 AI 종합 분석 (GPT-4o-mini)")
    print("=" * 60)
    print(result["insight"])

    print(f"\n⚠️  주의: AI 분석은 참고용이며, 투자 결정은 본인 책임입니다.")


def main():
    """메인 CLI 진입점"""
    parser = argparse.ArgumentParser(
        prog="crypto-ai",
        description="MacBook MPS + PyTorch 기반 암호화폐 AI 분석 서비스",
    )
    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # check
    subparsers.add_parser("check", help="시스템 상태 확인")

    # price
    price_parser = subparsers.add_parser("price", help="시세 조회")
    price_parser.add_argument("symbols", nargs="+", help="코인 심볼 (예: BTC ETH)")

    # market
    subparsers.add_parser("market", help="시장 인사이트")

    # insight
    insight_parser = subparsers.add_parser("insight", help="코인 인사이트")
    insight_parser.add_argument("symbol", help="코인 심볼 (예: AVAX)")

    # predict
    predict_parser = subparsers.add_parser("predict", help="AI 모델 예측")
    predict_parser.add_argument("symbol", help="코인 심볼 (예: BTC)")
    predict_parser.add_argument(
        "--model", "-m",
        choices=["transformer", "lstm"],
        default="transformer",
        help="모델 타입 (기본: transformer)"
    )
    predict_parser.add_argument(
        "--interval", "-i",
        choices=["1h", "4h", "1d"],
        default="1h",
        help="타임프레임 (1h: 1시간봉, 4h: 4시간봉, 1d: 일봉, 기본: 1h)"
    )
    predict_parser.add_argument(
        "--checkpoint-dir", "-c",
        default="checkpoints",
        help="체크포인트 디렉토리 (기본: checkpoints)"
    )

    # insight-ai (LLM 기반 멀티 타임프레임 분석)
    ai_parser = subparsers.add_parser("insight-ai", help="LLM 기반 멀티 타임프레임 AI 분석")
    ai_parser.add_argument("symbol", help="코인 심볼 (예: AVAX)")
    ai_parser.add_argument(
        "--checkpoint-dir", "-c",
        default="checkpoints",
        help="체크포인트 디렉토리 (기본: checkpoints)"
    )

    args = parser.parse_args()

    if args.command == "check":
        check_system()
    elif args.command == "price":
        get_price(args.symbols)
    elif args.command == "market":
        get_market_insights()
    elif args.command == "insight":
        get_coin_insight(args.symbol)
    elif args.command == "predict":
        predict_price(args.symbol, args.model, args.checkpoint_dir, args.interval)
    elif args.command == "insight-ai":
        ai_insight(args.symbol, args.checkpoint_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
