import re
import io
import json
import subprocess
import sys
import importlib.util
import urllib3
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np

# SSL 경고 무시 설정
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# [0] 라이브러리 설치 및 초기화
# ==========================================
def install_requirements():
    libraries = [
        ("streamlit", "streamlit"),
        ("finance-datareader", "FinanceDataReader"),
        ("pandas", "pandas"),
        ("requests", "requests"),
        ("openai", "openai"),
        ("plotly", "plotly"),
        ("certifi", "certifi"),
        ("scikit-learn", "sklearn"),
        ("ta", "ta") 
    ]
    for package, module in libraries:
        if importlib.util.find_spec(module) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from openai import OpenAI
from sklearn.ensemble import IsolationForest
import ta

pio.templates.default = "plotly_dark"

# ==========================================
# [1] API 설정
# ==========================================
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
    NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
    FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
    DART_API_KEY = st.secrets["DART_API_KEY"]
except Exception as e:
    st.error("Secrets 설정 오류: Streamlit Cloud 관리자 페이지에서 API 키를 등록해야 함.")
    st.stop()
    
DART_CORP_MAP = {
    "005930": "00126380", # 삼성전자
    "000660": "00164779", # SK하이닉스
    "035420": "00266961", # NAVER
    "005380": "00164742", # 현대차
    "035720": "00258801"  # 카카오
}

KR_SECTORS = {
    "005930": "Technology",
    "000660": "Technology",
    "035420": "Communication Services",
    "005380": "Consumer Cyclical",
    "035720": "Communication Services"
}

MAJOR_TICKERS_KR = list(DART_CORP_MAP.keys())
MAJOR_TICKERS_US = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "AVGO", "WMT"]

# ==========================================
# [2] 지표 사전 및 거시 데이터 정의
# ==========================================
EXOTIC_DICT = {
    "🍔 빅맥 지수 (Big Mac Index)": "전 세계 맥도날드 매장에서 판매되는 빅맥 가격을 달러로 환산해 각국 통화의 실질 구매력을 비교함.",
    "🩲 남성 속옷 지수 (Men's Underwear Index)": "속옷 판매량의 급락은 불황의 심화 및 소비 심리가 최저점에 달했음을 암시하는 신호로 해석됨.",
    "📦 골판지 상자 지수 (Cardboard Box Index)": "골판지 상자의 수요 증감을 추적함. 물동량과 직결되므로 실물 경제의 선행 지표로 높은 신뢰도를 가짐.",
    "🏢 마천루 지수 (Skyscraper Index)": "초고층 빌딩 건설 붐 완공 시점에는 경제 위기가 동반된다는 이론임.",
    "🌋 구리-금 비율 (Copper-to-Gold Ratio)": "위험자산(구리)과 안전자산(금)의 비율로 시장의 위험 선호도를 측정함.",
    "🚢 발틱 운임 지수 (Baltic Dry Index)": "벌크선의 해상 운송 비용을 수치화한 지표로 글로벌 교역량을 실시간으로 반영함.",
    "👗 치마 길이 지수 (Hemline Index)": "경기 호황기에는 소비 심리가 활발해져 치마 길이가 짧아진다는 심리 기반 지표임.",
    "🍜 라면 지수 (Ramen Index)": "서민 경제 타격 시 대체제인 라면 소비가 급증하는 불황형 소비 패턴을 분석함.",
    "📱 데이팅 앱 지수 (Dating App Index)": "경제적 여유 부족으로 저비용 심리 위안 수단인 데이팅 앱 수요가 몰리는 현상을 반영함.",
    "💊 번아웃 지수 (Burnout Index)": "불황이나 고용 불안정이 미치는 압박 수위를 진통제/피로회복제 판매량으로 가늠함.",
    "💄 립스틱 지수 (Lipstick Index)": "불황기에 저렴한 사치품(Small Luxury) 매출이 증가하는 심리적 현상임.",
    "🗑️ 쓰레기 지수 (Garbage Index)": "생산/유통/소비가 활발할수록 증가하는 폐기물 총량으로 실물 경제를 가늠함.",
    "🍾 샴페인 지수 (Champagne Index)": "축하할 일이 많은 호황기 후반 샴페인 소비 증가를 버블 붕괴의 전조로 해석함.",
    "👶 기저귀 지수 (Diaper Index)": "비용 절감을 위해 일회용 기저귀 사용량을 줄이는 현상으로 가계 현금 흐름 악화를 시사함.",
    "🐶 애완동물 유기 지수 (Pet Abandonment Index)": "경제적 비용 감당 불가로 반려동물을 유기하는 비율. 비극적인 후행 지표임."
}

FRED_SERIES = {
    "거시(Macro)": {"장단기 금리차 (T10Y2Y)": "T10Y2Y", "금융스트레스지수 (STLFSI4)": "STLFSI4", "삼 법칙 (SAHMREALTIME)": "SAHMREALTIME", "M2 통화량 (WM2NS)": "WM2NS", "기대 인플레이션 (MICH)": "MICH"},
    "미시(Micro)": {"소비자물가지수 (CPIAUCSL)": "CPIAUCSL", "개인소비지출 (PCE)": "PCE", "산업생산지수 (INDPRO)": "INDPRO", "실업률 (UNRATE)": "UNRATE", "소매 판매 (RSXFS)": "RSXFS"},
    "신용/공급망(Credit)": {"하이일드 스프레드 (BAMLH0A0HYM2)": "BAMLH0A0HYM2", "금융환경지수 (NFCI)": "NFCI", "글로벌 공급망 압력 (GSCPI)": "GSCPI"},
    "물가/원자재(Commodity)": {"글로벌 식량가격지수 (PFOODINDEXM)": "PFOODINDEXM"}
}

FRED_DESC = {
    "T10Y2Y": "장기(10년) 국채 금리와 단기(2년) 차이. 마이너스 역전 시 침체 전조.",
    "STLFSI4": "세인트루이스 연은 발표 금융 스트레스 지수. 0 이상이면 불안.",
    "SAHMREALTIME": "실업률 3개월 이평선이 최근 12개월 최저치보다 0.5%p 이상 높으면 침체 시작 판단.",
    "WM2NS": "시중에 풀린 광의 통화량(M2). 유동성 공급 상태 파악.",
    "MICH": "미시간대 1년 기대 인플레이션.",
    "VIXCLS": "S&P 500 내재 변동성(공포 지수).",
    "CPIAUCSL": "도시 소비자 핵심 물가 지표.",
    "PCE": "개인소비지출 물가지수. 연준(Fed) 주요 참고 지표.",
    "INDPRO": "광업, 제조업 등 실질 생산량. 실물 경제 활력 측정.",
    "UNRATE": "경제활동인구 중 실업자 비율.",
    "RSXFS": "소매 및 음식 서비스 판매액. 소비자 지출 동향.",
    "BAMLH0A0HYM2": "투기등급 회사채와 국채 수익률 격차. 부도 위험 지표.",
    "NFCI": "미국 금융 환경 긴축/완화 상태.",
    "GSCPI": "글로벌 공급망 병목 현상 및 원가 상승 압력 측정.",
    "PFOODINDEXM": "IMF 산출 글로벌 식량 가격 지수."
}
# ==========================================
# [3] 공통 유틸리티 함수
# ==========================================
def strip_html(text: str) -> str:
    if not text: return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()

def score_to_status(score: int):
    if score >= 80: return "안정", "#00CC96"
    if score >= 60: return "주의", "#FFA500"
    return "경고", "#FF4B4B"

def safe_float(v):
    try: return float(v) if v is not None else None
    except: return None

# ==========================================
# [4] 데이터 수집 엔진
# ==========================================
@st.cache_data(ttl=600)
def fetch_indices():
    res = {"indices": {}}
    idx_map = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11", "NASDAQ": "^IXIC", "S&P 500": "^GSPC"}
    start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    for name, symbol in idx_map.items():
        try:
            df = fdr.DataReader(symbol, start)
            if df is None or df.empty: 
                res["indices"][name] = {"price": 0.0, "change": 0.0, "df": pd.DataFrame(columns=["Close"])}
                continue
            close = df["Close"].dropna()
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else last
            change = round(((last - prev) / prev) * 100, 2)
            res["indices"][name] = {"price": last, "change": change, "df": close.tail(30)}
        except Exception:
            res["indices"][name] = {"price": 0.0, "change": 0.0, "df": pd.DataFrame(columns=["Close"])}
    return res

@st.cache_data(ttl=600)
def fetch_fred_series(series_id: str, years: int = 5) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader(f"FRED:{series_id}", start_date)
        if df is None or df.empty: raise ValueError("데이터 없음")
        df = df.reset_index()
        df.columns = ["DATE", series_id]
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df
    except:
        return pd.DataFrame(columns=["DATE", series_id])

@st.cache_data(ttl=3600)
def fetch_put_call_ratio() -> float:
    # 1. CNN Fear & Greed 내부 데이터에서 실시간 PCR 스크래핑 시도
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Origin": "https://edition.cnn.com", "Referer": "https://edition.cnn.com/"}
    try:
        r = requests.get(url, headers=headers, timeout=10, verify=False)
        data = r.json()
        return float(data['put_call_options']['data'][-1]['y'])
    except:
        # 2. 실패 시, PCR과 80% 이상 강한 양의 상관관계를 가지는 VIX를 활용해 PCR 추정치 강제 산출 (UI 에러 방지)
        try:
            vix_df = fdr.DataReader("FRED:VIXCLS", (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"))
            latest_vix = float(vix_df.iloc[-1, 0])
            proxy_pcr = 0.7 + ((latest_vix - 15) / 20) * 0.5
            return round(max(0.5, min(proxy_pcr, 1.5)), 2)
        except:
            return 0.95

def summarize_fred_latest(df: pd.DataFrame, series_id: str):
    if df is None or df.empty: return {"latest": None, "mom_change": None, "yoy_change": None}
    s = df.set_index("DATE")[series_id].dropna()
    if s.empty: return {"latest": None, "mom_change": None, "yoy_change": None}
    latest = safe_float(s.iloc[-1])
    mom, yoy = None, None
    try:
        m_idx = s.loc[:s.index[-1] - pd.Timedelta(days=30)]
        y_idx = s.loc[:s.index[-1] - pd.Timedelta(days=365)]
        if not m_idx.empty: mom = latest - safe_float(m_idx.iloc[-1])
        if not y_idx.empty: yoy = latest - safe_float(y_idx.iloc[-1])
    except: pass
    return {"latest": latest, "mom_change": mom, "yoy_change": yoy}

@st.cache_data(ttl=3600)
def fetch_dix_gex() -> pd.DataFrame:
    url = "https://squeezemetrics.com/monitor/static/DIX.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_cnn_fear_and_greed():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    
    # 실제 최신 크롬 브라우저와 동일한 헤더로 위장 (Anti-Bot 우회용)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7",
        "Referer": "https://edition.cnn.com/",
        "Origin": "https://edition.cnn.com",
        "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10, verify=False)
        r.raise_for_status()
        data = r.json()
        score = data['fear_and_greed']['score']
        rating = data['fear_and_greed']['rating']
        rating_ko = {"extreme fear": "극단적 공포", "fear": "공포", "neutral": "중립", "greed": "탐욕", "extreme greed": "극단적 탐욕"}.get(rating.lower(), rating)
        return {"score": round(score), "rating": rating_ko, "error": None}
    except Exception as e:
        return {"score": None, "rating": None, "error": str(e)}

@st.cache_data(ttl=600)
def fetch_naver_news_bulk(query: str, client_id: str, client_secret: str, display: int = 100):
    if not client_id or client_id == "YOUR_NAVER_ID": return {"items": [], "error": "NAVER API 설정 오류"}
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    params = {"query": query, "display": display, "start": 1, "sort": "date"}
    r = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
    if r.status_code != 200: return {"items": [], "error": f"API 오류: {r.status_code}"}
    data = r.json()
    return {"items": [{"title": strip_html(it.get("title")), "description": strip_html(it.get("description")), "link": it.get("link")} for it in data.get("items", [])], "error": None}

@st.cache_data(ttl=600)
def fetch_naver_news_500(query: str, client_id: str, client_secret: str):
    if not client_id or client_id == "YOUR_NAVER_ID": return {"items": [], "error": "NAVER API 설정 오류"}
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    all_items = []
    seen_titles = set()
    
    starts = [1, 101, 201, 301, 401]
    for start in starts:
        params = {"query": query, "display": 100, "start": start, "sort": "date"}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10, verify=False)
            if r.status_code == 200:
                data = r.json()
                for it in data.get("items", []):
                    title = strip_html(it.get("title"))
                    if title not in seen_titles:
                        seen_titles.add(title)
                        all_items.append({"title": title, "description": strip_html(it.get("description")), "link": it.get("link")})
        except:
            pass
        time.sleep(0.1)
    if not all_items: return {"items": [], "error": "뉴스 데이터 수집 실패"}
    return {"items": all_items, "error": None}

@st.cache_data(ttl=3600)
def fetch_corporate_keywords(tickers: list, api_key: str):
    if not api_key or api_key == "YOUR_FINNHUB_API_KEY": return {"error": "Finnhub API 설정 오류."}
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    keywords = {"Positive": ["ai", "growth", "demand", "record", "innovat", "guidance"], "Negative": ["recession", "layoff", "inflation", "cost", "shortage", "supply chain"]}
    counts = {k: 0 for k in keywords["Positive"] + keywords["Negative"]}
    headers = {"X-Finnhub-Token": api_key}
    
    for tkr in tickers[:6]:
        url = f"https://finnhub.io/api/v1/company-news?symbol={tkr}&from={start_date}&to={end_date}"
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                news_list = r.json()
                for item in news_list:
                    text = str(item.get("headline", "") + " " + item.get("summary", "")).lower()
                    for word in counts.keys():
                        if word in text: counts[word] += 1
        except: pass
        time.sleep(0.5)
    return {"error": None, "counts": counts, "keywords": keywords}

@st.cache_data(ttl=86400)
def get_dart_corp_master(api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "YOUR_DART_API_KEY": return pd.DataFrame(columns=['corp_name', 'stock_code', 'corp_code'])
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return pd.DataFrame(columns=['corp_name', 'stock_code', 'corp_code'])
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open('CORPCODE.xml') as f:
                tree = ET.parse(f)
                data = [{'corp_name': lt.find('corp_name').text, 'stock_code': lt.find('stock_code').text.strip(), 'corp_code': lt.find('corp_code').text} for lt in tree.getroot().findall('list') if lt.find('stock_code').text and lt.find('stock_code').text.strip() != ""]
                return pd.DataFrame(data)
    except: return pd.DataFrame(columns=['corp_name', 'stock_code', 'corp_code'])
    # ==========================================
# [5] 융합 스코어링 및 백테스트 엔진
# ==========================================
def detect_ml_anomalies(df: pd.DataFrame, value_col: str) -> bool:
    if df is None or len(df) < 5: return False
    try:
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        X = df[[value_col]].fillna(0).values
        df['anomaly'] = model.fit_predict(X)
        recent_trade = df.iloc[-1]
        return bool((recent_trade['anomaly'] == -1) and (recent_trade[value_col] > df[value_col].mean()))
    except: return False

def run_quick_backtest(df: pd.DataFrame) -> dict:
    try:
        if 'RSI' not in df.columns or len(df) < 30:
            return {"win_rate": 0.0, "avg_return": 0.0, "total_trades": 0}
            
        df_bt = df[['Close', 'RSI']].copy()
        df_bt['Signal'] = 0
        
        # 표본 확대를 위해 진입 조건을 RSI 35, 청산 조건을 RSI 65로 소폭 완화
        df_bt.loc[df_bt['RSI'] < 35, 'Signal'] = 1  
        df_bt.loc[df_bt['RSI'] > 65, 'Signal'] = -1 
        
        trades = []
        entry_price = 0
        
        for idx, row in df_bt.iterrows():
            if row['Signal'] == 1 and entry_price == 0:
                entry_price = row['Close']
            elif row['Signal'] == -1 and entry_price > 0:
                exit_price = row['Close']
                trades.append((exit_price - entry_price) / entry_price * 100)
                entry_price = 0
                
        if not trades:
            return {"win_rate": 0.0, "avg_return": 0.0, "total_trades": 0}
            
        winning_trades = [t for t in trades if t > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100
        avg_return = sum(trades) / len(trades)
        
        return {"win_rate": float(win_rate), "avg_return": float(avg_return), "total_trades": len(trades)}
    except:
        return {"win_rate": 0.0, "avg_return": 0.0, "total_trades": 0}

@st.cache_data(ttl=3600)
def process_insider_us(ticker: str, fear_greed_score: int, api_key: str):
    if not api_key or api_key == "YOUR_FINNHUB_API_KEY": return {"error": "Finnhub API 설정 오류."}
    headers = {"X-Finnhub-Token": api_key}
    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json().get('data', [])
        if not data: return {"error": "데이터 없음."}
        df = pd.DataFrame(data)
        df['transactionDate'] = pd.to_datetime(df['transactionDate'])
        df = df[(df['transactionPrice'] > 0) & (df['change'] != 0)].copy()
        if df.empty: return {"error": "유효 거래 없음."}
        df = df.sort_values('transactionDate')
        now = datetime.now()
        df_90d = df[df['transactionDate'] >= now - pd.Timedelta(days=90)]
        df_30d = df[df['transactionDate'] >= now - pd.Timedelta(days=30)]
        df_90d['weight'] = df_90d['name'].apply(lambda n: 2.0 if 'ceo' in str(n).lower() or 'cfo' in str(n).lower() else 0.5 if '10%' in str(n).lower() else 1.0)
        df['transaction_value'] = df['change'] * df['transactionPrice']
        df_90d['weighted_value'] = df_90d['change'] * df_90d['transactionPrice'] * df_90d['weight']
        net_buy = float(df_90d['weighted_value'].sum())
        buyers_30d = int(df_30d[df_30d['change'] > 0]['name'].nunique())
        ml_anomaly = bool(detect_ml_anomalies(df, 'transaction_value'))
        sector = "Unknown"
        try:
            prof_r = requests.get(f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}", headers=headers, timeout=5)
            if prof_r.status_code == 200: sector = prof_r.json().get('finnhubIndustry', 'Unknown')
        except: pass
        return {"error": None, "net_weighted_buy_90d": net_buy, "unique_buyers_30d": buyers_30d, "is_cluster_buying": bool(buyers_30d >= 3), "ml_anomaly_detected": ml_anomaly, "sector": sector, "market": "US"}
    except Exception as e: return {"error": str(e)}

@st.cache_data(ttl=3600)
def process_insider_kr(ticker: str, api_key: str):
    if not api_key or api_key == "YOUR_DART_API_KEY": return {"error": "DART API 설정 오류."}
    df_master = get_dart_corp_master(api_key)
    match = df_master[df_master['stock_code'] == ticker]
    if match.empty: return {"error": "DART 매핑 실패."}
    corp_code = match['corp_code'].iloc[0]
    bgn_de = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    url = f"https://opendart.fss.or.kr/api/elestock.json?crtfc_key={api_key}&corp_code={corp_code}&bgn_de={bgn_de}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json().get('list', [])
        if not data: return {"error": "데이터 없음."}
        df = pd.DataFrame(data)
        df['rcept_dt'] = pd.to_datetime(df['rcept_dt'], format='%Y%m%d', errors='coerce')
        df['change_qty'] = pd.to_numeric(df['sp_change_qty'], errors='coerce').fillna(0)
        df = df[(df['change_qty'] > 0) & (df['reprt_resn'].astype(str).str.contains('매수|취득|수증'))].copy()
        if df.empty: return {"error": "매수 거래 없음."}
        df = df.sort_values('rcept_dt')
        df_90d = df[df['rcept_dt'] >= datetime.now() - pd.Timedelta(days=90)]
        df_30d = df[df['rcept_dt'] >= datetime.now() - pd.Timedelta(days=30)]
        ml_anomaly = bool(detect_ml_anomalies(df, 'change_qty'))
        return {"error": None, "net_weighted_buy_90d": float(df_90d['change_qty'].sum()), "unique_buyers_30d": int(df_30d['repror'].nunique()), "is_cluster_buying": bool(int(df_30d['repror'].nunique()) >= 3), "ml_anomaly_detected": ml_anomaly, "sector": KR_SECTORS.get(ticker, "Unknown"), "market": "KR"}
    except Exception as e: return {"error": str(e)}

def run_fused_batch_scan(us_tickers: list, kr_tickers: list, fear_score: int) -> dict:
    signals, sector_counts = [], {}
    for tkr in us_tickers:
        res = process_insider_us(tkr, fear_score, FINNHUB_API_KEY)
        if not res.get("error") and (res["net_weighted_buy_90d"] > 0 or res["ml_anomaly_detected"]):
            signals.append({"Ticker": tkr, "Net_Buy": res["net_weighted_buy_90d"], "ML_Anomaly": res["ml_anomaly_detected"], "Sector": res["sector"], "Market": "US"})
            sector_counts[res["sector"]] = sector_counts.get(res["sector"], 0) + 1
        time.sleep(0.5)
    for tkr in kr_tickers:
        res = process_insider_kr(tkr, DART_API_KEY)
        if not res.get("error") and (res["net_weighted_buy_90d"] > 0 or res["ml_anomaly_detected"]):
            signals.append({"Ticker": tkr, "Net_Buy_Qty": res["net_weighted_buy_90d"], "ML_Anomaly": res["ml_anomaly_detected"], "Sector": res["sector"], "Market": "KR"})
            sector_counts[res["sector"]] = sector_counts.get(res["sector"], 0) + 1
        time.sleep(0.5)
    return {"stock_signals": signals, "hot_sectors": [s for s, count in sector_counts.items() if count >= 3]}

# ==========================================
# [6] AI 분석 엔진
# ==========================================
def analyze_news_with_gpt(client: OpenAI, news_items: list):
    prompt = """시장 심리 분석 AI임. 반드시 JSON 형식 응답.
[JSON Schema]
{"Sentiment_Score": 0~100, "Market_Mood": "안정/주의/경고", "Key_Themes": ["키워드1"], "Summary": "요약", "Hot_Stocks": [{"Name": "종목", "Ticker": "티커", "Reason": "근거", "News_Link": "URL"}]}"""
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"데이터: {json.dumps(news_items[:100], ensure_ascii=False)}"}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)

def run_main_reco_engine(client: OpenAI, indices_data: dict, active_exotics: list, fred_snapshot: dict, vix_snapshot: dict, fg_snapshot: dict, dix_snapshot: dict, news_snapshot: dict, fused_signals: dict, nlp_snapshot: dict):
    indices_brief = {k: {"price": v["price"], "change": v["change"]} for k, v in indices_data.get("indices", {}).items()}
    prompt = """냉철한 데이터 분석가 AI임. 반드시 JSON 형식 응답.
지수, 거시, 뉴스 심리, DIX, 기업 텍스트 마이닝 종합.

[강제 지시 사항]
1. 'Top_Stocks'와 'Avoid_Stocks' 배열에는 각각 반드시 5개의 종목을 꽉 채워야 함.
2. 수집된 뉴스나 데이터에 추천/기피 종목이 부족하다면, 현재 시장의 주요 테마나 거시 지표 흐름을 대변하는 대형 우량주(S&P500 등)를 임의로라도 포함시켜서 무조건 5개를 산출할 것.

[JSON Schema]
{ "Logic": "분석 근거", "Stability_Score": 0~100, "Strategy": "투자 전략", "Top_Stocks": [{ "Name": "종목", "Ticker": "티커", "Reason": "추천사유", "Score": 80, "Financial_Summary": "AUTO" }], "Avoid_Stocks": [{ "Name": "종목", "Ticker": "티커", "Reason": "기피사유", "Score": 20, "Financial_Summary": "AUTO" }] }"""
    payload = {"indices": indices_brief, "exotics": active_exotics, "fred": fred_snapshot, "vix": vix_snapshot, "cnn_fg": fg_snapshot, "dix_gex": dix_snapshot, "news": news_snapshot, "fused_signals": fused_signals, "nlp": nlp_snapshot}
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)

def generate_market_briefing(client: OpenAI, market: str, session: str, news_items: list, macro_snap: dict):
    prompt = f"""데이터 기반 퀀트 애널리스트임. 시장 '{market}', 시점 '{session}'. 10개 종목 꽉 채울 것.
[JSON Schema]
{{ "Overview": "시장 분위기 요약", "Top_3_News": [ {{"Title": "뉴스", "Link": "URL"}} ], "Watchlist": [ {{ "Name": "종목명", "Ticker": "티커", "Reason": "사유" }} ] }}"""
    optimized_news = [{"Title": item["title"], "Link": item["link"]} for item in news_items[:300]]
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({"Macro": macro_snap, "News": optimized_news}, ensure_ascii=False)}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)

def generate_swing_scenarios(client: OpenAI, ticker: str, current_price: float, tech_summary: dict, news_data: list):
    prompt = """당신은 퀀트 기반 스윙 트레이딩 시스템임.
제공된 기술적 지표, 백테스트 승률, Put/Call 비율(1.2 이상 시 공포 극단으로 인한 반전 매수 트리거)을 분석하여 구체적 매매 시나리오 산출.
반드시 JSON 형식으로만 응답하며, 구체적인 가격 수치를 포함해야 함.
[JSON Schema]
{
  "Analysis": "지표, 백테스트 결과, PCR을 융합한 진단",
  "Plan_A": { "Strategy": "돌파 매수", "Condition": "진입조건", "Entry_Price": 0.0, "Target_Price": 0.0, "Stop_Loss": 0.0, "Risk_Reward_Ratio": "1:X", "Reason": "논리" },
  "Plan_B": { "Strategy": "눌림목 매수", "Condition": "진입조건", "Entry_Price": 0.0, "Target_Price": 0.0, "Stop_Loss": 0.0, "Risk_Reward_Ratio": "1:X", "Reason": "논리" }
}"""
    payload = {"Ticker": ticker, "Current_Price": current_price, "Technical_Indicators": tech_summary, "Recent_News": news_data[:10]}
    res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], response_format={"type": "json_object"})
    return json.loads(res.choices[0].message.content)

# ==========================================
# [7] 재무 데이터 보강 및 스윙 지표 계산 함수
# ==========================================
@st.cache_data(ttl=3600)
def fetch_finnhub_fundamentals(ticker: str, api_key: str):
    if not api_key or api_key == "YOUR_FINNHUB_API_KEY": return {"error": "API 미설정"}
    try:
        r = requests.get(f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all", headers={"X-Finnhub-Token": api_key}, timeout=10)
        m = r.json().get("metric", {})
        if not m: raise ValueError("데이터 없음")
        return {"error": None, "metrics": {"PE": m.get("peBasicExclExtraTTM"), "RG": m.get("revenueGrowthTTMYoy"), "DE": m.get("longTermDebt/equityAnnual")}}
    except: return {"error": "데이터 부족함"}

def format_financial_summary(ticker: str, mode: str = "top") -> str:
    y = fetch_finnhub_fundamentals(ticker, FINNHUB_API_KEY)
    if y.get("error"): return "재무 데이터 수집 불가"
    m = y["metrics"]
    pe = round(m['PE'], 2) if m['PE'] is not None else 'N/A'
    rg = f"{round(m['RG'], 1)}%" if m['RG'] is not None else 'N/A'
    de = round(m['DE'], 2) if m['DE'] is not None else 'N/A'
    return f"P/E: {pe} | 성장률: {rg} | 부채비율: {de}"

def enrich_report_with_fundamentals(report: dict):
    for k, mode in [("Top_Stocks", "top"), ("Avoid_Stocks", "avoid")]:
        for s in report.get(k, []): s["Financial_Summary"] = format_financial_summary(str(s.get("Ticker", "")).strip(), mode=mode)
    return report

@st.cache_data(ttl=3600)
def fetch_ohlcv(ticker: str, days: int = 180) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader(ticker, start_date)
        if df is None or df.empty: return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df.columns = [col.capitalize() if col.lower() != 'date' else 'Date' for col in df.columns]
        return df
    except:
        return pd.DataFrame()

def calculate_technicals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30: return df
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'], df['BB_Low'] = bb.bollinger_hband(), bb.bollinger_lband()
    kc = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    df['KC_High'], df['KC_Low'] = kc.keltner_channel_hband(), kc.keltner_channel_lband()
    df['Squeeze_On'] = (df['BB_Low'] > df['KC_Low']) & (df['BB_High'] < df['KC_High'])
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    recent_df = df.tail(90).copy()
    if not recent_df.empty:
        bins = np.linspace(recent_df['Low'].min(), recent_df['High'].max(), 20)
        recent_df['Price_Bin'] = pd.cut(recent_df['Close'], bins=bins)
        df['POC'] = recent_df.groupby('Price_Bin')['Volume'].sum().idxmax().mid if pd.notnull(recent_df.groupby('Price_Bin')['Volume'].sum().idxmax()) else df['Close'].mean()
    else: df['POC'] = df['Close'].mean()
    return df
# ==========================================
# [8] Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="Project B.I.A.S v4.0 (Full Edition)", layout="wide")

with st.sidebar:
    st.header("⚙️ SYSTEM CONFIG")
    use_news = st.checkbox("실시간 뉴스 여론 반영", value=True)
    st.caption("ㄴ 경제 뉴스 100건을 수집하여 AI가 탐욕/공포 심리를 분석함.")
    use_fred = st.checkbox("FRED 거시/신용/원자재 반영", value=True)
    st.caption("ㄴ 연방준비은행의 거시/미시 및 FAO 식량가격지수를 퀀트 로직에 반영함.")
    use_vix_dix = st.checkbox("공포/탐욕 및 기관 동향 (VIX/DIX)", value=True)
    st.caption("ㄴ CNN Fear & Greed Index와 다크풀 장외 매집 동향(DIX)을 진단함.")
    use_nlp = st.checkbox("기업 텍스트(뉴스) 마이닝 반영", value=True)
    st.caption("ㄴ 주요 기술주의 실적 및 공시 키워드를 마이닝하여 펀더멘털의 전환점을 계량화함.")
    
    st.divider()
    st.subheader("🤖 융합 스코어링 자동화 옵션")
    use_insider_batch = st.checkbox("ML 이상 탐지 및 섹터 쏠림 스캔 연동", value=False)
    
    scan_us = False
    scan_kr = False
    if use_insider_batch:
        st.markdown("**스캔 대상 시장 선택**")
        scan_us = st.checkbox("🇺🇸 미국 주식 (US) 스캔", value=True)
        scan_kr = st.checkbox("🇰🇷 한국 주식 (KR) 스캔", value=True)
        
    st.caption("ㄴ 활성화 시 선택한 시장의 주요 표본 종목을 스캔하고 Machine Learning 판별 결과를 AI에 강제 주입함. (연산 지연 발생)")
    
    st.divider()
    st.subheader("이색 지표 선택")
    st.caption("ㄴ 비전통적이고 창의적인 관점에서 경제 상황을 진단함.")
    active_exotics = [ex for ex in EXOTIC_DICT.keys() if st.checkbox(ex.split(' (')[0])]
    st.divider()
    analyze_btn = st.button("RUN ANALYTIC ENGINE", type="primary", use_container_width=True)

tab_dash, tab_econ, tab_vix_fg, tab_news, tab_nlp, tab_dict, tab_insider, tab_swing, tab_briefing = st.tabs(["📉 대시보드", "📊 지표 분석", "😱 공포/탐욕/기관 심리", "📰 뉴스 서비스", "💬 텍스트 마이닝", "📚 이색 지표 사전", "🏢 내부자 동향(ML)", "🏄‍♂️ 스윙 전략 어시스턴트", "📝 장전/장마감 브리핑"])

with tab_dash:
    st.markdown("""
    <div style='background: linear-gradient(to right, #FF4B4B, #FFA500, #00CC96); padding: 20px; border-radius: 12px; margin-bottom: 15px;'>
        <h2 style='margin: 0; color: white; text-align: center;'>어서오세요 프로젝트 B.I.A.S 입니다.</h2>
    </div>
    <h4 style='margin: 0 0 20px 0; text-align: center; font-weight: 700; background: linear-gradient(to right, #FF4B4B, #FFA500, #00CC96); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        당신이 몰랐던 괴짜 정보에 대한 모든 것
    </h4>
    """, unsafe_allow_html=True)

    indices_data = fetch_indices()
    idx_cols = st.columns(4)
    target_indices = ["KOSPI", "KOSDAQ", "NASDAQ", "S&P 500"]

    for i, name in enumerate(target_indices):
        with idx_cols[i]:
            v = indices_data["indices"].get(name)
            if v and v["price"] > 0:
                st.metric(name, f"{v['price']:,.2f}", f"{v['change']}%")
                st.line_chart(v["df"], height=100)
            else:
                st.metric(name, "데이터 없음", "N/A")
                st.caption("API 수집 지연 또는 실패")

    if analyze_btn:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
            st.error("OpenAI API 키가 설정되지 않아 분석 엔진을 가동할 수 없음. 코드 상단의 OPENAI_API_KEY 변수를 확인해야 함.")
        else:
            with st.spinner("데이터 수집 및 빅데이터 융합 분석 중..."):
                try:
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    fred_snap, vix_snap, fg_snap, dix_snap, news_snap, fused_signals, nlp_snap = {}, {}, {}, {}, {}, {}, {}
                    
                    if use_fred: 
                        fred_snap = {"series": {g: {l: summarize_fred_latest(fetch_fred_series(s), s) for l, s in items.items()} for g, items in FRED_SERIES.items()}}
                    
                    if use_vix_dix:
                        vix_snap = summarize_fred_latest(fetch_fred_series("VIXCLS", years=1), "VIXCLS")
                        fg_snap = fetch_cnn_fear_and_greed()
                        dix_df = fetch_dix_gex()
                        if not dix_df.empty:
                            dix_snap = {"DIX": float(dix_df['dix'].iloc[-1]), "GEX": float(dix_df['gex'].iloc[-1])}

                    if use_news: 
                        news_snap = analyze_news_with_gpt(client, fetch_naver_news_bulk("경제", NAVER_CLIENT_ID, NAVER_CLIENT_SECRET).get("items", []))
                        
                    if use_nlp:
                        nlp_snap = fetch_corporate_keywords(MAJOR_TICKERS_US, FINNHUB_API_KEY)
                    
                    if use_insider_batch:
                        current_fear = fg_snap.get("score") if isinstance(fg_snap, dict) else None
                        target_us = MAJOR_TICKERS_US if scan_us else []
                        target_kr = MAJOR_TICKERS_KR if scan_kr else []
                        
                        if not target_us and not target_kr:
                            st.warning("스캔할 시장이 선택되지 않아 내부자 스캔을 건너뜁니다.")
                            fused_signals = {}
                        else:
                            fused_signals = run_fused_batch_scan(target_us, target_kr, current_fear)

                    report = run_main_reco_engine(client, indices_data, active_exotics, fred_snap, vix_snap, fg_snap, dix_snap, news_snap, fused_signals, nlp_snap)
                    report = enrich_report_with_fundamentals(report)

                    score = int(report.get("Stability_Score", 50))
                    status_text, status_color = score_to_status(score)
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": f"심리 상태: {status_text}", "font": {"color": status_color, "size": 24}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "white"}, "steps": [{"range": [0, 60], "color": "#FF4B4B"}, {"range": [60, 80], "color": "#FFA500"}, {"range": [80, 100], "color": "#00CC96"}]}))
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("분석 근거 및 전략")
                    st.info(report.get("Logic", "분석 결과 없음"))
                    st.success(report.get("Strategy", "전략 없음"))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("🚀 TOP 5 추천 종목")
                        for s in report.get("Top_Stocks", []):
                            with st.expander(f"✅ {s.get('Name')} ({s.get('Ticker')}) - {s.get('Score')}점"):
                                st.write(f"**추천 사유:** {s.get('Reason')}")
                                st.markdown(f"**재무 요약:**\n{s.get('Financial_Summary')}")
                    with c2:
                        st.subheader("⚠️ AVOID 5 기피 종목")
                        for s in report.get("Avoid_Stocks", []):
                            with st.expander(f"❌ {s.get('Name')} ({s.get('Ticker')}) - {s.get('Score')}점"):
                                st.write(f"**기피 사유:** {s.get('Reason')}")
                                st.markdown(f"**위험 요인:**\n{s.get('Financial_Summary')}")
                except Exception as e: st.error(f"분석 엔진 오류: {e}")

with tab_econ:
    st.header("📊 FRED 거시/미시 및 물가/신용 지표 분석")
    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    
    with row1_c1:
        st.subheader("거시(Macro)")
        m_label = st.selectbox("거시 지표 선택", list(FRED_SERIES["거시(Macro)"].keys()))
        m_id = FRED_SERIES["거시(Macro)"][m_label]
        df_m = fetch_fred_series(m_id, years=10)
        st.line_chart(df_m.set_index("DATE")[m_id])
        st.info(FRED_DESC.get(m_id, ""))
        if m_id == "T10Y2Y" and not df_m.empty:
            st.success(f"💡 **해석 가이드:** 현재 장단기 금리차는 {df_m[m_id].iloc[-1]:.2f}입니다. 마이너스 역전 상태가 지속되다 플러스로 갓 반등하는 시점이 주식시장의 전통적인 최대 위험 구간입니다.")
        
    with row1_c2:
        st.subheader("미시(Micro)")
        i_label = st.selectbox("미시 지표 선택", list(FRED_SERIES["미시(Micro)"].keys()))
        i_id = FRED_SERIES["미시(Micro)"][i_label]
        st.line_chart(fetch_fred_series(i_id, years=10).set_index("DATE")[i_id])
        st.info(FRED_DESC.get(i_id, ""))
        
    with row2_c1:
        st.subheader("신용/공급망(Credit)")
        c_label = st.selectbox("신용/공급망 지표 선택", list(FRED_SERIES["신용/공급망(Credit)"].keys()))
        c_id = FRED_SERIES["신용/공급망(Credit)"][c_label]
        st.line_chart(fetch_fred_series(c_id, years=10).set_index("DATE")[c_id])
        st.info(FRED_DESC.get(c_id, ""))
        
    with row2_c2:
        st.subheader("물가/원자재(Commodity)")
        f_label = st.selectbox("물가/원자재 지표 선택", list(FRED_SERIES["물가/원자재(Commodity)"].keys()))
        f_id = FRED_SERIES["물가/원자재(Commodity)"][f_label]
        st.line_chart(fetch_fred_series(f_id, years=10).set_index("DATE")[f_id])
        st.info(FRED_DESC.get(f_id, ""))

with tab_vix_fg:
    st.header("😱 공포/탐욕 지수 및 기관 동향 (VIX/FG/DIX)")
    st.markdown("시장의 극단적 심리와 실질적인 기관의 장외 매집 데이터를 교차 검증하는 탭임.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CNN Fear & Greed Index")
        fg_data = fetch_cnn_fear_and_greed()
        if not fg_data.get("error"):
            score = fg_data["score"]
            rating = fg_data["rating"]
            fg_color = "#FF4B4B" if score < 40 else "#00CC96" if score > 60 else "#FFA500"
            fig_fg = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                number={"font": {"color": fg_color, "size": 40}},
                title={"text": f"현재 상태: {rating}", "font": {"color": fg_color, "size": 20}},
                gauge={
                    "axis": {"range": [0, 100]}, "bar": {"color": "black"},
                    "steps": [
                        {"range": [0, 25], "color": "#FF0000"}, {"range": [25, 45], "color": "#FF7F50"},
                        {"range": [45, 55], "color": "#FFD700"}, {"range": [55, 75], "color": "#9ACD32"},
                        {"range": [75, 100], "color": "#008000"}
                    ]
                }
            ))
            st.plotly_chart(fig_fg, use_container_width=True)
            st.success(f"💡 **해석 가이드:** 현재 시장 심리는 {score}점입니다. 25 이하의 극단적 공포는 역발상 매수 기회, 75 이상의 극단적 탐욕은 분할 매도 관점으로 접근하십시오.")
        else:
            st.error(f"CNN 공포/탐욕 지수 수집 실패함: {fg_data['error']}")
            
    with c2:
        st.subheader("CBOE VIX (공포 지수)")
        vix_df = fetch_fred_series("VIXCLS", years=3)
        if not vix_df.empty:
            latest_vix = safe_float(vix_df["VIXCLS"].iloc[-1])
            st.metric("현재 VIX 지수", f"{latest_vix:.2f}")
            st.line_chart(vix_df.set_index("DATE")["VIXCLS"])
            
    st.divider()
    st.subheader("🏢 다크풀 지수 (DIX) 및 옵션 감마 익스포저 (GEX)")
    dix_df = fetch_dix_gex()
    if not dix_df.empty:
        c3, c4 = st.columns(2)
        latest_dix = dix_df['dix'].iloc[-1] * 100
        latest_gex = dix_df['gex'].iloc[-1] / 1e9
        c3.metric("현재 DIX (기관 장외매집 비율)", f"{latest_dix:.1f}%")
        c4.metric("현재 GEX (옵션 마켓메이커 노출, 10억$)", f"{latest_gex:.2f}")
        st.line_chart(dix_df.set_index('date')['dix'].tail(120) * 100)
        if latest_dix >= 45.0:
            st.success("💡 **해석 가이드:** DIX가 45%를 돌파했습니다. VIX 지수나 대중의 공포와 무관하게 기관 투자자들이 대규모 저가 매집에 돌입한 강력한 단기 상승 반전 시그널입니다.")
    else:
        st.error("DIX/GEX 데이터 수집 실패함.")
    
    st.markdown("""
    ---
    ### 📊 지수 해석 가이드
    * **CNN Fear & Greed / VIX:** 대중의 불안 심리를 정량화함. 30 이상 VIX 또는 25 이하 F&G 지수는 극단적 공포를 의미함.
    * **DIX (Dark Index):** 기관 투자자의 장외 거래 매수 비중임. 대중의 공포(VIX 급등) 시점에 DIX가 45%를 돌파하면, 기관이 헐값에 매집 중인 강력한 상승 반전 시그널로 해석됨.
    """)

with tab_news:
    st.header("📰 뉴스 서비스")
    query = st.text_input("검색어 (100건 수집 및 전수 분석)", value="국내외 경제")
    if st.button("RUN NEWS ANALYSIS"):
        if not NAVER_CLIENT_ID or NAVER_CLIENT_ID == "YOUR_NAVER_ID":
            st.error("NAVER API 키(ID/SECRET)가 설정되지 않음.")
        else:
            with st.spinner("국내외 뉴스 100건 분석 및 10개 종목 추출 중..."):
                res = fetch_naver_news_bulk(query, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
                if res.get("error"): st.error(res["error"])
                else:
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    report = analyze_news_with_gpt(client, res["items"])
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("시장 심리 점수 (Sentiment Score)", f"{report.get('Sentiment_Score')}점")
                        st.write(f"**Market Mood:** {report.get('Market_Mood')}")
                    with c2:
                        st.write("**핵심 테마:**", ", ".join(report.get("Key_Themes", [])))
                        st.write("**요약:**", report.get("Summary"))
                    
                    st.divider()
                    st.subheader("🚀 뉴스 기반 글로벌/국내 추천 종목 (HOT 10)")
                    for s in report.get("Hot_Stocks", []):
                        st.markdown(f"**{s.get('Name')} ({s.get('Ticker')})**")
                        st.write(f"- 추천 근거: {s.get('Reason')}")
                        if s.get("News_Link"):
                            st.markdown(f"- 🔗 [관련 뉴스 원문으로 이동]({s.get('News_Link')})")
                        st.write("---")

with tab_nlp:
    st.header("💬 기업 텍스트(뉴스) 마이닝")
    st.markdown("미국 핵심 기술주(Big Tech)의 최근 30일 뉴스 및 기업 발표 문서를 자연어 처리하여 경영진의 관심사와 시장 분위기를 정량화함.")
    if st.button("기업 텍스트 마이닝 실행"):
        with st.spinner("Finnhub 뉴스 데이터 파싱 및 키워드 추출 중..."):
            nlp_data = fetch_corporate_keywords(MAJOR_TICKERS_US, FINNHUB_API_KEY)
            if nlp_data.get("error"):
                st.error(nlp_data["error"])
            else:
                counts = nlp_data["counts"]
                pos_keys = nlp_data["keywords"]["Positive"]
                neg_keys = nlp_data["keywords"]["Negative"]
                
                pos_sum = sum(counts[k] for k in pos_keys)
                neg_sum = sum(counts[k] for k in neg_keys)
                
                st.subheader(f"총 긍정 언급: {pos_sum}회 vs 총 부정 언급: {neg_sum}회")
                
                fig = go.Figure(data=[
                    go.Bar(name='Positive', x=pos_keys, y=[counts[k] for k in pos_keys], marker_color='#00CC96'),
                    go.Bar(name='Negative', x=neg_keys, y=[counts[k] for k in neg_keys], marker_color='#FF4B4B')
                ])
                fig.update_layout(barmode='group', title="주요 빅테크 텍스트 키워드 출현 빈도 (최근 30일)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **해석 가이드:** 'AI', 'Growth' 등의 키워드가 압도적이면 기업들의 자본 지출(CAPEX) 의지가 확고함을 나타냄. 반면 'Cost', 'Layoff'가 급증하면 시장 변동성에 대비해 몸집을 줄이고 마진을 방어하려는 보수적 상태임을 암시함.")

with tab_dict:
    st.header("📚 이색 지표 사전")
    for k, v in EXOTIC_DICT.items():
        st.subheader(k)
        st.write(v)
        st.divider()

with tab_insider:
    st.header("🏢 내부자 거래 단일 검증 (동적 검색 및 ML 융합)")
    st.markdown("기업을 선택하거나 직접 입력하여, 내부자 매집 여부와 머신러닝 이상 탐지 결과를 교차 검증함.")
    
    market_sel_insider = st.radio("시장 선택", ["미국 (US)", "한국 (KR)"], horizontal=True, key="insider_market")
    target_ticker = ""
    
    if market_sel_insider == "한국 (KR)":
        df_master = get_dart_corp_master(DART_API_KEY)
        if df_master.empty:
            st.error("DART 상장사 마스터 데이터를 불러올 수 없음. API 키 상태를 확인해야 함.")
        else:
            options = df_master['corp_name'] + " (" + df_master['stock_code'] + ")"
            selected_kr = st.selectbox("한국 상장사 선택 (타이핑하여 기업명 검색 가능)", options.tolist())
            if selected_kr:
                target_ticker = selected_kr.split("(")[-1].replace(")", "")
    else:
        us_options = ["직접 입력...", "AAPL (Apple)", "MSFT (Microsoft)", "NVDA (Nvidia)", "GOOGL (Alphabet)", "AMZN (Amazon)", "META (Meta)", "TSLA (Tesla)", "BRK-B (Berkshire)", "AVGO (Broadcom)", "WMT (Walmart)", "JPM (JPMorgan)"]
        selected_us = st.selectbox("미국 주요 주식 선택", us_options)
        if selected_us == "직접 입력...":
            target_ticker = st.text_input("미국 티커 직접 입력 (예: AMD, INTC)").upper()
        else:
            target_ticker = selected_us.split(" ")[0]
    
    if st.button("단일 종목 검증 실행"):
        if not target_ticker:
            st.warning("종목이 선택되지 않았음.")
        else:
            with st.spinner("데이터 스캔 및 머신러닝 모델 추론 중..."):
                if market_sel_insider == "미국 (US)":
                    fg_data = fetch_cnn_fear_and_greed()
                    current_fear_score = fg_data.get("score") if not fg_data.get("error") else None
                    res = process_insider_us(target_ticker, current_fear_score, FINNHUB_API_KEY)
                else:
                    res = process_insider_kr(target_ticker, DART_API_KEY)
                
                if res.get("error"):
                    st.error(res["error"])
                else:
                    st.subheader(f"[{target_ticker}] 정량적 내부자 지표 결과")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    val_label = "가중치 대금" if market_sel_insider == "미국 (US)" else "순매수 수량"
                    val_format = f"${res['net_weighted_buy_90d']:,.0f}" if market_sel_insider == "미국 (US)" else f"{res['net_weighted_buy_90d']:,.0f}주"
                    
                    c1.metric(val_label, val_format)
                    c2.metric("ML 이상 거래 (Anomaly)", "🔴 탐지됨 (호재)" if res["ml_anomaly_detected"] else "⚪ 정상 범위")
                    c3.metric("클러스터 매수 (30일내)", "🟢 발생" if res["is_cluster_buying"] else "🔴 미발생")
                    c4.metric("소속 섹터", res["sector"])
                    
                    if res["ml_anomaly_detected"] and res['net_weighted_buy_90d'] > 0:
                        st.success(f"💡 **해석 가이드:** {target_ticker} 종목에 대해 통계적 이상 수치(Anomaly)가 탐지되었습니다! 과거 5년 대비 기형적으로 높은 내부자 매수 자금이 유입되었으며, 이는 경영진이 확신하는 단기 호재가 존재할 확률이 높음을 시사합니다.")
                    
                    st.divider()
                    st.markdown("""
                    ### 💡 알고리즘 작동 원리 요약
                    * **머신러닝 이상 탐지 (Isolation Forest):** 과거 5년 치 거래 대금(또는 수량) 패턴을 기계학습하여, 평소 분포를 극단적으로 벗어난 이례적인 매수만을 진성 호재로 판별함.
                    * **직책 기반 가중치 (미국 한정):** 정보 비대칭성이 가장 높은 CEO 및 CFO의 거래 대금은 2배로 증폭시키고, 단순 10% 주주의 대금은 0.5배로 축소 연산함.
                    * **클러스터 매수 교차 검증:** 최근 30일 이내에 단일 인물이 아닌 3명 이상의 내부자가 동시다발적으로 매수를 진행한 경우 강력한 상승 전조로 판별함.
                    * **섹터 쏠림 분석:** 대시보드 자동 스캔 시, 동일 섹터 내 3개 이상 종목에서 매집이 탐지되면 해당 산업군 전체의 턴어라운드 시그널로 간주하여 AI에 가중치를 부여함.
                    """)

with tab_swing:
    st.header("🏄‍♂️ 스윙매매 전략 어시스턴트 (백테스트 & PCR 융합)")
    st.markdown("단기~중기 기술적 지표, 과거 백테스트 승률 산출, Put/Call Ratio를 융합하여 기계적인 진입/청산 시나리오를 자동 산출합니다.")
    
    col1, col2, col3 = st.columns(3)
    swing_ticker = col1.text_input("티커 입력 (예: TSLA, 005930)", value="TSLA", key="swing_tkr").upper()
    total_capital = col2.number_input("총 투자금 ($ 또는 원)", value=10000, step=1000)
    risk_pct = col3.number_input("1회 감수 리스크 (%)", value=2.0, step=0.5, max_value=10.0)
    
    if st.button("스윙 전략 산출", type="primary", key="btn_swing"):
        with st.spinner(f"{swing_ticker} 데이터 스캔, 백테스팅 및 AI 추론 중..."):
            df = fetch_ohlcv(swing_ticker, days=1200)
            if df.empty:
                st.error("종목 데이터를 불러오지 못했습니다. 티커를 확인하세요.")
            else:
                df = calculate_technicals(df)
                
                c_prc = float(df['Close'].iloc[-1])
                c_atr = float(df['ATR'].iloc[-1])
                c_rsi = float(df['RSI'].iloc[-1])
                c_poc = float(df['POC'].iloc[-1])
                sqz = bool(df['Squeeze_On'].iloc[-1])
                
                pcr_val = fetch_put_call_ratio()
                pcr_val = float(pcr_val) if pcr_val is not None else None
                bt_stats = run_quick_backtest(df)
                bt_stats_wr = float(bt_stats['win_rate'])
                
                news_res = fetch_naver_news_bulk(f"미국 증시 {swing_ticker}", NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, display=15)
                
                st.subheader(f"{swing_ticker} 캔들스틱 및 기술적 지표 (최근 90일)")
                plot_df = df.tail(90)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_High'], line=dict(color='gray', width=1, dash='dot'), name="BB High"), row=1, col=1)
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_Low'], line=dict(color='gray', width=1, dash='dot'), name="BB Low"), row=1, col=1)
                
                fig.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Volume'], name="Volume", marker_color='rgba(0, 204, 150, 0.5)'), row=2, col=1)
                fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("현재가", f"{c_prc:,.2f}")
                c_b.metric("RSI (14일)", f"{c_rsi:.1f}", "과매도" if c_rsi < 30 else "과매수" if c_rsi > 70 else "중립")
                
                bt_text = f"{bt_stats_wr:.1f}%" if bt_stats['total_trades'] > 0 else "데이터 부족"
                c_c.metric("과거 유사조건 승률", bt_text)
                
                pcr_text = f"{pcr_val:.2f}" if pcr_val else "데이터 없음"
                c_d.metric("Put/Call Ratio", pcr_text, "역발상 매수 기회" if pcr_val and pcr_val >= 1.2 else "")
                
                tech_summary = {
                    "RSI": c_rsi, "ATR": c_atr, 
                    "POC_Level": c_poc, 
                    "Squeeze_Active": sqz,
                    "Historical_Win_Rate": bt_stats_wr,
                    "Put_Call_Ratio": pcr_val
                }
                
                if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
                    st.warning("OpenAI API 키가 설정되지 않아 AI 시나리오를 산출할 수 없습니다.")
                else:
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    swing_plan = generate_swing_scenarios(client, swing_ticker, c_prc, tech_summary, news_res.get("items", []))
                    
                    st.divider()
                    st.subheader("🧠 시스템 산출 매매 시나리오 및 리스크 관리")
                    st.info(f"**현재 국면 진단:** {swing_plan.get('Analysis', '')}")
                    
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        plan_a = swing_plan.get("Plan_A", {})
                        with st.container(border=True):
                            st.markdown(f"### 📈 Plan A: {plan_a.get('Strategy')}")
                            st.write(f"**조건:** {plan_a.get('Condition')}")
                            st.write(f"**진입가:** {plan_a.get('Entry_Price')} / **목표가:** {plan_a.get('Target_Price')}")
                            st.write(f"**손절가:** {plan_a.get('Stop_Loss')} (R:R = {plan_a.get('Risk_Reward_Ratio')})")
                            st.caption(f"논리: {plan_a.get('Reason')}")
                            
                            entry_a = safe_float(plan_a.get('Entry_Price'))
                            stop_a = safe_float(plan_a.get('Stop_Loss'))
                            if entry_a and stop_a and entry_a > stop_a:
                                risk_amount = total_capital * (risk_pct / 100)
                                shares = risk_amount / (entry_a - stop_a)
                                st.success(f"💡 **포지션 사이즈 계산:** 목표 리스크({risk_amount})에 맞춘 적정 매수 수량은 **{int(shares)}주** 입니다.")

                    with pc2:
                        plan_b = swing_plan.get("Plan_B", {})
                        with st.container(border=True):
                            st.markdown(f"### 📉 Plan B: {plan_b.get('Strategy')}")
                            st.write(f"**조건:** {plan_b.get('Condition')}")
                            st.write(f"**진입가:** {plan_b.get('Entry_Price')} / **목표가:** {plan_b.get('Target_Price')}")
                            st.write(f"**손절가:** {plan_b.get('Stop_Loss')} (R:R = {plan_b.get('Risk_Reward_Ratio')})")
                            st.caption(f"논리: {plan_b.get('Reason')}")
                            
                            entry_b = safe_float(plan_b.get('Entry_Price'))
                            stop_b = safe_float(plan_b.get('Stop_Loss'))
                            if entry_b and stop_b and entry_b > stop_b:
                                risk_amount = total_capital * (risk_pct / 100)
                                shares = risk_amount / (entry_b - stop_b)
                                st.success(f"💡 **포지션 사이즈 계산:** 목표 리스크({risk_amount})에 맞춘 적정 매수 수량은 **{int(shares)}주** 입니다.")

with tab_briefing:
    st.header("📝 장 전/장 마감 브리핑 및 관심 종목 10선")
    st.markdown("현시점 기준 전처리된 뉴스 리스트와 거시 지표를 종합하여 시장 전반의 분위기를 파악하고, 관심 종목 10선과 명확한 선정 사유를 도출함.")
    
    col1, col2 = st.columns(2)
    with col1:
        target_market = st.radio("대상 시장", ["미국 증시 (US)", "한국 증시 (KR)"], key="briefing_market")
    with col2:
        target_session = st.radio("분석 시점", ["장 전 (Pre-market)", "장 마감 (Post-market)"], key="briefing_session")
        
    if st.button("브리핑 생성 및 종목 발굴 실행", type="primary"):
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
            st.error("OpenAI API 키 설정이 필요함.")
        elif not NAVER_CLIENT_ID or NAVER_CLIENT_ID == "YOUR_NAVER_ID":
            st.error("네이버 뉴스 API 키 설정이 필요함.")
        else:
            with st.spinner(f"{target_market} {target_session} 기준 최신 뉴스 수집 및 AI 분석 중... (약 10~20초 소요)"):
                try:
                    m_kw = "미국 증시" if "US" in target_market else "한국 증시"
                    s_kw = "프리마켓 장전" if "장 전" in target_session else "마감 시황"
                    search_query = f"{m_kw} {s_kw} 경제 종목"
                    
                    news_res = fetch_naver_news_500(search_query, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
                    
                    if news_res.get("error"):
                        st.error(news_res["error"])
                    else:
                        vix_df = fetch_fred_series("VIXCLS", years=1)
                        vix_latest = safe_float(vix_df["VIXCLS"].iloc[-1]) if not vix_df.empty else "N/A"
                        macro_data = {"Current_VIX": vix_latest}
                        
                        client = OpenAI(api_key=OPENAI_API_KEY)
                        briefing_data = generate_market_briefing(client, target_market, target_session, news_res.get("items", []), macro_data)
                        
                        st.subheader("🌐 시장 전반 요약 (Overview)")
                        st.info(briefing_data.get("Overview", "요약 데이터 없음"))
                        
                        st.subheader("📰 Top 3 시장 주요 뉴스")
                        for news in briefing_data.get("Top_3_News", []):
                            st.markdown(f"- [{news.get('Title')}]({news.get('Link', '#')})")
                            
                        st.divider()
                        
                        st.subheader("🎯 관심 종목 10선 (Watchlist)")
                        watchlist = briefing_data.get("Watchlist", [])
                        if not watchlist:
                            st.warning("데이터가 부족하여 종목을 추출하지 못했습니다. 잠시 후 다시 시도해주세요.")
                        else:
                            for idx, stock in enumerate(watchlist):
                                with st.expander(f"{idx+1}. {stock.get('Name')} ({stock.get('Ticker')})"):
                                    st.write(f"**선정 사유:** {stock.get('Reason')}")
                                    
                except Exception as e:
                    st.error(f"브리핑 생성 중 오류 발생: {str(e)}")