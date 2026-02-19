"""
Phase 2: RSS 크롤링 시도 코드 (실패 사례)
AI 에듀테크 뉴스클리핑 자동화 시스템

실제 진행 결과:
  - RSS 크롤링으로 데이터 수집 시도
  - 다수 언론사 RSS 피드 접근 불가 또는 불완전 데이터 확인
  - 빅카인즈 수동 다운로드 방식으로 전환

주의: API 호출 없는 순수 로컬 크롤링 코드입니다.
"""

import time
import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
import json


# ============================================================
# 메트릭 측정
# ============================================================

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_min = (time.time() - start) / 60
        print(f"[메트릭] {func.__name__}: {elapsed_min:.2f}분")
        return result, elapsed_min
    return wrapper


# ============================================================
# RSS 크롤링 시도 (실패 사례)
# ============================================================

# 시도한 언론사 RSS 피드 목록
RSS_FEEDS = {
    "연합뉴스_교육": "https://www.yna.co.kr/rss/society.xml",
    "한국경제_교육": "https://www.hankyung.com/feed/education",
    "조선일보_교육": "https://www.chosun.com/arc/outboundfeeds/rss/category/national/education/?outputType=xml",
    "중앙일보_교육": "https://rss.joins.com/joins_news_list.xml",
    "YTN": "https://www.ytn.co.kr/rss/rss.php?ct_id=01",
    "MBC": "https://imnews.imbc.com/rss/news/news_0000.xml",
    "KBS": "https://news.kbs.co.kr/rss/rss.xml",
    "교육부_공식": "https://www.moe.go.kr/boardCnts/rss.do?boardID=342",
}

KEYWORDS = [
    "AI 챗봇", "LMS", "에듀테크", "무전공", "자율전공",
    "글로컬", "교육부", "대학혁신", "AI 상담",
]


def fetch_rss_feed(url, timeout=10):
    """
    단일 RSS 피드 요청 (로컬 실행, API 없음)
    실패 시 오류 타입과 원인 기록
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewsClipBot/1.0)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    }
    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
            return raw.decode(encoding, errors="replace"), None
    except HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        return None, f"URL Error: {e.reason}"
    except Exception as e:
        return None, f"Unknown: {str(e)}"


def parse_rss_items(xml_content, source_name):
    """
    RSS XML 파싱 → 뉴스 아이템 추출
    """
    items = []
    try:
        root = ET.fromstring(xml_content)

        # RSS 2.0 또는 Atom 구조 처리
        channel = root.find("channel")
        if channel is None:
            # Atom feed
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            for entry in entries:
                title = entry.findtext("atom:title", namespaces=ns) or ""
                pub_date = entry.findtext("atom:published", namespaces=ns) or ""
                link = ""
                link_el = entry.find("atom:link", ns)
                if link_el is not None:
                    link = link_el.get("href", "")
                items.append({
                    "source": source_name,
                    "title": title.strip(),
                    "date": pub_date[:10] if pub_date else "",
                    "link": link,
                })
        else:
            # RSS 2.0
            for item in channel.findall("item"):
                title = item.findtext("title") or ""
                pub_date = item.findtext("pubDate") or ""
                link = item.findtext("link") or ""
                items.append({
                    "source": source_name,
                    "title": title.strip(),
                    "date": pub_date[:16] if pub_date else "",
                    "link": link,
                })
    except ET.ParseError as e:
        print(f"    XML 파싱 오류: {e}")

    return items


def filter_by_keywords(items, keywords):
    """키워드 필터링 (로컬 처리, API 없음)"""
    filtered = []
    for item in items:
        title = item.get("title", "")
        for kw in keywords:
            if kw in title:
                item["matched_keyword"] = kw
                filtered.append(item)
                break
    return filtered


def clean_text(text):
    """특수문자 제거 전처리"""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w\s가-힣a-zA-Z0-9.,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@measure_time
def run_rss_crawling():
    """
    RSS 크롤링 전체 시도 + 실패 기록
    """
    print("=== Phase 2: RSS 크롤링 시도 ===\n")

    crawl_report = {
        "attempted": len(RSS_FEEDS),
        "success": 0,
        "failed": 0,
        "total_items": 0,
        "filtered_items": 0,
        "failures": [],
        "results": [],
    }

    all_items = []

    for source_name, url in RSS_FEEDS.items():
        print(f"  [{source_name}] 시도 중...")
        xml_content, error = fetch_rss_feed(url)

        if error:
            print(f"    실패: {error}")
            crawl_report["failed"] += 1
            crawl_report["failures"].append({
                "source": source_name,
                "url": url,
                "error": error,
                "impact": "해당 언론사 뉴스 누락",
            })
            continue

        items = parse_rss_items(xml_content, source_name)
        filtered = filter_by_keywords(items, KEYWORDS)

        # 전처리
        for item in filtered:
            item["title"] = clean_text(item["title"])

        print(f"    성공: 전체 {len(items)}건 → 키워드 매칭 {len(filtered)}건")
        crawl_report["success"] += 1
        crawl_report["total_items"] += len(items)
        crawl_report["filtered_items"] += len(filtered)
        all_items.extend(filtered)

    # 결과 저장
    output_path = "report/rss_crawling_result.csv"
    if all_items:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "title", "date", "link", "matched_keyword"])
            writer.writeheader()
            writer.writerows(all_items)
        print(f"\n  결과 저장: {output_path}")

    # 실패 분석 출력
    print("\n=== RSS 크롤링 실패 분석 ===")
    print(f"  시도: {crawl_report['attempted']}개 언론사")
    print(f"  성공: {crawl_report['success']}개")
    print(f"  실패: {crawl_report['failed']}개")
    print(f"  수집: {crawl_report['filtered_items']}건 (전체 {crawl_report['total_items']}건 중)")

    if crawl_report["failures"]:
        print("\n  실패 원인:")
        for f in crawl_report["failures"]:
            print(f"    - {f['source']}: {f['error']}")

    print("\n=== 전환 결정 ===")
    print("  문제점:")
    print("    1. 다수 언론사 RSS 피드 접근 제한 (HTTP 403/404)")
    print("    2. RSS 피드 내용 불완전 (제목만 제공, 본문 없음)")
    print("    3. 언론사별 RSS 구조 상이 → 통합 파싱 어려움")
    print("    4. 업데이트 주기 불규칙 (실시간 아님)")
    print("    5. 교육부 등 주요 기관 RSS 미지원")
    print("\n  → 빅카인즈 수동 CSV 다운로드 방식으로 전환")
    print("    - 장점: 정제된 데이터, 언론사 통합 검색, 기간 설정 가능")
    print("    - 단점: 수동 작업 필요, 자동화 불가")

    return crawl_report


# ============================================================
# 빅카인즈 CSV 전처리 (실제 채택 방식)
# ============================================================

def preprocess_bigkinds_csv(input_path, output_path=None):
    """
    빅카인즈에서 다운로드한 CSV를 전처리합니다.
    - 특수문자 제거
    - 중복 헤더 제거
    - 날짜 형식 통일
    - 결측값 처리

    Parameters
    ----------
    input_path : str   빅카인즈 원본 CSV 경로
    output_path : str  전처리 결과 저장 경로 (None이면 input_path + '_clean.csv')
    """
    if output_path is None:
        output_path = input_path.replace(".csv", "_clean.csv")

    cleaned_rows = []

    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if value is None:
                    cleaned[key] = ""
                    continue
                # 특수문자 정리
                val = clean_text(str(value))
                cleaned[key] = val

            # 날짜 형식 통일 (YYYY-MM-DD)
            for date_col in ["일자", "date", "Date"]:
                if date_col in cleaned and cleaned[date_col]:
                    try:
                        for fmt in ["%Y%m%d", "%Y.%m.%d", "%Y/%m/%d", "%Y-%m-%d"]:
                            try:
                                dt = datetime.strptime(cleaned[date_col].replace(" ", ""), fmt)
                                cleaned[date_col] = dt.strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

            # 제목 결측 행 제거
            title_col = next((k for k in cleaned if "제목" in k or "title" in k.lower()), None)
            if title_col and not cleaned.get(title_col):
                continue

            cleaned_rows.append(cleaned)

    # 결과 저장
    if cleaned_rows and fieldnames:
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)

    print(f"[전처리 완료] {len(cleaned_rows)}건 저장 → {output_path}")
    return cleaned_rows, output_path


if __name__ == "__main__":
    # RSS 크롤링 시도 (실패 사례 기록)
    crawl_report, elapsed = run_rss_crawling()

    # 빅카인즈 전처리 사용 예시 (파일 있을 경우)
    print("\n=== 빅카인즈 CSV 전처리 사용법 ===")
    print("  from phase2_rss_crawling_attempt import preprocess_bigkinds_csv")
    print("  cleaned, path = preprocess_bigkinds_csv('bigkinds_raw.csv')")
