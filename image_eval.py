"""
RAG 챗봇 v4: 이미지 검색 정량 평가 (테스트셋 확장)
=====================================================
v3의 GPT-4o Vision 이미지 검색 기능을 정량 평가

평가 지표:
  - Category Accuracy  : GPT-4o가 이미지에서 추출한 카테고리가 정답과 일치하는가
  - Hit@1              : 상위 1개 결과에 정답 카테고리 상품이 있는가
  - Hit@3              : 상위 3개 결과에 정답 카테고리 상품이 있는가
  - Hit@5              : 상위 5개 결과에 정답 카테고리 상품이 있는가

테스트셋:
  - Unsplash 무료 이미지 URL 사용
  - Sneakers 6개, Boots 6개, Sandals 4개, Shoes 6개, Jackets 6개
  - 총 28개

실행 방법:
    set OPENAI_API_KEY=...
    set COHERE_API_KEY=...
    python image_eval.py
"""

import os
import requests
import pandas as pd
from multimodal_search import multimodal_product_search
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ── 설정 ──────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "shopping_rag"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
# ──────────────────────────────────────────────────


# ── 테스트셋 (Unsplash 이미지 URL, 총 28개) ────────
TEST_IMAGES = [
    # ── Sneakers (6개) ─────────────────────────────
    {
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
        "expected_category": "Sneakers",
        "description": "Nike 빨간 스니커즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=400",
        "expected_category": "Sneakers",
        "description": "흰색 캐주얼 스니커즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=400",
        "expected_category": "Sneakers",
        "description": "흰색 하이탑 스니커즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1600185365926-3a2ce3cdb9eb?w=400",
        "expected_category": "Sneakers",
        "description": "검정 스니커즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1514989940723-e8e51635b782?w=400",
        "expected_category": "Sneakers",
        "description": "컬러풀 러닝 스니커즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1556906781-9a412961a28c?w=400",
        "expected_category": "Sneakers",
        "description": "파란 스니커즈"
    },

    # ── Boots (6개) ────────────────────────────────
    {
        "url": "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?w=400",
        "expected_category": "Boots",
        "description": "갈색 하이킹 부츠"
    },
    {
        "url": "https://images.unsplash.com/photo-1605812860427-4024433a70fd?w=400",
        "expected_category": "Boots",
        "description": "검정 앵클 부츠"
    },
    {
        "url": "https://images.unsplash.com/photo-1638247025967-b4e38f787b76?w=400",
        "expected_category": "Boots",
        "description": "브라운 레더 부츠"
    },
    {
        "url": "https://images.unsplash.com/photo-1542280756-74b2f55e73ab?w=400",
        "expected_category": "Boots",
        "description": "워커 부츠"
    },
    {
        "url": "https://images.unsplash.com/photo-1573100925118-870b8efc799d?w=400",
        "expected_category": "Boots",
        "description": "여성 하이힐 부츠"
    },
    {
        "url": "https://images.unsplash.com/photo-1609709295948-17d77cb2a69b?w=400",
        "expected_category": "Boots",
        "description": "등산 트레킹 부츠"
    },

    # ── Sandals (4개) ──────────────────────────────
    {
        "url": "https://images.unsplash.com/photo-1603487742131-4160ec999306?w=400",
        "expected_category": "Sandals",
        "description": "여름 샌들"
    },
    {
        "url": "https://images.unsplash.com/photo-1562273138-f46be4ebdf33?w=400",
        "expected_category": "Sandals",
        "description": "가죽 샌들"
    },
    {
        "url": "https://images.unsplash.com/photo-1558171813-0c6e83e83e48?w=400",
        "expected_category": "Sandals",
        "description": "비치 플립플랍 샌들"
    },
    {
        "url": "https://images.unsplash.com/photo-1574180566232-aaad1b5b8450?w=400",
        "expected_category": "Sandals",
        "description": "스트랩 샌들"
    },

    # ── Shoes (6개) ────────────────────────────────
    {
        "url": "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=400",
        "expected_category": "Shoes",
        "description": "트레일 러닝화"
    },
    {
        "url": "https://images.unsplash.com/photo-1539185441755-769473a23570?w=400",
        "expected_category": "Shoes",
        "description": "캐주얼 신발"
    },
    {
        "url": "https://images.unsplash.com/photo-1511556532299-8f662fc26c06?w=400",
        "expected_category": "Shoes",
        "description": "로퍼 신발"
    },
    {
        "url": "https://images.unsplash.com/photo-1518894781549-99c8b044a0e3?w=400",
        "expected_category": "Shoes",
        "description": "남성 드레스 슈즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?w=400",
        "expected_category": "Shoes",
        "description": "여성 플랫 슈즈"
    },
    {
        "url": "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=400",
        "expected_category": "Shoes",
        "description": "러닝화"
    },

    # ── Jackets (6개) ──────────────────────────────
    {
        "url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400",
        "expected_category": "Jackets",
        "description": "아웃도어 자켓"
    },
    {
        "url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400",
        "expected_category": "Jackets",
        "description": "겨울 패딩 자켓"
    },
    {
        "url": "https://images.unsplash.com/photo-1544022613-e87ca75a784a?w=400",
        "expected_category": "Jackets",
        "description": "가죽 바이커 자켓"
    },
    {
        "url": "https://images.unsplash.com/photo-1483118714900-540cf339fd46?w=400",
        "expected_category": "Jackets",
        "description": "데님 자켓"
    },
    {
        "url": "https://images.unsplash.com/photo-1548126032-079a0fb0099d?w=400",
        "expected_category": "Jackets",
        "description": "후드 윈드브레이커"
    },
    {
        "url": "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=400",
        "expected_category": "Jackets",
        "description": "플리스 자켓"
    },
]

# 카테고리 허용 범위
CATEGORY_ALLOW = {
    "Sneakers": ["Sneakers", "Shoes"],
    "Boots":    ["Boots", "Shoes"],
    "Sandals":  ["Sandals", "Shoes"],
    "Shoes":    ["Shoes", "Sneakers", "Boots", "Sandals"],
    "Jackets":  ["Jackets", "Vests"],
}


def download_image(url: str) -> bytes:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.content


def evaluate_single(image_bytes, expected_category, vectorstore):
    query, detected_category, docs = multimodal_product_search(
        image_bytes, vectorstore, OPENAI_API_KEY, COHERE_API_KEY
    )

    allowed = CATEGORY_ALLOW.get(expected_category, [expected_category])
    category_correct = detected_category in allowed

    result_categories = [d.metadata.get("category", "") for d in docs]
    hit1 = any(c in allowed for c in result_categories[:1])
    hit3 = any(c in allowed for c in result_categories[:3])
    hit5 = any(c in allowed for c in result_categories[:5])

    return {
        "query": query,
        "detected_category": detected_category,
        "expected_category": expected_category,
        "category_correct": category_correct,
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "result_categories": result_categories,
    }


# ── 메인 ──────────────────────────────────────────
if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"벡터 수: {vectorstore._collection.count()}개")
    print(f"총 테스트 이미지: {len(TEST_IMAGES)}개")
    print("="*60)

    results = []
    for i, item in enumerate(TEST_IMAGES):
        print(f"\n[{i+1}/{len(TEST_IMAGES)}] {item['description']} (정답: {item['expected_category']})")
        try:
            image_bytes = download_image(item["url"])
            result = evaluate_single(image_bytes, item["expected_category"], vectorstore)
            result["description"] = item["description"]

            print(f"  감지 카테고리: {result['detected_category']} ({'✅' if result['category_correct'] else '❌'})")
            print(f"  생성 쿼리: {result['query']}")
            print(f"  Hit@1: {'✅' if result['hit@1'] else '❌'} | Hit@3: {'✅' if result['hit@3'] else '❌'} | Hit@5: {'✅' if result['hit@5'] else '❌'}")
            print(f"  검색 결과: {result['result_categories']}")
            results.append(result)

        except Exception as e:
            print(f"  ⚠ 오류: {e}")
            results.append({
                "description": item["description"],
                "expected_category": item["expected_category"],
                "detected_category": "ERROR",
                "category_correct": False,
                "hit@1": False,
                "hit@3": False,
                "hit@5": False,
                "query": "",
                "result_categories": [],
            })

    # ── 결과 집계 ──────────────────────────────────
    df = pd.DataFrame(results)

    category_acc = df["category_correct"].mean()
    hit1_acc     = df["hit@1"].mean()
    hit3_acc     = df["hit@3"].mean()
    hit5_acc     = df["hit@5"].mean()

    print("\n" + "="*60)
    print("이미지 검색 평가 결과 — v3 멀티모달 (테스트셋 28개)")
    print("="*60)
    print(f"  Category Accuracy : {category_acc:.4f} ({category_acc*100:.1f}%)")
    print(f"  Hit@1             : {hit1_acc:.4f} ({hit1_acc*100:.1f}%)")
    print(f"  Hit@3             : {hit3_acc:.4f} ({hit3_acc*100:.1f}%)")
    print(f"  Hit@5             : {hit5_acc:.4f} ({hit5_acc*100:.1f}%)")

    print("\n[카테고리별 결과]")
    print(df.groupby("expected_category")[["category_correct", "hit@1", "hit@3", "hit@5"]].mean().round(4))

    # ── 저장 ──────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    summary = pd.DataFrame([{
        "테스트 이미지 수": len(TEST_IMAGES),
        "Category Accuracy": round(category_acc, 4),
        "Hit@1": round(hit1_acc, 4),
        "Hit@3": round(hit3_acc, 4),
        "Hit@5": round(hit5_acc, 4),
    }])
    summary.to_csv("results/image_eval_results.csv", index=False, encoding="utf-8-sig")
    df.to_csv("results/image_eval_detail.csv", index=False, encoding="utf-8-sig")
    print("\n📊 결과 저장 완료: results/image_eval_results.csv")
    print("📊 세부 결과 저장 완료: results/image_eval_detail.csv")
    print("\n💡 이 수치를 README에 넣으세요!")
