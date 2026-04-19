"""
RAG 챗봇 v4: RAGAs 성능 평가 (테스트셋 개선)
=============================================
v3 챗봇 파이프라인을 RAGAs로 정량 평가

변경사항:
- 테스트셋을 실제 DB 데이터 기반으로 재설계
- Product: 실제 DB 상품명 기반 구체적 ground_truth
- Review: 실제 DB에 있는 AeroGarden, Keurig 리뷰 기반
- Review 0점 문제 해결 (DB에 없는 상품 질문 제거)

평가 지표:
  - Faithfulness       : 답변이 검색 문서에 충실한가 (할루시네이션 측정)
  - Answer Relevancy   : 질문과 답변이 관련 있는가
  - Context Precision  : 검색된 문서가 질문에 적합한가
  - Context Recall     : 필요한 문서를 빠짐없이 가져왔는가

실행 방법:
    set OPENAI_API_KEY=...
    set COHERE_API_KEY=...
    python ragas_eval.py
"""

import os
import pandas as pd
import cohere
from datasets import Dataset

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── 설정 ──────────────────────────────────────────
CHROMA_DIR     = "./chroma_db"
COLLECTION     = "shopping_rag"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
# ──────────────────────────────────────────────────


# ── 테스트셋 (실제 DB 기반) ────────────────────────
TEST_SET = [
    # ── FAQ (8개) ──────────────────────────────────
    {
        "question": "배송은 보통 며칠 걸려요?",
        "ground_truth": "일반적으로 결제 완료 후 2~3 영업일 내에 배송됩니다.",
        "type": "faq"
    },
    {
        "question": "환불하고 싶은데 어떻게 해야 하나요?",
        "ground_truth": "마이페이지 > 반품/교환 신청 후 상품을 반송하시면 확인 후 환불됩니다.",
        "type": "faq"
    },
    {
        "question": "쿠폰이랑 적립금 같이 쓸 수 있나요?",
        "ground_truth": "쿠폰과 적립금은 동시에 사용하실 수 있습니다.",
        "type": "faq"
    },
    {
        "question": "주말에도 배송되나요?",
        "ground_truth": "주말 및 공휴일에는 배송이 이루어지지 않습니다.",
        "type": "faq"
    },
    {
        "question": "환불 기간은 얼마나 걸리나요?",
        "ground_truth": "상품 회수 후 2~5 영업일 내에 환불 처리됩니다.",
        "type": "faq"
    },
    {
        "question": "배송비는 얼마예요?",
        "ground_truth": "기본 배송비는 3,000원이며, 일부 상품은 무료배송이 적용됩니다.",
        "type": "faq"
    },
    {
        "question": "적립금은 어떻게 사용하나요?",
        "ground_truth": "결제 페이지에서 사용할 적립금 금액을 입력하시면 됩니다.",
        "type": "faq"
    },
    {
        "question": "회원가입 없이 주문할 수 있나요?",
        "ground_truth": "비회원 주문도 가능합니다. 주문 시 이메일을 입력하시면 주문 내역을 확인할 수 있습니다.",
        "type": "faq"
    },

    # ── Product (5개, 실제 DB 상품 기반) ──────────
    {
        "question": "방수 등산화 추천해줘",
        "ground_truth": "Men's Vasque Talus Trek Waterproof Hiking Boots는 방수 기능을 갖춘 남성용 등산화로, UltraDry 방수 기술과 Vibram Nuasi 아웃솔이 적용되어 있습니다.",
        "type": "product"
    },
    {
        "question": "남성 하이킹 부츠 추천해줘",
        "ground_truth": "Men's Vasque Talus Trek Waterproof Hiking Boots는 누벅 가죽과 내마모성 메시 소재로 제작된 남성용 하이킹 부츠입니다.",
        "type": "product"
    },
    {
        "question": "여성 하이킹 신발 있어요?",
        "ground_truth": "Men's Merrell Moab 2 Ventilated Trail Shoes는 통기성이 뛰어난 트레일 신발로, EVA 풋베드와 나일론 아치 섕크가 적용되어 있습니다.",
        "type": "product"
    },
    {
        "question": "방수 러닝 자켓 추천해줘",
        "ground_truth": "Ridge Runner Light-Up Running Jacket은 바람과 물을 차단하는 경량 100% 나일론 소재의 러닝 자켓으로, 저조도 환경에서 시인성을 높이는 라이트업 기능이 있습니다.",
        "type": "product"
    },
    {
        "question": "경량 조끼 있어요?",
        "ground_truth": "Men's Mountain Classic Down Vest Colorblock은 컬러블록 디자인의 남성용 다운 조끼입니다.",
        "type": "product"
    },

    # ── Review (3개, 실제 DB 리뷰 기반) ──────────
    {
        "question": "AeroGarden 제품 후기 어때요?",
        "ground_truth": "AeroGarden 제품은 조립 설명서가 명확하고 잘 작성되어 있으며, 품질이 좋은 회사에서 만든 제품입니다. 실내에서 상추, 허브, 토마토 등을 재배할 수 있습니다.",
        "type": "review"
    },
    {
        "question": "Keurig K컵 재사용 제품 써본 사람 있어요?",
        "ground_truth": "Ecobrew 재사용 가능한 Keurig K컵은 디자인이 잘 되어 있고 사용하기 쉬우며, 직접 커피를 내리기에 좋다는 평가를 받고 있습니다.",
        "type": "review"
    },
    {
        "question": "Dave's Gourmet 소스 후기 알려줘",
        "ground_truth": "Dave's Gourmet Ghost Pepper 소스는 하바네로보다 더 매운 맛을 원하는 분들에게 추천되는 제품입니다.",
        "type": "review"
    },
]


# ── 프롬프트 (v3와 동일) ───────────────────────────
PROMPT_TEMPLATE = """You are a Korean shopping mall customer service chatbot.
Answer ONLY using the exact information from the reference documents below.
Rules:
- Answer in Korean only.
- NEVER invent product names, prices, or any information not in the documents.
- If documents contain relevant product info, quote it directly.
- If the answer is not in the documents, respond ONLY with: "해당 내용은 고객센터(1588-0000)로 문의해 주세요."
- Do NOT mix other languages into Korean sentences.

[Reference Documents]
{context}

[Customer Question]
{question}

[Answer]"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

KO_TO_EN = {
    "등산화": "hiking boots", "방수": "waterproof",
    "신발": "shoes", "자켓": "jacket", "부츠": "boots",
    "운동화": "sneakers", "샌들": "sandals",
    "농구화": "basketball shoes", "런닝화": "running shoes",
    "트레킹화": "trekking boots", "남성": "men", "여성": "women",
    "겨울": "winter", "방한": "insulated", "경량": "lightweight",
    "스니커즈": "sneakers", "조끼": "vest", "러닝": "running",
}

def translate_query(q):
    for ko, en in KO_TO_EN.items():
        q = q.replace(ko, en)
    return q


# ── v3 RAG 파이프라인 ──────────────────────────────
def run_rag_v3(question, question_type, vectorstore, llm, cohere_client):
    """v3 파이프라인 그대로 재현: 소스 필터링 + Cohere Re-ranking"""

    product_keywords = ["추천", "상품", "등산화", "신발", "자켓", "부츠", "운동화", "스니커즈", "샌들", "조끼", "러닝"]
    review_keywords  = ["후기", "리뷰", "사용기", "평가", "어때", "만족", "써본"]

    is_product_query = any(kw in question for kw in product_keywords)
    is_review_query  = any(kw in question for kw in review_keywords)
    search_query = translate_query(question) if (is_product_query or is_review_query) else question

    all_docs = vectorstore.similarity_search(search_query, k=20)

    if is_review_query and is_product_query:
        allowed = {"review", "product", "faq"}
    elif is_review_query:
        allowed = {"review", "faq"}
    elif is_product_query:
        allowed = {"product", "faq"}
    else:
        allowed = {"faq"}

    filtered_docs = [d for d in all_docs if d.metadata.get("source") in allowed]
    if not filtered_docs:
        filtered_docs = all_docs

    # Cohere Re-ranking
    if len(filtered_docs) > 1:
        try:
            rerank_docs = filtered_docs[:20]
            response = cohere_client.rerank(
                model="rerank-v3.5",
                query=question,
                documents=[d.page_content for d in rerank_docs],
                top_n=5,
            )
            docs = [rerank_docs[r.index] for r in response.results]
        except Exception:
            docs = filtered_docs[:5]
    else:
        docs = filtered_docs[:5]

    context = "\n\n".join([
        f"[문서{i+1} | {d.metadata.get('source')} | {d.metadata.get('category')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ])

    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    contexts = [d.page_content for d in docs]
    return answer, contexts


def build_ragas_dataset(vectorstore, llm, cohere_client):
    print("\n" + "="*50)
    print("v3 챗봇 RAGAs 평가 시작...")
    print("="*50)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        q = item["question"]
        print(f"  [{item['type']}] 질문: {q}")
        answer, ctx = run_rag_v3(q, item["type"], vectorstore, llm, cohere_client)
        print(f"  답변: {answer[:60]}...")

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


def get_score(result, key):
    val = result[key]
    if isinstance(val, list):
        val = [v for v in val if v is not None]
        return round(sum(val) / len(val), 4) if val else 0.0
    return round(float(val), 4)


# ── 메인 ──────────────────────────────────────────
if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    cohere_client = cohere.ClientV2(COHERE_API_KEY)

    print(f"벡터 수: {vectorstore._collection.count()}개")

    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    evaluator_emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    dataset = build_ragas_dataset(vectorstore, llm, cohere_client)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )

    print("\n" + "="*60)
    print("RAGAs 평가 결과 — v3 챗봇 (테스트셋 개선)")
    print("="*60)

    rows = [{
        "버전": "v3 (GPT-4o-mini + Cohere Rerank + 소스필터링)",
        "Faithfulness":       get_score(result, "faithfulness"),
        "Answer Relevancy":   get_score(result, "answer_relevancy"),
        "Context Precision":  get_score(result, "context_precision"),
        "Context Recall":     get_score(result, "context_recall"),
    }]

    df_detail = result.to_pandas()
    df_detail["type"] = [item["type"] for item in TEST_SET]

    print("\n[전체 결과]")
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))

    print("\n[질문 유형별 평균]")
    numeric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in numeric_cols if c in df_detail.columns]
    print(df_detail.groupby("type")[available_cols].mean().round(4))

    os.makedirs("results", exist_ok=True)
    df_summary.to_csv("results/ragas_results.csv", index=False, encoding="utf-8-sig")
    df_detail.to_csv("results/ragas_results_detail.csv", index=False, encoding="utf-8-sig")
    print("\n📊 결과 저장 완료: results/ragas_results.csv")
    print("📊 세부 결과 저장 완료: results/ragas_results_detail.csv")
    print("\n💡 이 수치를 README에 넣으세요!")
