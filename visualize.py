"""
RAG 챗봇 v4: 평가 결과 시각화
================================
ragas_eval, image_eval 결과를 차트로 시각화

생성 차트:
  1. RAGAs 전체 결과 막대 차트
  2. RAGAs 질문 유형별 히트맵
  3. 이미지 검색 카테고리별 막대 차트
  4. 이미지 검색 Hit@K 차트

실행 방법:
    python visualize.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("results", exist_ok=True)
os.makedirs("images", exist_ok=True)


# ── 1. RAGAs 전체 결과 막대 차트 ──────────────────
def plot_ragas_summary():
    metrics = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    scores  = [0.8507, 0.3879, 0.7534, 0.8750]
    colors  = ["#4A90D9" if s >= 0.7 else "#E8A838" if s >= 0.5 else "#E05C5C" for s in scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metrics, scores, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{score:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("RAGAs 평가 결과 — v3 챗봇 (GPT-4o-mini + Cohere Rerank + 소스필터링)", fontsize=13, fontweight="bold", pad=15)
    ax.axhline(y=0.7, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(3.6, 0.71, "기준선 0.7", fontsize=9, color="gray")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("images/ragas_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ images/ragas_summary.png 저장 완료")


# ── 2. RAGAs 질문 유형별 히트맵 ───────────────────
def plot_ragas_heatmap():
    data = {
        "Faithfulness":      {"faq": 1.0000, "product": 0.7222, "review": 0.6667},
        "Answer Relevancy":  {"faq": 0.5004, "product": 0.1883, "review": 0.4203},
        "Context Precision": {"faq": 0.8437, "product": 0.5833, "review": 0.7958},
        "Context Recall":    {"faq": 0.8750, "product": 0.8000, "review": 1.0000},
    }

    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(df.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=12)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=11)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("RAGAs 질문 유형별 성능 히트맵", fontsize=13, fontweight="bold", pad=15)
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("images/ragas_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ images/ragas_heatmap.png 저장 완료")


# ── 3. 이미지 검색 카테고리별 막대 차트 ──────────
def plot_image_category():
    categories = ["Boots", "Jackets", "Sandals", "Shoes", "Sneakers"]
    scores     = [0.6667, 0.8333, 0.5000, 0.6667, 0.8333]
    colors     = ["#4A90D9" if s >= 0.7 else "#E8A838" if s >= 0.5 else "#E05C5C" for s in scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(categories, scores, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{score*100:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Hit@5 Score", fontsize=12)
    ax.set_title("이미지 검색 카테고리별 성능 (테스트셋 28개)", fontsize=13, fontweight="bold", pad=15)
    ax.axhline(y=0.7143, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(4.6, 0.725, f"평균 71.4%", fontsize=9, color="gray")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    blue_patch  = mpatches.Patch(color="#4A90D9", label="≥ 70%")
    amber_patch = mpatches.Patch(color="#E8A838", label="50~70%")
    red_patch   = mpatches.Patch(color="#E05C5C", label="< 50%")
    ax.legend(handles=[blue_patch, amber_patch, red_patch], fontsize=10)

    plt.tight_layout()
    plt.savefig("images/image_eval_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ images/image_eval_category.png 저장 완료")


# ── 4. 이미지 검색 Hit@K 차트 ────────────────────
def plot_hitk():
    labels = ["Category\nAccuracy", "Hit@1", "Hit@3", "Hit@5"]
    scores = [0.7143, 0.7143, 0.7143, 0.7143]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color="#4A90D9", width=0.4, edgecolor="white", linewidth=1.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{score*100:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("이미지 검색 Hit@K 평가 결과 — v3 멀티모달", fontsize=13, fontweight="bold", pad=15)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("images/image_eval_hitk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ images/image_eval_hitk.png 저장 완료")


# ── 메인 ──────────────────────────────────────────
if __name__ == "__main__":
    print("📊 시각화 시작...\n")
    plot_ragas_summary()
    plot_ragas_heatmap()
    plot_image_category()
    plot_hitk()
    print("\n✅ 모든 차트 저장 완료! → images/ 폴더 확인")
