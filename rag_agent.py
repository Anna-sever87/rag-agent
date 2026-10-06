"""RAG-агент: Pinecone + эмбеддинги OpenAI + агент LangChain.

Умеет:
1. подключаться к Pinecone и искать ответы в векторной базе;
2. читать страницу по ссылке: парсинг -> чанки -> эмбеддинги -> сохранение в базу;
3. делать реальный GET-запрос к открытому API (случайный факт о котах).

При запуске файла как модуля выполняется тестовая проверка всех трёх частей.
"""
import hashlib
import os
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

OPENAI_BASE_URL = "https://api.openai.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = 60
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (rag-agent learning project)"}

SYSTEM_PROMPT = """Ты умный помощник с доступом к базе знаний. Отвечай кратко, на русском языке.
Правила:
1. Если вопрос про содержимое сохранённых страниц или документов, сначала вызови
   инструмент search_knowledge_base и отвечай только по найденному тексту.
2. Если пользователь прислал ссылку и просит прочитать или добавить её,
   вызови инструмент add_url_to_knowledge_base.
3. Если просят случайный факт о котах, вызови инструмент get_random_cat_fact
   и переведи факт на русский язык.
4. Если в базе ничего не найдено, прямо скажи об этом. Ничего не выдумывай."""


def fetch_page_text(url: str) -> str:
    """Скачивает страницу и возвращает её чистый текст."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    if len(text) < 200:
        text = soup.get_text(" ", strip=True)
    return text


def split_into_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Делит текст на куски (чанки) с небольшим нахлёстом."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def create_api_request_tool():
    """Инструмент для запроса к внешнему API: реальный GET-запрос, а не заглушка."""

    @tool
    def get_random_cat_fact() -> str:
        """Возвращает случайный факт о котах из открытого API (на английском языке)."""
        sources = [
            ("https://catfact.ninja/fact", lambda d: d["fact"]),
            ("https://meowfacts.herokuapp.com/", lambda d: d["data"][0]),
        ]
        for url, extract in sources:
            try:
                response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
                response.raise_for_status()
                fact = extract(response.json())
                print(f"[API] GET {url} -> {response.status_code}")
                return fact
            except (requests.RequestException, KeyError, IndexError, ValueError) as error:
                print(f"[API] Не получилось обратиться к {url}: {error}")
        return "Не удалось получить факт: API сейчас недоступен."

    return get_random_cat_fact


class RAGAgent:
    """RAG-агент: хранит знания в Pinecone и отвечает на вопросы по ним."""

    def __init__(self) -> None:
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not openai_key or not pinecone_key:
            raise SystemExit("Не найдены OPENAI_API_KEY или PINECONE_API_KEY. Проверь файл .env")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-agent")

        self.openai = OpenAI(api_key=openai_key, base_url=OPENAI_BASE_URL)
        self.pinecone = Pinecone(api_key=pinecone_key)
        self.index = self._connect_index()

        model = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=openai_key,
            base_url=OPENAI_BASE_URL,
            temperature=0,
            max_tokens=1500,
        )
        self.agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=self._build_tools(),
            checkpointer=InMemorySaver(),
        )

    # ---------- Pinecone ----------
    def _connect_index(self):
        """Подключается к индексу Pinecone, при необходимости создаёт его."""
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Создаю индекс {self.index_name} ...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pinecone.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        return self.pinecone.Index(self.index_name)

    def count_vectors(self) -> int:
        """Сколько векторов сейчас лежит в индексе."""
        return self.index.describe_index_stats().total_vector_count

    # ---------- эмбеддинги и поиск ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Превращает список текстов в список векторов."""
        response = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """Ищет в базе чанки, похожие на вопрос."""
        vector = self.embed([question])[0]
        result = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {
                "score": round(match.score, 4),
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("source", ""),
            }
            for match in result.matches
        ]

    # ---------- работа со страницей по ссылке ----------
    def add_url(self, url: str) -> int:
        """Парсит страницу, делит на чанки, делает эмбеддинги и сохраняет в Pinecone."""
        text = fetch_page_text(url)
        chunks = split_into_chunks(text)[:MAX_CHUNKS]
        if not chunks:
            raise ValueError("На странице не нашлось текста.")
        print(f"Страница прочитана: {len(text)} символов, чанков: {len(chunks)}")

        vectors = self.embed(chunks)
        url_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        records = [
            {
                "id": f"{url_id}-{i}",
                "values": vector,
                "metadata": {"text": chunk, "source": url},
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        for start in range(0, len(records), 50):
            self.index.upsert(vectors=records[start:start + 50])
        time.sleep(3)  # Pinecone нужно пару секунд, чтобы новые векторы стали доступны для поиска
        print(f"Сохранено в Pinecone чанков: {len(records)}")
        return len(records)

    # ---------- агент ----------
    def _build_tools(self) -> list:
        rag = self

        @tool
        def search_knowledge_base(question: str) -> str:
            """Ищет ответ на вопрос в базе знаний (сохранённые страницы и документы)."""
            matches = rag.search(question)
            if not matches:
                return "В базе знаний ничего не найдено."
            return "\n\n".join(f"[{m['source']}] {m['text']}" for m in matches)

        @tool
        def add_url_to_knowledge_base(url: str) -> str:
            """Читает страницу по ссылке и добавляет её в базу знаний."""
            try:
                count = rag.add_url(url)
            except (requests.RequestException, ValueError) as error:
                return f"Не удалось добавить страницу: {error}"
            return f"Страница добавлена в базу знаний, чанков: {count}."

        return [search_knowledge_base, add_url_to_knowledge_base, create_api_request_tool()]

    def chat(self, question: str, thread_id: str = "default") -> str:
        """Задаёт вопрос агенту и возвращает текст ответа."""
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content


if __name__ == "__main__":
    print("=== Тест 1: подключение к Pinecone ===")
    rag_agent = RAGAgent()
    print(f"Индекс: {rag_agent.index_name}, векторов в базе: {rag_agent.count_vectors()}")
    test_matches = rag_agent.search("тестовый запрос для проверки подключения")
    print(f"Тестовый поиск выполнен, найдено совпадений: {len(test_matches)}")

    print("\n=== Тест 2: чтение страницы по ссылке ===")
    rag_agent.add_url("https://ru.wikipedia.org/wiki/Земля")
    print(f"Векторов в базе теперь: {rag_agent.count_vectors()}")

    print("\n=== Тест 3: вопросы агенту ===")
    q1 = "Что на сохранённой странице сказано о форме Земли?"
    print(f"Вопрос: {q1}\nОтвет: {rag_agent.chat(q1, 'test')}")
    q2 = "Расскажи случайный факт о котах."
    print(f"\nВопрос: {q2}\nОтвет: {rag_agent.chat(q2, 'test')}")