# backend/agents/document_agent.py

import os
import logging
from enum import Enum
from typing import Dict, Literal, TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END

from backend.agents.rag_api.summarize import summarize_task
from backend.agents.rag_api.query import query_task
from backend.utils.groq_client import groq_client
from backend.utils.file_uploader import upload_single_file

logger = logging.getLogger(__name__)


class DocumentTask(str, Enum):
    summarize = "summarize"
    compare = "compare"
    query = "query"


class DocumentAgentState(TypedDict):
    input: str
    chat_id: Optional[str]
    doc_id: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    task: Optional[DocumentTask]
    response: Optional[str]


class DocumentAgent:
    def __init__(self):
        self.graph: Any = self._build_graph()  # CompiledGraph is not exposed directly

    async def _compare_task(self, state: DocumentAgentState) -> DocumentAgentState:
        uploads_dir = "uploads"
        all_files = sorted([
            os.path.join(uploads_dir, f)
            for f in os.listdir(uploads_dir)
            if f.lower().endswith(".pdf")
        ], key=os.path.getmtime, reverse=True)

        if len(all_files) < 2:
            return {**state, "response": "❌ Not enough PDF files to compare. Upload at least two."}

        try:
            file1 = upload_single_file(all_files[0])
            file2 = upload_single_file(all_files[1])

            response = f"✅ Uploaded `{file1}` and `{file2}` for comparison.\n(Stub: Implement comparison logic)"
            return {**state, "response": response}

        except Exception as e:
            logger.error(f"[❌ CompareTask Error]: {e}")
            return {**state, "response": "❌ Failed to upload or compare the documents."}

    async def _router_node(self, state: DocumentAgentState) -> DocumentAgentState:
        prompt = state.get("input", "")
        history = state.get("chat_history", [])

        try:
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
            route_prompt = (
                "You are a routing assistant inside the document agent.\n"
                "Your job is to choose a task based on context:\n"
                "- 'summarize' → summarize uploaded document\n"
                "- 'compare' → compare two uploaded documents\n"
                "- 'query' → answer questions based on uploaded document\n"
                "Respond ONLY with one of: summarize, compare, query.\n\n"
                f"History:\n{context}\n\nUser:\n{prompt}"
            )

            messages = [
                {"role": "system", "content": route_prompt},
                {"role": "user", "content": prompt}
            ]

            result = groq_client.client.chat.completions.create(
                model=groq_client.model,
                messages=messages,
                temperature=0,
                max_tokens=5,
            )
            task = result.choices[0].message.content.strip().lower()

            if task not in {"summarize", "compare", "query"}:
                logger.warning(f"[⚠️ Invalid Task Returned]: {task}, defaulting to 'query'")
                task = "query"

            logger.info(f"[📄 DocumentAgent Routed To]: {task}")
            return {**state, "task": task}

        except Exception as e:
            logger.error(f"[❌ Router Error]: {e}")
            return {**state, "task": "query"}

    def _build_graph(self) -> Any:
        graph = StateGraph(DocumentAgentState)

        graph.add_node("router", self._router_node)
        graph.add_node("summarize", summarize_task)
        graph.add_node("compare", self._compare_task)
        graph.add_node("query", query_task)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            lambda state: state["task"],
            {
                "summarize": "summarize",
                "compare": "compare",
                "query": "query"
            }
        )

        graph.add_edge("summarize", END)
        graph.add_edge("compare", END)
        graph.add_edge("query", END)

        return graph.compile()

    def get_graph(self) -> Any:
        return self.graph

    async def run(self, input: DocumentAgentState) -> DocumentAgentState:
        return await self.graph.ainvoke(input)
