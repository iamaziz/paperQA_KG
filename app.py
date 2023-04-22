import os
from time import sleep
from glob import glob
from functools import lru_cache
from typing import List, Tuple
from collections import namedtuple

from langchain.indexes import GraphIndexCreator  # pip install langchain
from langchain.llms import OpenAI  # pip install langchain
import pandas as pd  # pip install pandas
import streamlit as st  # pip install streamlit
from streamlit_agraph import agraph, Node, Edge, Config  # pip install streamlit-agraph
from paperqa import Docs  # pip install paper-qa
from semanticscholar import SemanticScholar  # pip install semanticscholar


os.environ["OPENAI_ORG_ID"] = "org-6kZ5XQXVZQXQXQXQXQXQXQXQXQXQXQXQXQXQXQX"


def header():
    st.set_page_config(layout="wide")
    # st.header("Q/A Chatbot & Knowledge Graph Visualizer")
    head = st.columns(2)
    st.header("Knowledge Graphs from Scientific Literature")
    st.markdown("> Build and visualize graphs graphs from research paper abstracts")
    st.sidebar.markdown(
        "> Powered by<br>**LangChain Graph** & **OpenAI GPT**", unsafe_allow_html=True
    )
    st.sidebar.image("assets/openai.jpg", width=300)
    st.sidebar.image("assets/langchain.jpg", width=300)
    st.markdown("---")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Demo** result currently constrained by **OpenAI API** model's maximum context length of 4097 tokens."
    )


class UI:
    def __init__(self):
        header()
        msg = "Search papers from literature"
        paper_topic = st.text_input(
            "Enter a topic e.g. Quantum Spintronics", placeholder=msg
        )
        if paper_topic:
            with st.spinner("Searching papers from literature"):
                papers = self._get_papers(paper_topic)
            df = pd.DataFrame(papers)
            docs = df["abstract"].tolist()
        else:
            return

        st.write(df)
        with st.expander("Show abstracts"):
            st.write(docs)

            if st.checkbox("store locally"):
                paths, bulk_path, big_string = self.store_document_locally(papers)

        # -- Apps -- #
        # if st.checkbox("Start Chatbot"):
        #     # -- PaperQA
        #     self._build_paperqa(paths)

        if st.checkbox("Build and Visualize Knowledge Graph"):
            # -- Knowledge Graph
            res = self._build_knowledge_graph(docs)
            self._render_knowledge_graph(res)

    def _build_paperqa(self, paths: List[str], papers: List[str] = None):
        # -- initialize PaperQA index
        if not hasattr(st.session_state, "index_is_built"):
            st.session_state.index_is_built = False
        if not st.session_state.index_is_built or not hasattr(
            st.session_state, "paperqa"
        ):
            st.session_state.paperqa = PaperQA()
            with st.spinner("Building index"):
                st.session_state.paperqa.build_index(tuple(paths[:3]))  # , papers[:3])
                st.session_state.index_is_built = True

        # -- ask questions
        paperqa = st.session_state.paperqa
        st.success("Ready to answer questions about the papers above")
        question = st.text_input("Ask a question about the papers")
        st.write(paperqa)
        st.write(paperqa.docs.docs)
        if question:
            question = "What type of cancer is discussed in the text?"
            answer = paperqa.ask(question)
            st.write(answer)

    @staticmethod
    @st.cache_resource
    def _build_knowledge_graph(abstracts):
        """Build knowledge graph from abstracts using LangChain Graph"""
        # -- initialize GraphIndexCreator
        index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))

        # Limiting to 42 papers for now, knowledge graph triplets is a bit intesive at the moment
        # -- This model's maximum context length is 4097 tokens
        text = " ".join([txt for txt in abstracts[:20] if txt])
        graph = index_creator.from_text(text)
        return graph

    @staticmethod
    @st.cache_resource  # (allow_output_mutation=True)
    def _get_papers(paper_topic: str) -> List[str]:
        s2 = SemanticScholarSearch(paper_topic)
        return s2.get_papers()

    def store_document_locally(self, docs: List[str]):
        if not os.path.exists("docs"):
            os.mkdir("docs")
        for doc in docs:
            year, title, abstract = doc.year, doc.title, doc.abstract
            with open(f"docs/{year}_{'_'.join(title.split())}.txt", "w") as f:
                f.write(f"{title} {abstract}")

        paths = glob("docs/*.txt")
        bulk_path = "docs/bulk.txt"
        with open(bulk_path, "w") as f:
            paper_str_list = [
                f"<START-OF-PAPER>{doc.title} {doc.abstract}<END-OF-PAPER>"
                for doc in docs
            ]
            big_string = " ".join(paper_str_list)
            f.write(big_string)

        return paths, bulk_path, big_string

    def _render_knowledge_graph(self, graph):
        """Render knowledge graph using streamlit-agraph
        see: https://github.com/ChrisDelClea/streamlit-agraph#example-app
        """
        triplets = graph.get_triples()
        # TODO: save the graph https://python.langchain.com/en/latest/modules/chains/index_examples/graph_qa.html#save-the-graph

        st.write(f"> Number of the extracted triplets: {len(triplets)}")
        df = pd.DataFrame(triplets, columns=["subject", "object", "predicate"])
        arrange_cols = ["subject", "predicate", "object"]
        df = df[arrange_cols]
        with st.expander("View raw triplets"):
            st.write(triplets)
        with st.expander("View triplets as dataframe"):
            st.write(df)

        # -- visualize graph

        nodes = []
        edges = []
        seen = set()
        for triple in triplets:
            h, t, r = triple
            if not h in seen:
                nodes.append(Node(id=h, label=h, size=10, shape="star"))
                seen.add(h)
            if not t in seen:
                nodes.append(Node(id=t, label=t, size=10, shape="diamond"))
                seen.add(t)

            edges.append(Edge(source=h, target=t, label=r, size=5))

        config = Config(width=1200, height=1200, directed=True, physics=True)

        with st.expander("View knowledge graph"):
            graph = agraph(nodes=nodes, edges=edges, config=config)


class SemanticScholarSearch:
    Paper = namedtuple("Paper", ["title", "abstract", "year", "url", "authors"])

    def __init__(self, paper_topic: str, limit: int = 100):
        self.sch = SemanticScholar()
        self.response = self.search_papers(paper_topic, limit=limit)

    def get_papers(self):
        with st.spinner("Parsing papers"):
            return self.parse_response(self.response)

    @lru_cache(maxsize=32)
    def search_papers(self, paper_topic: str, limit: int = 100) -> List[str]:
        return self.sch.search_paper(paper_topic, limit=limit)

    @lru_cache(maxsize=32)
    def parse_response(self, response: List[str]) -> List[Paper]:
        papers = []
        for i in range(len(response)):
            papers.append(
                self.Paper(
                    title=response[i]["title"],
                    abstract=response[i]["abstract"],
                    year=response[i]["year"],
                    url=response[i]["url"],
                    authors=", ".join([a["name"] for a in response[i]["authors"]]),
                )
            )
        return papers

    def search(self, query: str) -> List[str]:
        return self.docs.search(query)


class PaperQA:
    """add a docstring here with URL to repo"""

    def __init__(self):  # , my_docs: List[str]):
        self.docs = Docs()
        # self.build_index(my_docs)

    @lru_cache(maxsize=32)
    def build_index(self, my_docs_paths: Tuple[str]):  # , citation: List[str] = None):
        for path in my_docs_paths:  # ): #, my_docs):
            sleep(5)
            self.docs.add(path)  # , citation=doc.authors + " " + doc.year)

    @lru_cache(maxsize=32)
    def ask(self, question: str) -> List[str]:
        return self.docs.query(question)


if __name__ == "__main__":
    UI()
