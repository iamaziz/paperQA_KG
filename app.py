import os
from time import sleep
from glob import glob
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple
from collections import namedtuple

from langchain.indexes import GraphIndexCreator  # pip install langchain
from langchain.llms import OpenAI  # pip install langchain
import pandas as pd  # pip install pandas
import numpy as np  # pip install numpy
import streamlit as st  # pip install streamlit
from streamlit_agraph import agraph, Node, Edge, Config  # pip install streamlit-agraph
from paperqa import Docs  # pip install paper-qa
from semanticscholar import SemanticScholar  # pip install semanticscholar



Paper = namedtuple("Paper", ["title", "abstract", "year", "url", "authors"])

def header():
    st.set_page_config(layout="wide")
    st.sidebar.caption("Demo | 2023")
    st.header("Scientific Literature _Distilled_ ⚗️", anchor="top")

    st.markdown("#### <ins>**3 in 1**</ins>: _Distlling scientific literature abstracts about a specific topic_", unsafe_allow_html=True)
    subheader = """
    #####  through **_`Slide Deck Generator`_** | **_`PaperQA chatbot`_** | **_`Knowelge Graphs Builder/Visualizer`_**
    
    """
    st.markdown(subheader, unsafe_allow_html=True)
    # st.markdown(subheader, unsafe_allow_html=True)

    # st.sidebar.image("assets/graph.jpg", width=50)
    st.sidebar.caption("# Powered by")
    col = st.sidebar.columns(2)
    with col[0]:
        st.sidebar.markdown("**SemanticScholar** | **LangChain Graph** | **OpenAI GPT**")
    with col[0]:
        st.sidebar.image("assets/openai.jpg", width=150)
    with col[0]:
        st.sidebar.image("assets/langchain.jpg", width=150)
    with col[0]:
        st.sidebar.image("assets/semanticscholar.png", width=150)
    st.markdown("---")
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input("Enter OpenAI API key", type="password")
    st.sidebar.info(
        "**Demo** result currently constrained by **API**'s maximum context length of 8192 tokens."
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


class UI:
    def __init__(self):
        header()
        msg_placeholder = "Search papers from literature"
        sample_topics = [
            "AI Safety and Alignment",
            "AI Ethics",
            "Quantum Spintronics",
            "protein structure folding",
            "economy recession causes",
            "LLMs evaluation methods",
            "Generative AI applications in medical health",
            "Financial services cybersecurity for quantum computing",
            "economy recession reasons",
            "medical treatment for alzheimer's",
            "Risk of global recession in 2023",
            "Trusted AI",
            "AI Governance",
            "Transformer architecture and LLMs"
        ]
        str_topics = " `or` ".join([f"**_{s}_**" for s in sample_topics[:6]])
        msg_header = f"Search any topic `e.g.` {str_topics}, ... etc."
        st.caption(msg_header)
        col = st.columns([2, 1, 1])
        with col[0]:
            paper_topic = st.text_input("Enter a topic", placeholder=msg_placeholder)
        with col[1]:
            from_year = st.selectbox("From year", options=[i for i in range(1950, datetime.now().year + 1)], index=67)
        with col[2]:
            to_year = st.selectbox("To year", options=[i for i in range(1950, datetime.now().year + 1)], index=73)
        assert from_year <= to_year, (st.warning("From year must be less than or equal to To year"), st.stop())
        if not paper_topic or not hasattr(st.session_state, "paper_topic"):
            picker = st.button("Pick a random topic")  # , on_click=st.snow)
            if picker:
                paper_topic = np.random.choice(sample_topics)
                st.session_state.paper_topic = paper_topic
        else:
            st.session_state.paper_topic = paper_topic

        if "paper_topic" in st.session_state:
            paper_topic = st.session_state.paper_topic

        if paper_topic:
            st.markdown(f"_Topic_: <h3>{paper_topic}</h3>", unsafe_allow_html=True)
            with st.spinner("Searching papers from literature (SemanticScholar API)"):
                papers = self._get_papers(paper_topic, from_year, to_year)
            df = pd.DataFrame(papers)
            docs, urls = df["abstract"].tolist(), df["url"].tolist()
        else:
            return

        st.markdown("---")
        with st.expander(f"View results: {df.shape[0]} papers"):
            st.write(f"`{df.shape}`", df)
            # -- download dataframe
            cols = st.columns([4, 1])
            cols[1].download_button("Download abstracts", df.to_csv(), file_name=f"REF-{paper_topic}.csv", mime="text/csv")

        with st.expander("Show abstracts"):
            st.write(docs)
        
        st.markdown("---")

        # -- Apps -- #
        tabs = st.tabs(["_**QA Chatbot**_", "**Knowledge Graph** _(graphs generator)_", "**Deckify** _(slides generator)_"])

        with tabs[0]:
            if st.checkbox("Start Chatbot"):
                paths, bulk_path, big_string = self.store_document_locally(papers)
                # -- PaperQA
                paperqa = self._build_paperqa(paths[:5])
                self._chatbot(paperqa)

        with tabs[1]:
            if st.checkbox("Build and Visualize Knowledge Graph"):
                # -- Knowledge Graph
                try:
                    res = self._build_knowledge_graph(docs)
                except Exception as e:
                    try:
                        res = self._build_knowledge_graph(docs, max_tokens=5000, max_papers=10)
                    except Exception as e:
                        res = self._build_knowledge_graph(docs, max_tokens=1000, max_papers=5)
                self._render_knowledge_graph(res)
            
        with tabs[2]:
            title = "> Build slides for a presentation about **'" + paper_topic + "'** based on **top scientific papers**. Powered by OpenAI GPT."
            st.markdown(title, unsafe_allow_html=True)
            if st.checkbox("Deckify (_verb_ to mean generating Slide Deck)"):
                
                # -- Deckify (slides generator)
                from deckify import BuildSlides
                MAX_ABSTRACTS = 13
                docs = docs[:MAX_ABSTRACTS] # limiting to 13 abstracts for now to avoid exceeding the API's maximum context length of 8192 tokens
                st.session_state.selected_model = "gpt-3.5-turbo-0301"
                builder = BuildSlides(docs, urls, paper_topic)
                st.session_state.model = builder.model
                prompt = builder.build_prompt(tuple(docs), urls=urls, paper_topic=paper_topic)


                # -- view and select GPT model to use
                with st.expander("View GPT models and select one"):
                    available_models = builder.model.list_models()['data']
                    model_names = [model['id'] for model in available_models]
                    selected_model = st.selectbox("Select GPT model", options=[st.session_state.selected_model] + model_names)
                    model_details = [model for model in available_models if model['id'] == selected_model][0]
                    st.write(model_details)
                    if selected_model != st.session_state.selected_model:
                        st.session_state.selected_model = selected_model
                        from gpt import OpenAIService
                        builder.model = OpenAIService(model_name=selected_model)
                        builder.model_name = selected_model
                        st.session_state.model = builder.model
                    # st.write(available_models)# [selected_model])

                # -- deckify
                with st.expander("View prompt"):
                    st.code(prompt) #, language="text")
                with st.spinner("Deckifying"):
                    res, time_taken = builder.deckify(prompt)
                    res = res.choices[0].message.content
                
                
                with st.expander("View deckified result"):
                    if st.checkbox("raw"):
                        st.code(res)
                    else:
                        st.write(res)
                # st.markdown(f"> <sup>Time to deckify: {time_taken:.2f} seconds</sup>", unsafe_allow_html=True)
                msg = f"Created in `{time_taken:.2f}` seconds"
                st.markdown(f"> <sup>{msg}</sup>", unsafe_allow_html=True)

                # -- slides string formatting and saving
                markdown = f"""% {paper_topic}
% Scientific Literature Distilled | _{datetime.now().date()}_
% Covering literature between {from_year} and {to_year} <br> **By** `{builder.model_name}` <br> **Generated in** `{time_taken:.2f}`seconds
"""
                markdown += res
                markdown_path = "slides.md"
                slides_path = "index.html"
                with open(markdown_path, "w") as f:
                    f.write(markdown)
                
                # -- convert to slides
                # themes: https://revealjs.com/themes/
                SLIDES_THEMES = ["serif", "simple", "sky", "beige", "blood", "moon", "night", "solarized", "white", "league", "dracula", "black"]
                theme = st.selectbox("Select slides theme", options=SLIDES_THEMES, index=0)

                # src: https://gist.github.com/jsoma/629b9564af5b1e7fa62d0a3a0a47c296
                cmd_str = f"pandoc -t revealjs -s -o {slides_path} {markdown_path} -V revealjs-url=https://unpkg.com/reveal.js/ -V theme={theme} --include-in-header=assets/slides.css "
                import subprocess
                subprocess.run(cmd_str, shell=True)
                
                # -- render slides
                st.markdown(f"> {paper_topic}", unsafe_allow_html=True)
                import streamlit.components.v1 as components
                with open(slides_path, "r") as f:
                    html = f.read()
                components.html(html, width=950, height=769, scrolling=False)
                

                # -- download slides
                with open(slides_path, "r") as f:
                    st.download_button("Download slides", f, file_name=f"{paper_topic}.html", mime="text/html")


                ##########################################
                # CHATBOT assistant for a QA on the slides

                # -- chatbot with GPT (APPENDIX start here)
                if not hasattr(st.session_state, "chat_history"):
                    st.session_state.chat_history = {}
                
                with st.expander(f"Chat with GPT about `{paper_topic}` within the context of the papers above"):
                    # display chat history with st_chat as a chatbot-like conversation
                    from streamlit_chat import message

                    history_questions = list(st.session_state.chat_history.keys())
                    for q in history_questions:                    
                        message(st.session_state.chat_history[q]["user"], is_user=True, key=f"user:{q}")
                        message(st.session_state.chat_history[q]["bot"], is_user=False, key=f"bot:{q}")

                    # start a new chat
                    user_prompt = st.text_input("Ask a question")
                    if user_prompt:
                        with st.spinner("GPT is thinking ..."):
                            res, time_taken = builder.chatify(user_prompt=user_prompt, system_prompt=builder.system_prompt)
                            res = res.choices[0].message.content
                        took = f"\n\n<sub>Answered by `{builder.model_name}` in `{time_taken:.2f}` seconds.</sub>"
                        st.markdown(f"> {res}", unsafe_allow_html=True)
                        st.markdown(took, unsafe_allow_html=True)
                        st.markdown("---")
                        
                        # add to history conversation
                        st.session_state.chat_history[user_prompt] = {"user": user_prompt, "bot": res + took, "time_taken": time_taken}
                    
                    if st.button("Clear chat history"):
                        st.session_state.chat_history = {}
                    
                    if st.checkbox("View prompt"):
                        st.code(builder.system_prompt)

                    
                    if st.checkbox("Add Chat History as appendx to the slides"):
                        markdown += f"""\n\n\n## Q/A Appendix\n\n"""
                        for question, answer in st.session_state.chat_history.items():
                            markdown += f"""\n\n\n## {question}\n"""
                            for paragraph in answer["bot"].split("\n\n"):
                                markdown += f"""\n\n\n####\n<sub>{paragraph}</sub>\n"""
                            # markdown += f"""\n\n\n<sup>{took}</sup>\n"""
                        
                        with open(markdown_path, "w") as f:
                            f.write(markdown)
                        
                        # -- convert to slides with Q/A appendix
                        cmd_str = f"pandoc -t revealjs -s -o APPEDIX-{slides_path} {markdown_path} -V revealjs-url=https://unpkg.com/reveal.js/ -V theme={theme} --include-in-header=assets/slides.css "
                        import subprocess
                        subprocess.run(cmd_str, shell=True)
                        # st.markdown(markdown, unsafe_allow_html=True)

                        # -- render slides
                        st.markdown(f"> {paper_topic}", unsafe_allow_html=True)
                        import streamlit.components.v1 as components
                        with open(f"APPEDIX-{slides_path}", "r") as f:
                            html = f.read()
                        components.html(html, width=950, height=769, scrolling=False)

                        # -- download appendix slides
                        with open(f"APPEDIX-{slides_path}", "r") as f:
                            st.download_button("Download slides with Q/A appendix", f, file_name=f"APPEDIX-{paper_topic}.html", mime="text/html")


    @staticmethod
    @st.cache_resource
    def _build_paperqa(paths: List[str], papers: List[str] = None):
        # -- initialize PaperQA index
        # st.write(paths)
        # if not hasattr(st.session_state, "index_is_built"):
        #     st.session_state.index_is_built = False
        # if not st.session_state.index_is_built or not hasattr(
        #     st.session_state, "paperqa"
        # ):
        #     st.session_state.paperqa = PaperQA()
        #     with st.spinner("Building index"):
        #         st.session_state.paperqa.build_index(tuple(paths[:3]))
        #         st.session_state.index_is_built = True
        # paperqa = st.session_state.paperqa

        paperqa = PaperQA()
        paperqa.build_index(tuple(paths))
        return paperqa
    
    def _chatbot(self, paperqa):
        # # -- ask questions
        # st.info("WIP: Ready to answer questions about the papers above", icon="👷")
        question = st.text_input("Ask a question about the papers")
        if question:
            # question = "What type of cancer is discussed in the text?"
            # st.write(f"> **Question**: {question}")
            st.write(f"`{paperqa.ask.cache_info()}`")
            answer = paperqa.ask(question)
            st.markdown(answer)
        
        with st.expander("View docs object"):
            st.write(paperqa.docs)
        with st.expander("View indexed papers"):
            st.write(paperqa.docs.docs)
            

    @staticmethod
    @st.cache_resource
    def _build_knowledge_graph(abstracts, max_tokens=7500, max_papers=20, max_retries=12):
        """Build knowledge graph from abstracts using LangChain Graph"""
        # -- initialize GraphIndexCreator
        # index_creator = GraphIndexCreator(llm=OpenAI(model_name="gpt-4", temperature=0))#, max_context_length=4097))
        # try:
        MAX_TOKENS = max_tokens
        MAX_RETRIES = max_retries
        index_creator = GraphIndexCreator(llm=OpenAI(model_name="gpt-4", max_tokens=MAX_TOKENS, temperature=0, max_retries=MAX_RETRIES))#, max_context_length=4097))
            
        # Limiting papers for now, knowledge graph triplets is a bit intesive (for API usage) at the moment
        # -- This model's (key) maximum context length is 4097 tokens
        MAX_PAPERS = max_papers
        # st.write(os.environ["OPENAI_API_KEY"])
        text = " ".join([txt for txt in abstracts[:MAX_PAPERS] if txt])
        graph = index_creator.from_text(text)

        return graph

    @staticmethod
    @st.cache_resource  # (allow_output_mutation=True)
    def _get_papers(paper_topic: str, from_year: int, to_year: int) -> List[Paper]:
        s2 = SemanticScholarSearch(paper_topic, from_year, to_year)
        return s2.get_papers()

    def store_document_locally(self, docs: List[str]):
        if not os.path.exists("docs"):
            os.mkdir("docs")
        for doc in docs:
            import re

            year, title, abstract = doc.year, doc.title, doc.abstract
            clean_title = "_".join(re.findall(r"[a-zA-Z]+", title))
            with open(f"docs/{year}_{clean_title}.txt", "w") as f:
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
        st.success("Knowledge Graph is ready")  # , icon="✅")
        st.write(f"> Number of the extracted triplets: {len(triplets)}")
        df = pd.DataFrame(triplets, columns=["subject", "object", "predicate"])
        arrange_cols = ["subject", "predicate", "object"]
        df = df[arrange_cols]

        with st.expander("View triplets"):
            cols = st.columns(2)
            with cols[0]:
                st.dataframe(df, use_container_width=True, height=769)
            with cols[1]:
                st.write(triplets)

        # -- visualize graph
        nodes = []
        edges = []
        seen = set()
        for triple in triplets:
            h, t, r = triple
            if not h in seen:
                nodes.append(
                    Node(id=h, label=h, size=14, shape="diamond")
                )  # , color="green"))
                seen.add(h)
            if not t in seen:
                nodes.append(Node(id=t, label=t, size=10, shape="star", color="green"))
                seen.add(t)

            edges.append(Edge(source=h, target=t, label=r, size=5, color="red"))

        config = Config(
            width=2000, height=1200, directed=True, physics=True
        , hierarchical=True)

        with st.expander("Visualize graph"):
            graph = agraph(nodes=nodes, edges=edges, config=config)


class SemanticScholarSearch:
    Paper = namedtuple("Paper", ["title", "abstract", "year", "url", "authors"])

    def __init__(self, paper_topic: str, from_year: int, to_year: int, limit: int = 100):
        self.sch = SemanticScholar()
        self.response = self.search_papers(paper_topic, from_year, to_year, limit)

    def get_papers(self):
        with st.spinner("Parsing papers"):
            return self.parse_response(self.response)

    @lru_cache(maxsize=32)
    def search_papers(self, paper_topic: str, from_year: int, to_year: int, limit: int) -> List[str]:
        return self.sch.search_paper(paper_topic, year=f"{from_year}-{to_year}", limit=limit)

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
        self.docs = Docs(llm="gpt-4")
        # self.build_index(my_docs)

    # @lru_cache(maxsize=32)
    def build_index(self, my_docs_paths: Tuple[str]):  # , citation: List[str] = None):
        for path in my_docs_paths:  # ): #, my_docs):
            sleep(5)
            self.docs.add(path)  # , citation=doc.authors + " " + doc.year)

    @lru_cache(maxsize=32)
    def ask(self, question: str) -> List[str]:
        return self.docs.query(question)


if __name__ == "__main__":
    UI()
