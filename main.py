import streamlit as st
from src.utils.agent import create_workflow_graph
from src.semantic_cache.operations import get_from_cache, set_in_cache, clear_cache


@st.cache_resource
def get_graph():
    return create_workflow_graph()


def main():
    graph = get_graph()

    st.set_page_config(page_title="Data Agent Chat", page_icon="🤖", layout="wide")
    st.title("📊 HERMES")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your data agent anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        input_data = {"query": prompt, "max_tries": 0}

        response_container = st.chat_message("assistant")
        with response_container:
            response_placeholder = st.empty()
            final_output = None

            for chunk in graph.stream(input_data, stream_mode="values"):
                final_output = chunk  

            if isinstance(final_output, dict) and "result" in final_output:
                answer = final_output["result"]

            else:
                answer = str(final_output)

            set_in_cache(prompt, answer)
            response_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.sidebar.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

def test_search(db, query: str, k: int = 3):
    results = db.similarity_search_with_score(query, k=k)
    for i, (res, score) in enumerate(results, 1):
        print(f"Result {i} (Score: {score:.4f})\n{res.page_content}\n")


if __name__ == "__main__":
    main()



