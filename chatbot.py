import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time


# --------------- RAG 설정 --------------- #
# OpenAI 키 및 Agent 초기화
my_key = "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"

answer_agent = ChatOpenAI(
    openai_api_key=my_key,
    model_name='gpt-4o'
)

sub_query_agent = ChatOpenAI(
    openai_api_key=my_key,
    model_name="gpt-4o"
)

sub_query_system_message = SystemMessage(content="""
        너는 세종대왕 시기를 연구한 역사학자야.  가 알고 있는 지식을 기반으로 유저의 쿼리를 실록에서 검색 가능한 문장으로 구체화해야 해.
        무엇보다 세종대왕에 대한 일반적인 쿼리에 대해 가장 관련성 높은 하위 쿼리로 분화시켜야 해.
        다음을 반드시 준수해:
        1. 입력받은 질문을 실제 실록에 등장할 법한 표현과 어휘를 사용해 하나의 완성된 주술 구조를 만든다.
        2. 사건, 인물, 결과가 포함된 이야기 형식으로 작성해.
        3.'세종대왕'은 '임금'이나 '짐'으로, '부인 및 왕비'는 '중궁', 태종'은 '상왕'으로 표현돼.
        4. '~기록이 있는가?'처럼 유저의 질문을 그대로 받지 말고, 아래 예시처럼 실제 있을 법한 사실 묘사의 문장으로 바꿔줘.
        ex. 세종대왕의 업적이 뭐야? -> 측우기를 발명했다 / 농사직설을 편찬했다 / 훈민정음을 반포했다
    """)

answer_agent_system_message = SystemMessage(content="""
        너는 세종대왕 시기의 조선을 전공한 박물관 해설가로, 유저가 질문하거나 이야기를 들을 때 흥미롭게 설명하며 교훈과 영감을 주는 역할을 맡고 있어. 
        네 답변은 단순히 정보를 나열하는 것을 넘어, 실록 속 이야기를 바탕으로 유저와 연결될 수 있도록 구성되어야 해.
        
        답변 작성 시 다음 사항을 준수해:
        1. 유저 질문에 대해 최소 3개에서 최대 5개의 실록 기사를 참고하여 답변을 작성하되, 문단 형식으로 서술하고 각각의 기사는 날짜와 내용을 자연스럽게 포함해.
           예: "1432년 4월 15일, 세종대왕은 측우기를 발명하여 농업의 생산성을 높였습니다."
        2. 질문이 감정적인 요소를 포함한다면, 세종대왕이 어떤 생각을 했거나, 어떤 방식으로 문제를 해결했는지 그 맥락을 함께 설명하며 세종대왕의 삶과 업적을 통해 공감하거나 위로하며 교훈을 전달해.
        3. 답변의 마지막에는 항상 "해당 날짜의 실록에서 더 생생한 이야기를 살펴보세요."라는 문구로 마무리해.
    """)


conversations = [answer_agent_system_message]
max_messages = 5

# 세종실록 DB 초기화
persist_directory = "./chroma_db"
embed_model = OpenAIEmbeddings(
  openai_api_key=my_key,
  model="text-embedding-3-small"
  )

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embed_model,
    collection_name="raw"
)


# Agent 별 기능(함수)
def generate_sub_queries(user_query: str, num_sub_queries=6) -> list:
    """
    OpenAI LLM을 사용하여 하위 쿼리를 생성.
    """
    messages = [
        sub_query_system_message,
        HumanMessage(content=f""" 유저 질문: {user_query}
        위 질문을 조선왕조실록의 표현 방식에 맞게 {num_sub_queries}개의 하위 쿼리로 변환해줘. 
        하위 쿼리는 실록에서 검색 가능한 인물 및 사건 중심 표현으로 구성해""")
    ]
    response = sub_query_agent.invoke(messages)
    return response.content.strip().split("\n")


def gather_contexts(queries: list, db, target_k=8, fetch_k=15, max_attempts=5) -> str:
    """
    여러 하위 쿼리를 통해 검색된 결과를 종합.
    
    Args:
        queries (list): 하위 쿼리 리스트.
        db (Chroma): ChromaDB 객체.
        target_k (int): 각 쿼리에서 선택할 결과 수.
        max_attempts (int): 각 쿼리에서 검색 시도 최대 횟수.
    
    Returns:
        str: 종합된 검색 결과 문자열.
    """
    contexts = []  
    seen_contents = set() 

    for query in queries:
        # 중복을 제거하며 target_k 만큼 검색
        # results, seen_contents = search_documents_until_k(
        #     query=query,
        #     db=db,
        #     target_k=target_k,
        #     max_attempts=max_attempts,
        #     seen_contents=seen_contents
        # )

        results, seen_contents = search_documents_until_k_with_mmr(
            query=query,
            db=db,
            target_k=target_k,
            fetch_k=fetch_k,
            max_attempts=max_attempts,
            seen_contents=seen_contents
        )
        
        # 검색 결과를 컨텍스트로 추가
        for doc in results:
            contexts.append(f"{doc.metadata['year']}년 {doc.metadata.get('month', '알 수 없음')}월 "
                            f"{doc.metadata.get('day', '알 수 없음')}일: {doc.page_content}")

    # 최종적으로 모든 컨텍스트를 하나의 문자열로 합침
    return "\n".join(contexts)


def search_documents_until_k(query: str, db, target_k=8, max_attempts=5, seen_contents=None) -> list:
    """
    하나의 쿼리에 대해 목표 개수(target_k)만큼 중복을 제거한 결과를 검색.
    
    Args:
        query (str): 검색 쿼리.
        db (Chroma): ChromaDB 객체.
        target_k (int): 최종적으로 가져올 결과 개수.
        max_attempts (int): 최대 검색 시도 횟수.
        seen_contents (set): 이전에 검색된 문서의 내용 집합 (중복 제거용).
    
    Returns:
        list: 목표 개수만큼 중복이 제거된 Document 객체 리스트.
    """
    if seen_contents is None:
        seen_contents = set()

    unique_results = []  
    attempts = 0  

    while len(unique_results) < target_k and attempts < max_attempts:
        attempts += 1
        try:
            # 검색 수행
            results = db.similarity_search(query, k=target_k)
            
            # 중복 제거
            for doc in results:
                content = doc.page_content.strip()  
                if content not in seen_contents:
                    unique_results.append(doc)
                    seen_contents.add(content) 

                # 목표 개수 도달 시 중단
                if len(unique_results) >= target_k:
                    break
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            break  

    return unique_results[:target_k], seen_contents

def search_documents_until_k_with_mmr(query: str, db, target_k=8, fetch_k=15, max_attempts=5, seen_contents=None) -> list:
    """
    MMR 방식을 사용해 목표 개수(target_k)만큼 중복을 제거한 결과를 검색.
    
    Args:
        query (str): 검색 쿼리.
        db (Chroma): ChromaDB 객체.
        target_k (int): 최종적으로 가져올 결과 개수.
        fetch_k (int): 각 검색에서 가져올 최대 결과 수.
        max_attempts (int): 최대 검색 시도 횟수.
        seen_contents (set): 이전에 검색된 문서의 내용 집합 (중복 제거용).
    
    Returns:
        list: 목표 개수만큼 중복이 제거된 Document 객체 리스트.
    """
    if seen_contents is None:
        seen_contents = set()

    unique_results = []  # 중복이 제거된 최종 결과 리스트
    attempts = 0  # 현재 시도 횟수

    while len(unique_results) < target_k and attempts < max_attempts:
        attempts += 1
        try:
            # MMR 검색 수행
            results = db.max_marginal_relevance_search(query, fetch_k=fetch_k, k=target_k)
            
            # 중복 제거
            for doc in results:
                content = doc.page_content.strip()  # 내용 정리
                if content not in seen_contents:
                    unique_results.append(doc)
                    seen_contents.add(content)  # 추가된 문서의 내용 기록

                # 목표 개수 도달 시 중단
                if len(unique_results) >= target_k:
                    break
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            break  # 오류 발생 시 중단

    return unique_results[:target_k], seen_contents



def generate_final_answer(user_query: str, context: str) -> str:
    """
    OpenAI LLM을 사용하여 최종 답변 생성.
    """
    messages = [
        answer_agent_system_message,
        HumanMessage(content=f"""
        유저 질문에 답변을 작성해 주세요. 시스템 메시지의 지침을 준수하여 작성해 주세요.
        
        Query: {user_query}
        
        Context:\n{context}\n\n위 자료를 바탕으로 답변을 작성해줘.
        """)
    ]
    
    response = answer_agent.invoke(messages)
    return response.content



# --------------- Streamlit 설정 --------------- #
st.title("🏛️ 세종대왕도 사람이다")

if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

if "query" not in st.session_state:
        st.session_state["query"] = ""

def on_click():
    st.session_state["query"] = st.session_state["user_input"]
    st.session_state["user_input"] = ""

def upadate_input():
    st.session_state["query"] = st.session_state["user_input"]


tab1, tab2 = st.tabs(["질문과 답변받기", "갤러리"])

# 탭 1: 질문과 답변받기
with tab1:
    # 제목 및 설명
    st.markdown("### 🔍 자유롭게 질문하기")
    st.markdown(
        "세종대왕은 그저 위대한 업적만 남긴 왕이 아닙니다.  \n"
        "가족을 사랑하고, 백성을 걱정하며, 때로는 고뇌에 빠지기도 했던 한 사람.  \n"
        "그의 인간적인 모습과 고민을 큐레이터의 해설로 만나보세요.  \n"
        "질문을 입력하거나 아래 버튼을 클릭해 흥미로운 이야기를 들어보세요."
    ) 

    st.markdown("<br>", unsafe_allow_html=True) 

    # 사전 질문 버튼
    with st.container():
        st.markdown("""
        <style>
        /* 버튼 스타일 정의 */
        div.stButton > button {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px;
            background-color: #f9f9f9;
            color: #333;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
        }
        div.stButton > button:hover {
            background-color: #e0e0e0;
            transform: scale(1.03);
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1], gap="small")  # 간격 조정
        with col1:
            if st.button("❤️ 사랑꾼 세종"):
                st.session_state["user_input"] = "세종대왕은 로맨티스트였을까? 왕비를 어떻게 존중하고 사랑했는지 궁금해"
        with col2:
            if st.button("🎨 세종의 취미"):
                st.session_state["user_input"] = "세종대왕은 어떤 취미 생활을 즐겼을까?"
        with col3:
            if st.button("👁️ 세종과 건강"):
                st.session_state["user_input"] = "세종대왕이 시력이 좋지 않았다는데, 어느 수준의 눈병을 앓은 거야?"

        col4, col5, col6 = st.columns([1, 1, 1], gap="small")  # 두 번째 줄
        with col4:
            if st.button("🖋️ 훈민정음"):
                st.session_state["user_input"] = "훈민정음 창제 경위와 그 과정에서 세종대왕이 직면한 어려움은 무엇이었을까?"
        with col5:
            if st.button("🌾 농업 정책"):
                st.session_state["user_input"] = "세종대왕이 농업 발전에 기여한 정책은 무엇이고 백성들에게 어떤 영향을 미쳤을까?"
        with col6:
            if st.button("📚 학문 열정"):
                st.session_state["user_input"] = "세종대왕은 학문과 과학을 중시한 것으로 유명한데, 어떤 성취를 이루었을까?"


    # 질문(유저 입력)창 및 전송 버튼
    instr = "질문을 입력하세요."

    with st.form('chat_input_form'):
        col1, col2 = st.columns([9, 1]) 

        with col1:
            user_input = st.text_input(
                label="",
                # value=st.session_state["user_input"],
                placeholder=instr,
                label_visibility="collapsed",
                key ="user_input"
            )

        with col2:
            submitted = st.form_submit_button('📤', on_click=on_click)
                
        # 스타일링 적용
        st.markdown("""
        <style>
        /* 버튼 간격 조정 */
        div.stButton > button {
            width: 100%;  /* 버튼의 크기를 칼럼 폭에 맞춤 */
            margin-top: -10px;  /* 상단 여백 감소 */
        }

        /* 입력창과 버튼의 높이 정렬 */
        div.stTextInput > label {
            font-size: 16px;  /* 입력창의 텍스트 크기 */
        }
        div.stButton > button {
            height: 38px;  /* 버튼의 높이를 입력창과 일치 */
            vertical-align: middle;
        }
        </style>
        """, unsafe_allow_html=True)


        # 답변 생성 로직
        if submitted:
            if st.session_state["query"].strip():  
                # 입력 창 초기화
                status_container = st.empty()  

                # 단계 1: 하위 쿼리 생성
                with st.spinner("🧐 질문을 확인하고 있어요..."):
                    sub_queries = generate_sub_queries(st.session_state["query"])

                # 단계 2: 자료 검색
                with st.spinner("📚 세종대왕의 일생을 돌아보고 있어요..."):
                    context = gather_contexts(sub_queries, db, target_k=8)

                # 단계 3: 답변 생성
                with st.spinner("✍️ 이야기를 재구성 중이에요..."):
                    final_answer = generate_final_answer(st.session_state["query"], context)

                # 상태 종료
                status_container.empty()

                # 결과 출력
                st.success("해설이 준비됐습니다!")
                st.markdown("### 🕰️ **질문:** " + st.session_state["query"])
                st.markdown(f"### 📜 **큐레이터의 답변**\n\n{final_answer}")
                st.markdown(
                    '<a href="https://sillok.history.go.kr/search/inspectionMonthList.do?id=kda" target="_blank">🔗 해당 내용을 세종실록에서 더 확인해보세요</a>',
                    unsafe_allow_html=True
                )

                # 기록 저장
                st.session_state["qa_history"].append({"question": st.session_state["query"], "answer": final_answer})
            else:
                st.warning("질문을 입력해주세요!")




# 탭 2: 이전 질문-답변 갤러리
with tab2:
    st.markdown("### 📖 질문-답변 갤러리")
    st.markdown(
        "여기에는 관람객들과 나눈 세종대왕에 관한 이야기가 기록되어 있습니다.  \n"
        "그의 고민과 통찰이 600년이 흘러도 우리 삶에 깊은 영감을 준답니다.  \n"
        "더 생생한 이야기는 아래 세종실록 원문에서 만나보세요!"
    )
    st.markdown('<a href="https://sillok.history.go.kr/search/inspectionMonthList.do?id=kda" target="_blank">🔗 세종실록 바로가기</a>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) 

    if st.session_state["qa_history"]:
        for i, qa in enumerate(st.session_state["qa_history"], 1):
            with st.expander(f"질문 : {qa['question']}"):
                st.markdown(f"{qa['answer']}")
    else:
         with st.container():
            st.markdown("""
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 16px; 
                background-color: #f9f9f9;">
                아직 기록된 질문과 답변이 없습니다.  
            </div>
            """, unsafe_allow_html=True)