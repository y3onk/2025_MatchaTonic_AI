import sys
import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain / LangGraph 관련 임포트
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# =============================================================================
# [설정] 프롬프트 경로 및 로드
# =============================================================================
PROMPT_PATH = r"D:\2025_MatchaTonic_AI\Prompt"
if PROMPT_PATH not in sys.path:
    sys.path.append(PROMPT_PATH)

try:
    from prompt import (
        ANALYST_SYS_PROMPT,
        RECOMMENDER_SYS_PROMPT,
        ARCHITECT_SYS_PROMPT,
        EXECUTOR_SYS_PROMPT,
    )

    print(f" 프롬프트 로드 성공: {PROMPT_PATH}")
except ImportError:
    print(f" 오류: prompt.py를 찾을 수 없습니다. 경로: {PROMPT_PATH}")
    sys.exit(1)

# =============================================================================
# 1. 환경 변수 및 모델 설정
# =============================================================================
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(" .env 파일에 OPENAI_API_KEY가 없습니다.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================================================================
# 2. 데이터 모델 정의 (Pydantic)
# =============================================================================


# [Agent 1] Project Analyst (프로젝트 분석가) -> Output: ProjectAnalysis
class ProjectAnalysis(BaseModel):
    team_structure: str = Field(description="팀 구성 및 규모")
    work_cycle: str = Field(description="주요 협업 주기")
    pain_points: List[str] = Field(description="현재 겪고 있는 협업의 어려움")
    core_needs: List[str] = Field(description="팀에게 가장 시급한 관리 요소")


# [Agent 2] Template Recommender (템플릿 추천가) -> Output: TemplateRecommendation
class TemplateItem(BaseModel):
    name: str = Field(description="추천 템플릿/DB 이름")
    purpose: str = Field(description="사용 목적")
    reason: str = Field(description="추천 이유")


class TemplateRecommendation(BaseModel):
    recommended_templates: List[TemplateItem] = Field(description="추천된 템플릿 목록")


# [Agent 3] System Architect (시스템 설계자) -> Output: SystemArchitecture
class DBProperty(BaseModel):
    name: str
    type: str = Field(description="속성 타입 (Text, Number, Person, Date, Relation 등)")


class DatabaseSchema(BaseModel):
    title: str
    properties: List[DBProperty] = Field(description="DB 속성 목록")
    relations: List[str] = Field(description="다른 DB와의 연결 관계")


class SystemArchitecture(BaseModel):
    databases: List[DatabaseSchema] = Field(description="설계된 전체 시스템 아키텍처")


# [Agent 4] API Executor (API 실행자) -> Output: ApiExecutionPayload
class ApiPayloadItem(BaseModel):
    resource_type: str = Field(description="생성할 리소스 타입")
    payload_json: str = Field(description="Notion API 전송용 JSON")


class ApiExecutionPayload(BaseModel):
    api_payload_list: List[ApiPayloadItem]


# =============================================================================
# 3. LangGraph 상태(State) 및 노드(Node) 정의
# =============================================================================


class AgentState(TypedDict):
    user_input: str
    # 명칭 통일: 각 에이전트의 결과물(result)을 저장
    project_analysis_result: dict  # Project Analyst Output
    template_recommendation_result: dict  # Template Recommender Output
    system_architecture_result: dict  # System Architect Output
    api_execution_result: dict  # API Executor Output


# --- Node 1: Project Analyst ---
def node_project_analyst(state: AgentState):
    print("\n1️[Project Analyst] 프로젝트 요구사항 분석 중...")

    prompt = ANALYST_SYS_PROMPT.format(user_input=state["user_input"])

    chain = llm.with_structured_output(ProjectAnalysis)
    result = chain.invoke(prompt)

    return {"project_analysis_result": result.model_dump()}  # pyright: ignore


# --- Node 2: Template Recommender ---
def node_template_recommender(state: AgentState):
    print("2️[Template Recommender] 최적 템플릿 추천 중...")

    # 이전 단계 결과 가져오기
    prev_data = json.dumps(state["project_analysis_result"], ensure_ascii=False)
    prompt = RECOMMENDER_SYS_PROMPT.format(analysis_data=prev_data)

    chain = llm.with_structured_output(TemplateRecommendation)
    result = chain.invoke(prompt)

    return {"template_recommendation_result": result.model_dump()}  # pyright: ignore


# --- Node 3: System Architect ---
def node_system_architect(state: AgentState):
    print("3️[System Architect] 시스템 아키텍처 및 ERD 설계 중...")

    # 이전 단계 결과 가져오기
    prev_data = json.dumps(state["template_recommendation_result"], ensure_ascii=False)
    prompt = ARCHITECT_SYS_PROMPT.format(recommendation_data=prev_data)

    chain = llm.with_structured_output(SystemArchitecture)
    result = chain.invoke(prompt)

    return {"system_architecture_result": result.model_dump()}  # pyright: ignore


# --- Node 4: API Executor ---
def node_api_executor(state: AgentState):
    print("4️[API Executor] Notion API Payload 생성 중...")

    # 이전 단계 결과 가져오기
    prev_data = json.dumps(state["system_architecture_result"], ensure_ascii=False)
    prompt = EXECUTOR_SYS_PROMPT.format(blueprint_data=prev_data)

    chain = llm.with_structured_output(ApiExecutionPayload)
    result = chain.invoke(prompt)

    return {"api_execution_result": result.model_dump()}  # pyright: ignore


# =============================================================================
# 4. 그래프(Workflow) 구성
# =============================================================================

workflow = StateGraph(AgentState)

# 노드 등록 (snake_case로 통일)
workflow.add_node("project_analyst", node_project_analyst)
workflow.add_node("template_recommender", node_template_recommender)
workflow.add_node("system_architect", node_system_architect)
workflow.add_node("api_executor", node_api_executor)

# 흐름 연결 (Sequential)
workflow.set_entry_point("project_analyst")
workflow.add_edge("project_analyst", "template_recommender")
workflow.add_edge("template_recommender", "system_architect")
workflow.add_edge("system_architect", "api_executor")
workflow.add_edge("api_executor", END)

app = workflow.compile()

# =============================================================================
# 5. 실행 및 결과 출력
# =============================================================================


def print_pretty_json(title: str, data: dict):
    print(f"\n{'='*10} {title} {'='*10}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    user_scenario = """
    이번에 진행할 프로젝트는 '사내 점심 메뉴 추천 및 법인카드 결제 자동화 시스템' 구축이야. 
    매일 점심 메뉴 고르느라 버리는 시간이 너무 아깝고, 영수증 증빙 처리하는 것도 귀찮아서 이걸 싹 다 자동화해서 업무 효율을 높이는 게 목표야.
    
    우리 팀은 총 5명으로 구성되어 있어. 전체 일정을 관리할 PM인 나 1명, 서버를 담당할 백엔드 개발자 2명, 화면을 만들 프론트엔드 1명, 그리고 UI/UX 디자이너 1명 이렇게 한 팀이야.
    
    일정이 좀 빠듯한데, 올해 12월 31일까지는 무조건 마무리를 해야 해.
    그때까지 나와야 할 산출물은 요구사항 명세서, 시스템 설계도(ERD), 피그마 디자인 시안, 그리고 실제로 작동하는 MVP 웹사이트랑 최종 결과 보고서까지야.
    """

    print(f"[Start] Notion Workspace Builder (Prompt Path: {PROMPT_PATH})")

    try:
        # 실행
        final_state = app.invoke({"user_input": user_scenario})  # type: ignore

        # 결과 확인 (변경된 Key 이름 반영)
        print_pretty_json(
            "1. Project Analyst Result", final_state["project_analysis_result"]
        )
        print_pretty_json(
            "2. Template Recommender Result",
            final_state["template_recommendation_result"],
        )
        print_pretty_json(
            "3. System Architect Result", final_state["system_architecture_result"]
        )
        print_pretty_json("4. API Executor Result", final_state["api_execution_result"])

        print("\n 워크플로우 완료.")

    except Exception as e:
        print(f"\n 에러 발생: {e}")
        import traceback

        traceback.print_exc()
