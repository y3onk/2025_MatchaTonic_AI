import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import List, Optional, TypedDict

# =============================================================================
# [설정] 프롬프트 경로 및 로드
# =============================================================================
PROMPT_PATH = r"D:\2025_MatchaTonic_AI\Prompt"

# 시스템 경로에 추가 (없을 경우에만)
if PROMPT_PATH not in sys.path:
    sys.path.append(PROMPT_PATH)

# prompt.py에서 필요한 변수들을 직접 가져옵니다.
try:
    from prompt import (
        PARSER_SYSTEM,
        PARSER_USER_TEMPLATE,
        DECOMPOSER_SYSTEM,
        DECOMPOSER_USER_TEMPLATE,
        ESTIMATOR_SYSTEM,
        ESTIMATOR_USER_TEMPLATE,
        SCHEDULER_SYSTEM,
        SCHEDULER_USER_TEMPLATE,
    )

    print(f" 프롬프트 로드 성공: {PROMPT_PATH}")
except ImportError:
    print(f" 오류: prompt.py를 찾을 수 없습니다. 경로: {PROMPT_PATH}")
    sys.exit(1)

# --- 0. 설정 ---
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print(" .env 파일이나 환경 변수에 OPENAI_API_KEY가 없습니다.")

llm = ChatOpenAI(model="gpt-4o", temperature=0)  # type: ignore

# --- 1. Pydantic 모델 (데이터 구조) ---


class ProjectInfo(
    BaseModel
):  # [Agent 1: Input Parser 결과] 자연어에서 추출된 프로젝트 핵심 정보
    title: str
    goal: str
    people: int
    roles: List[str]
    deadline: str
    deliverables: str


class Task(BaseModel):  # [Agent 2 보조] 개별 작업 단위 (누가, 무엇을)
    role: str
    task_name: str


class TaskList(
    BaseModel
):  # [Agent 2: Task Decomposer 결과] 역할별로 분해된 핵심 과업 리스트
    tasks: List[Task]


class EstimatedTask(
    BaseModel
):  # [Agent 3 보조] 시간(Hours)과 순서(Dependency)가 포함된 기술적 작업 정보
    id: int
    role: str
    task_name: str
    estimated_hours: int
    dependencies: Optional[str]


class EstimationList(
    BaseModel
):  # [Agent 3: Effort Estimator 결과] 테크 리드가 분석한 견적서 목록
    estimates: List[EstimatedTask]


class ScheduledItem(
    BaseModel
):  # [Agent 4 보조] 실제 달력 날짜(Start/End)가 배정된 최종 일정 항목
    role: str
    task_name: str
    start_date: str
    end_date: str
    note: str


class ScheduleResult(
    BaseModel
):  # [Agent 4: Scheduler 결과] 마감일을 고려하여 확정된 최종 프로젝트 일정표
    schedule: List[ScheduledItem]


# --- 2. State 정의 ---
class ProjectState(TypedDict):
    raw_input: str
    project_info: ProjectInfo  # Output of Input Parser
    task_list: List[Task]  # Output of Task Decomposer
    estimations: List[EstimatedTask]  # Output of Effort Estimator
    final_schedule: List[dict]  # Output of Scheduler


# --- 3. Node 함수 (Agent 명칭 반영) ---


def node_input_parser(state: ProjectState):
    """Agent 1: Input Parser (요구사항 분석)"""
    print("--- [Input Parser] 프로젝트 정보 분석 중... ---")

    user_msg = PARSER_USER_TEMPLATE.format(
        raw_text=state["raw_input"], current_year=datetime.now().year
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", PARSER_SYSTEM), ("human", user_msg)]
    )

    chain = prompt | llm.with_structured_output(ProjectInfo)
    return {"project_info": chain.invoke({})}


def node_task_decomposer(state: ProjectState):
    """Agent 2: Task Decomposer (역할별 업무 분해)"""
    print("--- [Task Decomposer] 역할별 핵심 과업 분해 중... ---")
    info = state["project_info"]

    user_msg = DECOMPOSER_USER_TEMPLATE.format(
        title=info.title,
        goal=info.goal,
        people=info.people,
        deliverables=info.deliverables,
        roles=", ".join(info.roles),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", DECOMPOSER_SYSTEM), ("human", user_msg)]
    )

    chain = prompt | llm.with_structured_output(TaskList)
    return {"task_list": chain.invoke({}).tasks}  # type: ignore


def node_effort_estimator(state: ProjectState):
    """Agent 3: Effort Estimator (견적 및 의존성 산출)"""
    print("--- [Effort Estimator] 소요 시간 및 의존성 계산 중... ---")

    tasks_text = "\n".join([f"- [{t.role}] {t.task_name}" for t in state["task_list"]])

    user_msg = ESTIMATOR_USER_TEMPLATE.format(tasks_text=tasks_text)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ESTIMATOR_SYSTEM), ("human", user_msg)]
    )

    chain = prompt | llm.with_structured_output(EstimationList)
    return {"estimations": chain.invoke({}).estimates}  # type: ignore


def node_scheduler(state: ProjectState):
    """Agent 4: Scheduler (일정 배정)"""
    print("--- [Scheduler] 최종 일정 조율 및 JSON 생성 중... ---")
    estimations = state["estimations"]
    deadline = state["project_info"].deadline

    est_text = "\n".join(
        [
            f"ID:{e.id} | Role:{e.role} | Task:{e.task_name} | Est:{e.estimated_hours}h | Dep:{e.dependencies}"
            for e in estimations
        ]
    )

    # 시스템 프롬프트 포맷팅
    system_msg = SCHEDULER_SYSTEM.format(deadline=deadline)

    user_msg = SCHEDULER_USER_TEMPLATE.format(
        start_date=datetime.now().strftime("%Y-%m-%d"),
        deadline=deadline,
        est_text=est_text,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", user_msg)]
    )

    chain = prompt | llm.with_structured_output(ScheduleResult)
    result = chain.invoke({})

    # JSON 변환
    return {"final_schedule": [item.dict() for item in result.schedule]}  # type: ignore


# --- 4. Graph 구성 ---
def build_graph():
    workflow = StateGraph(ProjectState)

    # 노드 이름 변경 (snake_case 적용)
    workflow.add_node("input_parser", node_input_parser)
    workflow.add_node("task_decomposer", node_task_decomposer)
    workflow.add_node("effort_estimator", node_effort_estimator)
    workflow.add_node("scheduler", node_scheduler)

    # 엣지 연결
    workflow.set_entry_point("input_parser")
    workflow.add_edge("input_parser", "task_decomposer")
    workflow.add_edge("task_decomposer", "effort_estimator")
    workflow.add_edge("effort_estimator", "scheduler")
    workflow.add_edge("scheduler", END)

    return workflow.compile()


# --- 5. Main Execution ---
if __name__ == "__main__":
    user_input_text = """
    이번에 진행할 프로젝트는 '사내 점심 메뉴 추천 및 법인카드 결제 자동화 시스템' 구축이야. 
    매일 점심 메뉴 고르느라 버리는 시간이 너무 아깝고, 영수증 증빙 처리하는 것도 귀찮아서 이걸 싹 다 자동화해서 업무 효율을 높이는 게 목표야.
    
    우리 팀은 총 5명으로 구성되어 있어. 전체 일정을 관리할 PM인 나 1명, 서버를 담당할 백엔드 개발자 2명, 화면을 만들 프론트엔드 1명, 그리고 UI/UX 디자이너 1명 이렇게 한 팀이야.
    
    일정이 좀 빠듯한데, 올해 12월 31일까지는 무조건 마무리를 해야 해.
    그때까지 나와야 할 산출물은 요구사항 명세서, 시스템 설계도(ERD), 피그마 디자인 시안, 그리고 실제로 작동하는 MVP 웹사이트랑 최종 결과 보고서까지야.
    """

    print(f"[지능형 스케줄러] 프로젝트 분석 시작...")

    try:
        app = build_graph()
        result = app.invoke({"raw_input": user_input_text})  # type: ignore

        print("\n JSON 일정이 생성되었습니다!\n")
        print(json.dumps(result["final_schedule"], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f" 에러 발생: {e}")
        import traceback

        traceback.print_exc()
