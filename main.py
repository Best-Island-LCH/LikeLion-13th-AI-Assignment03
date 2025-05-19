import os ## 운영 체제와 상호작용하기 위한 모듈 (환경 변수 접근, 디렉토리 탐색, 파일 경로 설정 등에 사용됨.)

import json ## JSON 데이터를 파싱하거나 문자열로 변환하기 위한 모듈
# 문자열 형태의 JSON 데이터를 파이썬에서 쓸 수 있게 바꾸거나 or 파이썬 데이터를 JSON 문자열로 바꾼다는 뜻

from dotenv import load_dotenv, find_dotenv
## API 키나 비밀번호 등 민감한 정보를 코드에 직접 작성하지 않고 .env에 저장해 불러올 때 사용.
# - find_dotenv() : .env 파일의 경로를 찾아줍니다.
# - load_dotenv() : 해당 파일의 내용을 읽어 환경 변수로 등록

from openai import OpenAI 
## OpenAI 라이브러리의 OpenAI 클래스를 가져옴.

import tiktoken
## OpenAI 모델의 토큰 단위로 텍스트를 인코딩/디코딩할 수 있는 라이브러리.
# 토큰 수를 계산하거나, 프롬프트 길이를 제한할 때 사용됨.

load_dotenv(find_dotenv())
## from dotenv import load_dotenv, find_dotenv 를 실행하는 코드.

API_KEY = os.environ["API_KEY"]
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]
## .env 파일에 저장된 값을 환경 변수로부터 읽어오는 부분.

BASE_URL = "https://api.together.xyz" # API 호출을 보낼 기본 URL.
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" # 호출할 기본 모델 이름을 정의.
## 여기선 Meta에서 만든 최신 고성능 모델인 Llama 3.1 70B Turbo가 설정되어 있음.
FILENAME = "message_history.json"
## 메시지 기록을 저장하거나 불러오기 위한 파일 이름.
INPUT_TOKEN_LIMIT = 2048 
# 입력 토큰 제한.
# 사용자가 입력할 수 있는 최대 토큰 수를 제한해두는 용도.

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
## OpenAI API를 사용하기 위해 OpenAI 클라이언트를 생성하는 코드

def chat_completion(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    ## messages : GPT에 보낼 대화 내용 리스트
    ## model : 사용할 모델 이름(기본값으로 DEFAULT_MODEL 사용)
    ## temperature : 창의성 정도를 조절(0이면 매우 보수적, 1에 가까울수록 다양하고 창의적이게)
    ## **kwargs : 기타 추가 옵션(필요하면 stream, max_tokens 등 추가 가능)
    response = client.chat.completions.create(
        ## 모델에 요청을 보내고, 응답을 받아오는 메서드
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False,
        **kwargs,
    )
    return response.choices[0].message.content
    ## 응답 중 첫 번째 메시지의 실제 텍스트만 반환
    ### 왜 첫 번째만 반환할까?
    # -1 속도와 효율성
    #   하나의 답변만 필요하면 나머지는 불필요한 연산.
    #   빠르게 응답 받고, 간단하게 처리하기 좋음
    # -2 모델은 첫 번째 응답을 "가장 자신 있는 답변"으로 내놓는 경향
    #   생성된 답변 리스트는 "좋은 순서"로 정렬되어 있지 않지만, 보통 첫 번째 결과가 가장 직관적이고 안정적인 경향이 있음.
    #   특히 temperature가 낮을수록, 모델은 가장 확신 있는 결과를 먼저 내놓음.

def chat_completion_stream(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    ## 응답을 스트리밍으로 받아오는 함수
    ## 스트리밍? -> 모델이 전체 응답을 한 번에 주는 게 아니라, 한 글자 또는 한 문장 단위로 조금씩 쪼개서 계속 보내줌.
    ### 왜 스트리밍을 쓸까?
    # 1. 모델이 다 생성하기 전에도 사용자에게 일부 보여줄 수 있음
    # 2. UI에 좋음 -> 대화형 챗봇, 터미널 출력, 웹 프론트엔드 등에 자연스러움
    # 3. 중간에 끊을 수도 있고, 실시간으로 처리 가능
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        **kwargs,
    )

    response_content = "" # 이름 그대로 "응답 내용"을 저장할 변수
    ## 아무 내용도 없는 상태로 시작해서, 이후 스트리밍으로 받아온 텍스트 조각들을 하나씩 이어붙이기 위해 초기화.
    ## 스트리밍 방식에서는 초기화가 꼭 필요!!    

    for chunk in response:
        ## 스트리밍 응답은 조각단위로 들어옴. 이 루프는 모든 조각을 하나씩 순회.
        chunk_content = chunk.choices[0].delta.content
        # 해당 조각의 텍스트만 추출
        if chunk_content is not None: # 어떤 chunk에는 content가 없을 수도 있어서, 빈 조각은 건너뜀
            print(chunk_content, end="") # 실시간으로 한 조각씩 출력. end=""덕분에 줄바꿈 없이 이어 출력돼서, 자연스러운 대화 출력 가능.
            response_content += chunk_content # 각 조각을 response_content 문자열에 하나씩 붙임임

    print()
    return response_content
    ## 지금까지 스트리밍으로 받아온 응답 조각들을 전부 이어붙여서 하나의 문자열로 만들어 "결과값"으로 반환.

def count_tokens(text, model):
    # 주어진 텍스트가 얼마나 많은 토큰으로 구성되어 있는지를 게산해서 반환.
    encoding = tiktoken.get_encoding("cl100k_base")
    # tiktoken 은 OpenAI 모델들이 사용하는 "토크나이저" 라는 라이브러리
    # "cl100k_base"는 GPT-4, GPT-3.5-turbo 등이 사용하는 기본 토크나이저 규칙 이름.
    # 규칙에 맞는 인코더(텍스트 -> 토큰)를 가져옴.
    tokens = encoding.encode(text)
    # 실제로 문자열 text를 토큰 목록으로 인코딩.
    return len(tokens)
    # 토큰 개수를 세어서 반환. 즉, 이 텍스트를 모델에 보낼 때 몇 토큰이 드는지 계산하는 함수

def count_total_tokens(messages, model):
    # 여러 메시지들의 전체 토큰 수를 계산해서 반환하는 함수.
    total = 0 # 전체 토큰 수를 저장할 변수. 처음에는 0으로 싲가하고, 점점 누적시킴.
    for message in messages: # 메시지 리스트를 하나씩 돌면서 반복.
        total += count_tokens(message["content"], model)
        # 각 메시지의 "content"(실제 텍스트 내용)만 꺼내서 count_tokens()로 토큰 수를 계산하고, 
        # 그 값을 total에 누적함.
    return total # 전체 메시지들의 토큰 수 합계를 반환.

def enforce_token_limit(messages, token_limit, model=DEFAULT_MODEL):
    ## 총 토큰 수가 token_limit을 넘지 않도록 조정해주는 함수. 
    # 즉, 토큰이 너무 많으면 오래된 메시지를 제거해서 제한 안에 들어오도록 만듦.
    while count_total_tokens(messages, model) > token_limit:
        # 현재 messages에 있는 전체 토큰 수가 token_limit보다 많다면 계속 반복.
        if len(messages) > 1: 
            # 메시지가 2개 이상 있을 때만 제거. 
            # 보통 첫 번째 메시지는 system메시지이므로 그건 유지하려고 조건을 둔 것.
            messages.pop(1)
            # 두 번째 메시지(index=1)를 삭제.
            # 즉, 가장 오래된 사용자/assistant 메시지부터 제거해서 토큰을 줄이는 방식
            # index=1을 쓰는 이유는, index=0은 일반적으로 system message이기 때문에 삭제하지 않으려는 목적.
        else:
            break 
            # 만약 메시지가 1개밖에 없다면(=system 메시지밖에 없음) 
            # 더 이상 삭제할 수 없기 때문에 반복을 중지합니다.

def save_to_json_file(obj, filename):
    ## 파이썬 객체(obj)를 JSON 형식으로 파일에 저장하는 함수.
    with open(filename, "w", encoding="utf-8") as file:
        # filenam 이름의 파일을 쓰기 모드("w")로 열기
        # with문을 사용해서 자동으로 파일 닫기까지 처리
        # encoding="utf-8"로 저장 -> 한글 같은 비영어 문자도 깨지지 않게 처리.
        json.dump(obj, file, indent=4, ensure_ascii=False)
        # obj라는 파이썬 객체(dict, list 등)를 file에 JSON 형식으로 저장
        # indent=4 : 보기 좋게 들여쓰기 4칸으로 정리(사람이 읽기 좋음)
        # ensure_ascii=False: 한글 등 비 ASCII문자도 그대로 저장장

def load_from_json_file(filename):
    ## 지정한 JSON 파일을 읽어서 파이썬 객체로 변환해 반환하는 함수.
    ## 만약 파일이 없거나 읽는 도중 문제가 생기면, 오류 메시지를 출력하고 None을 반환.
    try: # 오류가 날 수 있는 코드 블록을 시작.
        with open(filename, "r", encoding="utf-8") as file:
            # filename 파일을 **읽기 모드("r")**로 열고, UTF-8 인코딩으로 처리.
            # with 문 덕분에 파일은 자동으로 닫힘.
            return json.load(file)
            # 파일 안의 JSON 데이터를 읽어 들여 파이썬 객체(dict, list 등)로 변환해서 반환.
    except Exception as e:
        ## 파일이 없거나 JSON 형식이 잘못되었거나 하면 예외가 발생.
        ## 그 예외를 잡아서 e에 저장.
        print(f"{filename} 파일을 읽는 중 오류 발생: {e}")
        return None


def chatbot():
    ## 터미널에서 작동하는 간단한 챗봇 인터페이스를 구현한 함수.
    ## 사용자의 입력을 받고, 모델의 응답을 출력하며, 대화 내용을 JSON 파일에 저장하는 구조.
    messages = load_from_json_file(FILENAME)
    ## 이전 대화 내용이 있으면 불러오고, 없으면 None을 받음.
    if not messages:
        # 대화가 없으면 시스템 메시지로 초기화화.
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    print("Chatbot: 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'quit' 또는 'exit'을 입력하세요.)")

    while True:
        user_input = input("You: ") ## 사용자 입력을 무한히 반복해서 받음.
        if user_input.lower() in ['quit', 'exit']:
            break
        ## 사용자가 'quit'이나 'exit'을 입력하면 종료.
        messages.append({"role": "user", "content": user_input})
        # 사용자의 입력을 대화 메시지에 추가.

        total_tokens = count_total_tokens(messages, DEFAULT_MODEL)
        print(f"[현재 토큰 수: {total_tokens} / {INPUT_TOKEN_LIMIT}]")
        # 전체 메시지의 토큰 수를 계산하고, 토큰 제한 내인지 확인.

        enforce_token_limit(messages, INPUT_TOKEN_LIMIT)
        # 토큰이 너무 많으면 오래된 메시지를 삭제해서 제한 안에 맞춤.

        print("Chatbot: ", end="")
        response = chat_completion_stream(messages)
        print()
        # 챗봇의 응답을 실시간으로 스트리밍해서 출력.

        messages.append({"role": "assistant", "content": response})
        # AI의 응답을 대화 메시지에 추가.

        save_to_json_file(messages, FILENAME)
        # 현재까지의 대화 내용을 JSON 파일에 저장.

chatbot()