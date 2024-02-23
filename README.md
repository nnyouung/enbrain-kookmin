# enbrain-kookmin
## 2023 학생주도프로젝트 (23.03.22 ~ 23.11.13)

<br>

## 💡 프로젝트 팀원
<table>
      <tbody>
          <tr>
            <tr>
              <td align='center'>배진우</td>
              <td align='center'>배세은</td>
              <td align='center'>최선우</td>
              <td align='center'>하은영</td>
            </tr>
            <tr>
              <td align='center'>인공지능전공</td>
              <td align='center'>인공지능전공</td>
              <td align='center'>소프트웨어전공</td>
              <td align='center'>소프트웨어전공</td>
            </tr>
          </tr>
      </tbody>
  </table>

<br><br>

## 💡 프로젝트 소개
### 1. TF-IDF 유사도로 추천해주는 모듈
**개발 기간: 2023년 3월 22일 수요일 ~ 4월 6일 목요일**
<br>
- 네이버 영화 제목과 줄거리를 크롤링 (데이타 갯수는 많으면 많을수록 좋음.)
- 줄거리를 기준으로 TF-IDF 벡터화 하여, 유사도가 높은 영화를 검색해주는 모듈 구현
- 결과는 간단한 데모 앱에서 입력으로 넣은 영화와 유사도가 높은 영화가 결과로 나오게 개발 (데모앱에 신경 쓸 필요 없습니다. 에디터박스, 버튼만 있는 아주 간단한 형태여도 상관 없습니다.)
- 서버 api는 파이썬으로 구현 (주고받는 데이터 형식은 제한이 없습니다.)
<br>
<details>
<summary>영화 데이터 크롤링 (하은영)</summary>

## 네이버 영화에서 영화 제목과 줄거리를 추출하여 가공

### 기능 설명 및 코드

1. **줄거리 특수문자 제거**
- 정규표현식을 사용하기 위해 re 모듈 사용

```python
# re.sub（정규 표현식, 치환 문자, 대상 문자열）
text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》‘’“”]', '', readData)
```

1. **줄거리 명사 추출 / 불용어 / 한 글자 제거**
- 명사 추출을 위해 konlpy의 Okt 모듈 사용
- 불용어 텍스트 파일을 이용하여 줄거리에서 불용어 제거
- 한 글자는 의미 없는 경우가 많으므로, 한 글자로 이루어진 단어 제거

```python
okt = Okt()
nouns = okt.nouns(readData)  # 명사만 뽑아내기

# 텍스트 파일 열기
korean_stopwords_path = "./korean_stopwords.txt"
with open(korean_stopwords_path, encoding='utf-8') as f:
    stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]

# 불용어 및 한 글자 제거
remove_char = [x for x in nouns if (x not in stopwords) and (len(x) > 1)]
```

1. **영화 제목 및 줄거리 크롤링**
- HTTP 요청을 보내기 위해 requests 모듈 사용
- 웹페이지 파싱 및 파싱한 문서에서 필요한 정보를 추출하기 위해 BeautifulSoup 모듈 사용

```python
for i in range(start_code, finish_code):
  movie_code = str(i)
  raw = requests.get("https://movie.naver.com/movie/bi/mi/basic.nhn?code=" + movie_code)
  html = bs(raw.text, 'html.parser')

  # 전체 컨테이너
  movie = html.select("div.article")

  # 전체 컨테이너가 가지고 있는 영화 관련 정보
  for a, m in enumerate(movie):

      # 영화 제목 수집
      title = m.select_one("h3.h_movie a")
      # m: BeautifulSoup으로 파싱된 HTML 문서 객체
      # select_one: HTML 문서에서 하나의 요소만 선택하는 메서드
      # "h3.h_movie a": h3 태그의 class 속성 값이 h_movie인 요소의 하위 태그 중 a 태그를 선택

      # 영화 줄거리 수집
      story = m.select("div.story_area p.con_tx")
      # select: HTML 문서에서 여러 요소를 선택하는 메서드
      # "div.story_area p.con_tx": div 태그의 class 속성 값이 story_area인 요소의 하위 태그 중 p 태그의 class 속성 값이 con_tx인 모든 요소를 선택

      # 줄거리가 없으면 넘어가기
      if len(story) == 0:
          continue

      # 영화 관련 정보 엑셀(xlsx) 형식 저장
      # 데이터 만들기 1: HTML로 가져온 정보에서 TEXT 정보만 뽑아서 리스트 형태로 만들기
      story_list = [s.text for s in story]

      # 데이터 만들기 2: 여러 개로 이루어진 리스트 형태를 하나의 문자열 형태로 만들고, 정보 가공
      story_str = ''.join(story_list).replace('\xa0', ' ')
      story_del = stopwords(story_str)   # 명사 추출 + 불용어 및 한 글자 제거
      story_clean = cleanText(story_del)  # 특수문자 제거

      # 데이터 만들기 3: 엑셀에 넣기 위해 리스트 형태로 만들기
      story_split = story_clean.split(' ')
      story_split.insert(0, title.text)  # 엑셀 한 행에 넣기 위해 타이틀을 줄거리(단어형식) 리스트 맨 앞에 넣기

      # 영화 관련 정보 엑셀 행 추가: line by line으로 추가
      sheet.append(story_split)
```

1. **엑셀 파일 생성 및 저장**
- 엑셀 파일을 생성하고 저장하기 위해 openpyx 모듈 사용
- 엑셀 파일을 csv 파일로 바꾸기 위해 pandas 모듈 사용

```python
global is_ok
is_ok = False
wb = openpyxl.Workbook()   # Workbook(): 빈 엑셀 파일을 생성
sheet = wb.active   # active: 현재 활성화된 시트 선택
```

```python
wb.save("navermovie1.xlsx")
df = pd.read_excel('navermovie1.xlsx')
df.to_csv("navermovie1.csv", index=False, header=False, encoding="utf-8-sig")
```

---

### 전체 코드

```python
import re   # 정규표현식을 위한 모듈
import requests   # HTTP 요청을 보내는 모듈
import openpyxl   # 엑셀 관련 모듈
import pandas as pd   # xslx -> csv로 바꾸기 위한 모듈
from bs4 import BeautifulSoup as bs   # 파싱 및 파싱한 문서에서 필요한 정보를 추출하는 모듈
from konlpy.tag import Okt   # 한국어 자연어 처리 모듈

# 특수문자 제거 위한 함수
def cleanText(readData):
    # 줄거리에 포함되어 있는 특수문자 제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》‘’“”]', '', readData)  # re.sub（정규 표현식, 치환 문자, 대상 문자열）
    return text

def stopwords(readData):
    okt = Okt()
    nouns = okt.nouns(readData)  # 명사만 뽑아내기

    # 텍스트 파일 열기
    korean_stopwords_path = "./korean_stopwords.txt"
    with open(korean_stopwords_path, encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]

    # 불용어 및 한 글자 제거
    remove_char = [x for x in nouns if (x not in stopwords) and (len(x) > 1)]

    # 문자열로 만들기
    text = ' '.join(remove_char)

    return text

def crawling(start_code, finish_code):
    try:
        global is_ok
        is_ok = False
        wb = openpyxl.Workbook()   # Workbook(): 빈 엑셀 파일을 생성
        sheet = wb.active   # active: 현재 활성화된 시트 선택

        # HTML 파싱
        j = 0
        # 네이버 영화의 영화 코드 범위 지정
        for i in range(start_code, finish_code):

            movie_code = str(i)
            raw = requests.get("https://movie.naver.com/movie/bi/mi/basic.nhn?code=" + movie_code)
            html = bs(raw.text, 'html.parser')

            # 전체 컨테이너
            movie = html.select("div.article")

            # 전체 컨테이너가 가지고 있는 영화 관련 정보
            for a, m in enumerate(movie):

                # 영화 제목 수집
                title = m.select_one("h3.h_movie a")
                # m: BeautifulSoup으로 파싱된 HTML 문서 객체
                # select_one: HTML 문서에서 하나의 요소만 선택하는 메서드
                # "h3.h_movie a": h3 태그의 class 속성 값이 h_movie인 요소의 하위 태그 중 a 태그를 선택

                # 영화 줄거리 수집
                story = m.select("div.story_area p.con_tx")
                # select: HTML 문서에서 여러 요소를 선택하는 메서드
                # "div.story_area p.con_tx": div 태그의 class 속성 값이 story_area인 요소의 하위 태그 중 p 태그의 class 속성 값이 con_tx인 모든 요소를 선택

                # 줄거리가 없으면 넘어가기
                if len(story) == 0:
                    continue

                # 출력용 (지워도 무방)
                print("=" * 50)
                print("제목:", title.text)
                print("줄거리: ")
                for s in story:
                    print(s.text)
                print("-" * 50)

                # 영화 관련 정보 엑셀(xlsx) 형식 저장
                # 데이터 만들기 1: HTML로 가져온 정보에서 TEXT 정보만 뽑아서 리스트 형태로 만들기
                story_list = [s.text for s in story]

                # 데이터 만들기 2: 여러 개로 이루어진 리스트 형태를 하나의 문자열 형태로 만들고, 정보 가공
                story_str = ''.join(story_list).replace('\xa0', ' ')
                story_del = stopwords(story_str)   # 명사 추출 + 불용어 및 한 글자 제거
                story_clean = cleanText(story_del)  # 특수문자 제거

                # 데이터 만들기 3: 엑셀에 넣기 위해 리스트 형태로 만들기
                story_split = story_clean.split(' ')
                story_split.insert(0, title.text)  # 엑셀 한 행에 넣기 위해 타이틀을 줄거리(단어형식) 리스트 맨 앞에 넣기

                # 영화 관련 정보 엑셀 행 추가: line by line으로 추가
                sheet.append(story_split)

                is_ok = True

            # 출력용 (지워도 무방)
            if is_ok == True:
                j = j + 1
            print(finish_code - start_code, "개 중에", finish_code - i, "개 남음")
            print((i - start_code)+1, "번째 영화 체크 중", j, "개의 영화 정보 저장 완료")

		# 엑셀 저장
    except:
        print("에러 발생")
        wb.save("navermovie1.xlsx")
        df = pd.read_excel('navermovie1.xlsx')
        df.to_csv("navermovie1.csv", index=False, header=False, encoding="utf-8-sig")

    finally:
        print("완료")
        wb.save("navermovie2.xlsx")
        df = pd.read_excel('navermovie2.xlsx')
        df.to_csv("navermovie2.csv", index=False, header=False, encoding="utf-8-sig")

crawling(165932, 215932)
```
</details>

<details>
<summary>TF-IDF (최선우)</summary>

  ## TF-IDF를 활용한 유사 영화 도출
### 정의: Term Frequency - Inverse Document Frequency

### 사용

- 문서의 유사도를 구하는 작업
- 검색 시스템에서 검색 결과의 중요도를 정하는 작업
- 문서 내에서 특정 단어의 중요도를 구하는 작업

### TF: 특정 줄거리(영화의) d에서의 특정 단어 t의 등장 횟수

```python
docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
```

- 위의 예시에서 문자열을 각각의 줄거리라고 했을 때 세번째 줄거리의 단어 “바나나”가 나온 횟수는 23번

### DF: 특정 단어 t가 등장한 줄거리의 수

- 특정 단어 t가 등장한 줄거리의 수
    - 한 줄거리에서 t가 몇번 나왔는지는 중요하지 않음
- 위의 예시에서 “바나나”가 등장한 줄거리의 개수는 23개

### IDF: DF에 반비례하는 수

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/46ac930a-5f48-4f69-8196-3f70f67e4e94/Untitled.png)

- 줄거리의 개수가 많아질 때 값이 너무 커지는 것을 막기 위해 log 사용
- df가 0일 때를 대비해 분모에 +1

- 많은 문서에서 나온 단어 ⇒ 어디에나 쓰이는 흔한 단어 ⇒ 중요한 단어 xx

### 사용 모듈

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfIdfVector= TfidfVectorizer().fit(self.movieList) # movieList에 있는 모든 단어를 TFIDF 벡터로 변환
result = tfIdfVector.transform(self.movieList).toarray() # movieList를 TFIDF 벡터로 변환
```

- self.movieList에는 각 영화의 줄거리가 존재
- 줄거리의 단어들에 대해 벡터화를 진행
- self.moveList를 벡터화된 단어들로 변경

---

## 유사도: 코사인 유사도

- 벡터화된 줄거리들끼리의 유사도를 구하기 위해 코사인 유사도 사용

### 코사인 유사도

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/e872cc57-1d23-46e2-9c0e-1bdfa3661a23/Untitled.png)

- 같은 차원의 벡터들의 각도를 구한 것
- 각도가 작을 수록 유사 ⇒ cos()이 클수록 유사 ⇒ 1에 가까울 수록 유사

```python
cos_sim =np.dot(story_input, self.moveTFIDF[move])/(norm(story_input)*norm(self.moveTFIDF[movie]))
# story_input: 입력으로 들어온 영화의 제목의 줄거리
# story_input에 대해 다른 영화들의 줄거리 벡터와 cosine sim를 구함
```

### 구현

```python
class cosine_sim:
	def __init__(self):
		# 영화 줄거리 TF-IDF 벡터화
	def cosine_sim_cal(self, name_input);
		# 입력받은 영화 제목과 다른 영화들의 cos_sim을 구하여 
		# 높은 cos_sim을 가진 영화 리스트를 출력
```

### 예시

입력
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/5cdb92cd-32e1-4883-84d5-0fc97f0f5a71/Untitled.png)

출력
</details>

<details>
<summary>서버(API) (배진우)</summary>
  - urls.py
  <br>
  - 내 ip:8000/api/movies/를 통해서 데이터를 받아야함.

```python
from django.contrib import admin
from django.urls import path
from myapp.views import movies

urlpatterns = [
     path('api/movies/', movies, name='movies'),
]
```

- settings.py
    - 현재 사용하고 있는 ip를 접근 허용
    
    ```python
    ALLOWED_HOSTS = ['10.30.116.172']
    ```
    
- models.py
    - 기존에 models.py를 사용하지 않고 cosin_sim.py를 models.py로 이용
    - django.db에서 models 를 불러 기존 models.py의 형식을 유지
    
    ```python
    import numpy as np
    from numpy.linalg import norm
    from django.db import models
    from .tf_idf import TFIDF
    
    class cosine_sim:
        
        def __init__(self):
    
            self.movieName, self.movieTFIDF = TFIDF().TFIDF_use_module()
    
        def cosine_sim_cal(self, name_input):
            if name_input in self.movieName:
    
                story_input = self.movieTFIDF[self.movieName.index(name_input)]
                result = []
    
                for movie in range(len(self.movieTFIDF)):
                    cos_sim =np.dot(story_input, self.movieTFIDF[movie])/(norm(story_input)*norm(self.movieTFIDF[movie]))
                    if cos_sim > 0.3:
                        result.append([movie,cos_sim])
    
                result = sorted(result, key = lambda x : -x[1]) # 정렬
                result.pop(0)
    
                movieSimName = []
                for m in result:
                    movieSimName.append(self.movieName[m[0]])
                
                return movieSimName
    
            
            else:
                return ["그런 영화는 없어요 ㅠㅠ"]
    ```
    
- view.py
    - 가장 메인인 파일로 데이터를 받고 전송 기능 구현
    - http 전송 방식을 이용하며 데이터의 형식은 Json파일을 이용한다.
    - API에서 사이트 간 요청 위조인 csrf 보안이 필요없기 때문에 간단하게 해체 가능한 csrf_exempt 사용
    - 받은 데이터를 movie_data에 넣어 cosine_sim 모듈을 이용해 유사한 영화 탐색
    - 나온 영화 제목을 리스트에 넣은 후 title 에 해당하는 value에 저장 후 json 형식으로 전송
    
    ```python
    from django.shortcuts import render
    from django.http import JsonResponse
    from django.views.decorators.csrf import csrf_exempt
    from .cosine_sim import cosine_sim
    import json
    # Create your views here
    
    @csrf_exempt
    def movies(request):
        if request.method == 'POST':
            print("DATA RECEIEVED!")
            movie_data = request.POST.dict()
            movie_list = []
            movie_object = cosine_sim()
            movie_title = movie_object.cosine_sim_cal(movie_data)
            for movie in movie_title:
                movie_list.append(movie )
            movie_data = {
                    'title' : movie_list
                    }
            return JsonResponse(movie_data, content_type='application/json; charset=utf-8')
    ```
    

[myproject.zip](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/73bdf794-01d9-4e38-a58d-49c3ed25bf3e/myproject.zip)
</details>

<details>
<summary>데모 앱 (배세은)</summary>
  ## 데모 앱 만들기 및 서버와 연결하기

## 1. MainActivity

![2023-04-06 (2).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/33e36d7b-6a0f-4538-9d10-50f800a71172/2023-04-06_(2).png)

### -영화 제목 입력 받기

### -서버에 요청 보내기

- 입력된 텍스트를  String형식으로 movietitle이라는 변수에 저장
- JSON형식으로 데이터를 만듦
- HttpURLConnection을 사용하여 서버에 POST요청

```kotlin
val movietitle = binding.movietitle.text.toString() // EditText에서 텍스트 가져오기

            try {
                var json = JSONObject()
                json.put("title", movietitle); // JSON형태로 변환하여 변수에 저장

                val conn = URL("http://10.30.116.62:8000/api/movies/").openConnection() as HttpURLConnection
                conn.doOutput = true
                conn.requestMethod = "POST" // POST방식으로 보냄
                conn.addRequestProperty("Content-Type", "application/json") // HTTP 요청 본문에 JSON데이터를 넣을 것이라는 것을 서버에 알림

								val output = conn.outputStream // 서버에 JSON 데이터 전송
								output.write(json.toString().toByteArray())
								output.flush()
								output.close()
```

### -서버에서 응답 받기

- JSON형식으로 응답을 받아옴
- 예시)

  {
  "title": ["영화1","영화2","영화3"]
  }    

- “title” key에 대한 value를 추출
- 각각의 영화 제목을 String으로 변환하여 리스트 형식으로 SubActivity에 보냄

```kotlin
conn.inputStream.use { `in` -> // 서버로부터 응답 받음
    ByteArrayOutputStream().use { out -> // 응답 데이터 받아옴
        val buf = ByteArray(1024 * 8)
        var length = 0
        while (`in`.read(buf).also { length = it } != -1) {
            out.write(buf, 0, length)
        }
        val response = String(out.toByteArray()) // 응답받은 데이터를 response 변수에 저장

				val jsonObject = JSONObject(response) 
				val titleList = jsonObject.getJSONArray("title") // 서버에서 전달받은 JSON 데이터에서 "title" key에 해당하는 value들 가져옴 
				val titles = ArrayList<String>() // 위 값들 추출하여 ArrayList에 추가
				for (i in 0 until titleList.length()){
				    val title = titleList.getString(i)
				    titles.add(title)
				}

				val titleListObj = TitleList(titles)
				
				val intent = Intent(this, SubActivity::class.java).apply { // SubActivity로 전환
				    putExtra("movieTitle", movietitle) // 검색한 영화 제목 SubActivity로 보내기
				    putExtra("titleList", titleListObj) // 응답받은 영화 제목들 SubActivity로 보내기
				}
```

## 2. SubActivity

MainActivity에서 받은 데이터 리사이클러뷰를 이용해 출력

![2023-04-07.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/a3615819-863f-4cae-9aa2-3e90419cde4e/2023-04-07.png)

SubActivity

제목 : 검색한 영화 제목

item : 검색 결과 영화 제목들

오른쪽 Recyclerview의 영화 제목이

item으로 하나씩 들어감

![2023-04-07 (1).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/051f3a77-7957-40a3-bb40-b35f960db4c2/2023-04-07_(1).png)

Recyclerview

# 결과물

</details>

<br>

### 2. Word2Vec을 이용한 추천시스템
**개발 기간: 2023년 월 일 요일 ~ 월 일 요일**
<details>
<summary>다음 영화 데이터 크롤링 (하은영)</summary>
  ## 다음 영화에서 영화 제목, 줄거리, 장르를 추출하여 가공

### 기능 설명 및 코드

1. **크롤링 (daum_movie_crawling.py)**

1-1. **영화 제목 크롤링**

- 존재하지 않는 페이지가 있을 수도 있으므로, head에서 따오기
    
    ![스크린샷 2023-05-10 오후 2.57.21.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/23190727-b717-4f1b-abd8-3d099327ca87/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.57.21.png)
    

```python
def crawling(start_code, finish_code):
	for i in range(finish_code, start_code, -1):
		movie_code = str(i)
    raw = requests.get("https://movie.daum.net/moviedb/main?movieId=" + movie_code)
    html = bs(raw.text, 'html.parser')

		# 영화 제목 수집
    title = html.find("head").find("title").text.replace(" | 다음영화", "")
    # 존재하지 않는 영화일 때 넘어가기
    if title == "다음영화":
        continue
```

1-2. **영화 줄거리, 장르 크롤링**

- 셀레니움 사용
- 줄거리와 다르게 장르는 같은 이름의 속성값이 많아서 CSS가 아닌 Xpath를 이용해서 찾음
    
    ![스크린샷 2023-05-09 오후 10.48.23.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/0151bdb3-68cf-4dda-badf-753c4cbefc49/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-05-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.48.23.png)
    

```python
		# 페이지가 완전히 로드되는 걸 기다리지 않게끔 웹 드라이버 설정
    caps = DesiredCapabilities().CHROME
    caps["pageLoadStrategy"] = "none"   # default: caps["pageLoadStrategy"] = "normal"

    options = Options()
    options.add_argument("--headless")  # 창을 띄우지 않게끔
    driver = webdriver.Chrome('chromedriver', options=options)
    driver.get("https://movie.daum.net/moviedb/main?movieId=" + movie_code)          

    # 영화 줄거리 수집
    try:
        raw_story = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#mainContent > div > div.box_detailinfo > div.contents > div.detail_basicinfo > div > div > div'))
        ).text
        # 해당 요소가 로딩될 때까지 최대 5초까지 대기
        # presence_of_element_located: 로딩된 페이지에 조건 요소가 있는지 확인
        story = raw_story.replace("\n", " ")
    except:   # 줄거리 정보가 없을 때
        story = ""

    # 영화 장르 수집
    try:
        genre = driver.find_element(By.XPATH, '//*[@id="mainContent"]//dt[contains(text(), "장르")]').find_element(By.XPATH, 'following-sibling::dd').text
    except:   # 장르 정보가 없을 때
        genre = ""
```

- 줄거리가 있는 영화들의 개수를 3만 개 이상으로 하기 위한 코드

```python
		# 줄거리가 없는 영화 제외했을 때의 영화 정보의 개수 정하기
    if len(data[-1]['story']) != 0:
        cnt += 1
        if cnt == 35000:
            return
```

1. **줄거리, 장르 가공 (processed_daum_movie.py)**

2-1. **줄거리 특수문자 제거 함수**

- 정규표현식을 사용하기 위해 re 모듈 사용

```python
# re.sub（정규 표현식, 치환 문자, 대상 문자열）
text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》‘’“”]', '', readData)
```

2-2. **줄거리 명사 추출 / 불용어 / 한 글자 제거 함수**

- 명사 추출을 위해 konlpy의 Okt 모듈 사용
- 불용어 텍스트 파일을 이용하여 줄거리에서 불용어 제거
- 한 글자는 의미 없는 경우가 많으므로, 한 글자로 이루어진 단어 제거

```python
okt = Okt()
nouns = okt.nouns(readData)  # 명사만 뽑아내기

# 텍스트 파일 열기
korean_stopwords_path = "./korean_stopwords.txt"
with open(korean_stopwords_path, encoding='utf-8') as f:
    stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]

# 불용어 및 한 글자 제거
remove_char = [x for x in nouns if (x not in stopwords) and (len(x) > 1)]
```

2-3. **줄거리, 장르를 단어로 나눠서 리스트에 넣기**

```python
processed_data = []
for d in data:
    # 줄거리 가공
    story_del = stopwords(d['story'])  # 명사 추출 + 불용어 및 한 글자 제거
    story_clean = cleanText(story_del)  # 특수문자 제거

    # story value값 리스트로 만들기
    if len(story_clean) == 0:
        story_final = []
    else:
        story_final = story_clean.split(" ")

    d['story'] = story_final

		# 장르 가공
    # genre value값 리스트로 만들기
    if len(d['genre']) == 0:
        genre_final = []
    else:
        genre_final = d['genre'].split("/")

    d['genre'] = genre_final

    # 줄거리 및 장르 저장
    processed_data.append(d)
```

1. **피클 저장**

```python
with open('daum_moive.pickle', 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
```

- 피클로 저장된 최종 형태:
    
    [{'title': '가디언즈 오브 갤럭시: Volume 3', 'story': ['가모라', '슬픔', '피터', '위기', '은하계', '동료', '위해', '다시', '한번', '가디언즈', '경우', '마지막', '미션', '이야기'], 'genre': ['액션', '어드벤처', 'SF']}, {'title': '그루지 2020', ···}
    

---

### 전체 코드

- **daum_movie_crawling.py**

```python
import requests   # HTTP 요청을 보내는 모듈
from bs4 import BeautifulSoup as bs   # 파싱 및 파싱한 문서에서 필요한 정보를 추출하는 모듈
from selenium import webdriver   # 웹 브라우저를 조작하는 모듈
from selenium.webdriver.common.by import By   # 웹 페이지에서 요소를 찾는 방법에 대한 모듈
from selenium.webdriver.support.ui import WebDriverWait   # 특정 조건이 충족될 때까지 대기하는 모듈
from selenium.webdriver.support import expected_conditions as EC   # 특정 조건이 충족될 때까지 대기하는 모듈에서 사용하는, 예상 조건에 대한 모듈
from selenium.webdriver.chrome.options import Options   # Chrome 브라우저 설정에 대한 모듈
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities   # 웹 드라이버 설정에 대한 모듈
import pickle   # 데이터를 파일로 저장하고 불러오는 모듈

def crawling(start_code, finish_code):
    try:
        global is_ok
        is_ok = False
        cnt = 0
        j = 0
        data = []  # 수집한 데이터를 저장할 리스트

        # 영화 코드 범위 지정
        for i in range(finish_code, start_code, -1):
            movie_code = str(i)
            raw = requests.get("https://movie.daum.net/moviedb/main?movieId=" + movie_code)
            html = bs(raw.text, 'html.parser')

            # 페이지가 완전히 로드되는 걸 기다리지 않게끔 웹 드라이버 설정
            caps = DesiredCapabilities().CHROME
            caps["pageLoadStrategy"] = "none"   # default: caps["pageLoadStrategy"] = "normal"

            options = Options()
            options.add_argument("--headless")  # 창을 띄우지 않게끔
            driver = webdriver.Chrome('chromedriver', options=options)
            driver.get("https://movie.daum.net/moviedb/main?movieId=" + movie_code)

            # 영화 제목 수집
            title = html.find("head").find("title").text.replace(" | 다음영화", "")
            # 존재하지 않는 영화일 때 넘어가기
            if title == "다음영화":
                continue

            # 영화 줄거리 수집
            try:
                raw_story = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '#mainContent > div > div.box_detailinfo > div.contents > div.detail_basicinfo > div > div > div'))
                ).text
                # 해당 요소가 로딩될 때까지 최대 5초까지 대기
                # presence_of_element_located: 로딩된 페이지에 조건 요소가 있는지 확인
                story = raw_story.replace("\n", " ")
            except:   # 줄거리 정보가 없을 때
                story = ""

            # 영화 장르 수집
            try:
                genre = driver.find_element(By.XPATH, '//*[@id="mainContent"]//dt[contains(text(), "장르")]').find_element(By.XPATH, 'following-sibling::dd').text
            except:   # 장르 정보가 없을 때
                genre = ""

            # 데이터를 리스트에 추가
            data.append({'title': title, 'story': story, 'genre': genre})

						# 저장
            with open('daum_moive.pickle', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

            # 창 닫기
            driver.quit()

            # 줄거리가 없는 영화 제외했을 때의 영화 정보의 개수 정하기
            if len(data[-1]['story']) != 0:
                cnt += 1
                if cnt == 35000:
                    return

            # 출력용 (지워도 무방)
            is_ok = True
            print("=" * 50)
            print("제목:", title)
            print("줄거리: ", story)
            print("장르: ", genre)
            print("-" * 50)
            if is_ok == True:
                j = j + 1
            print((finish_code - i) + 1, "번째 영화 체크 중", j, "개의 영화 정보 저장 완료 (줄거리가 존재하는 영화 정보는", cnt, "개)")
            print(finish_code - start_code, "개 중에", (i - start_code) - 1, "개 남음")

    except:
        print((finish_code - i) + 1, "번째 영화 체크 중 error")

crawling(24157, 129157)   # 총 105,000만 개
```

- **processed_daum_movie.py**

```python
import re  # 정규표현식을 위한 모듈
import pickle
from konlpy.tag import Okt  # 한국어 자연어 처리 모듈

# 특수문자 제거 위한 함수
def cleanText(readData):
    # 줄거리에 포함되어 있는 특수문자 제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》‘’“”]', '',
                  readData)  # re.sub（정규 표현식, 치환 문자, 대상 문자열）
    return text

def stopwords(readData):
    okt = Okt()
    nouns = okt.nouns(readData)  # 명사만 뽑아내기

    # 텍스트 파일 열기
    korean_stopwords_path = "./korean_stopwords.txt"
    with open(korean_stopwords_path, encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]

    # 불용어 및 한 글자 제거
    remove_char = [x for x in nouns if (x not in stopwords) and (len(x) > 1)]

    # 문자열로 만들기
    text = ' '.join(remove_char)

    return text

def processedData():
    with open('daum_moive.pickle', 'rb') as f:
        data = pickle.load(f)

    processed_data = []
    for d in data:
        # 줄거리 가공
        story_del = stopwords(d['story'])  # 명사 추출 + 불용어 및 한 글자 제거
        story_clean = cleanText(story_del)  # 특수문자 제거

        # story value값 리스트로 만들기
        if len(story_clean) == 0:
            story_final = []
        else:
            story_final = story_clean.split(" ")

        d['story'] = story_final

				# 장르 가공
        # genre value값 리스트로 만들기
        if len(d['genre']) == 0:
            genre_final = []
        else:
            genre_final = d['genre'].split("/")

        d['genre'] = genre_final

        # 줄거리 및 장르 저장
        processed_data.append(d)

    with open("processed_daum_movie.pickle", "wb") as f:
        pickle.dump(processed_data, f)

processedData()
```
</details>

<details>
<summary>서버 (배진우)</summary>
  - 방화벽의 대한 접근 허용
    - MySQL은 보통 3306의 포트를 사용하며 , 외부에서 3000포트로 접근을 허용, SSH를 허용했으나 실패, ICMP를 허용해 ping을 확인.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/b77f94fa-16c7-488e-9889-5cdbf4550880/Untitled.png)

고정 IP주소를 부여해 코드 변화 없음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/11369fa6-a94a-4ddb-8675-a42586248376/Untitled.png)

MySQL 콘솔 접속후 새로운 사용자를 생성하고 액세스 권한 부여  및 데이터베이스 생성

```jsx
CREATE USER '사용자_이름'@'locaIhost' IDENTIFIED BY '비밀번호';
GRANT ALL PRIVILEGES ON *.* TO '사용자_이름'@'Iocalhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
CREATE DATABASE 데이터베이스_이름;
```

MySQL 로그인 후 데이터베이스 선택

```jsx
mysql -u [사용자명] -p
USE [데이터베이스명];
CREATE TABLE [테이블명] (
  id INT AUTO_INCREMENT PRIMARY KEY, // id 자동 증가되는 기본 키(PK) 열
  title VARCHAR(255), // 최대 255글자
  plot TEXT // 긴 줄거리
);
```

- 데이터베이스에 데이터 저장

```python
import pickle
import MySQLdb

# 데이터베이스 연결 설정
db = MySQLdb.connect(host='localhost', user='bgw4399', password='qowlsdn4399', database='word2vec')
cursor = db.cursor()

# pickle 파일에서 데이터 추출
with open('processed_daum_movie_final.pickle', 'rb') as file:
    data = pickle.load(file)

# 데이터베이스에 전송할 SQL 쿼리 작성

query = "INSERT INTO move (title, plot) VALUES (%S, %s)"

# 데이터베이스에 데이터 전송
for item in data:
    cursor.execute(query, (item['titIe'], " ".join(item['story'])))
# 변경 사항 커밋
db.commit()

# 연결 종료
db.close()
```

- 바뀐 [settings.py](http://settings.py) 부분

모든 사람들의 ip를 허용 (보안이 취약하다)

```jsx
ALLOWED_HOST = ['*']
```

기존에 만들어놓은 계정을 등록

```jsx
DATABASS = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'word2vec',
        'USER': 'bgw4399',
        'PASSWORD': 'qowlsdn4399',
        'HOST': 'localhost',
        'PORT': '3300',
    }
}
```

문제점

ssh를 이용한 연결 오류(이유 찾지 못함..)

```python
ssh: connect to host ec2-3-105-118-201.ap-southeast-2.compute.amazonaws.com port 202: Connection timed out
```
</details>

<details>
<summary>Word2Vec (최선우)</summary>
  ## 워드 임베딩: 단어를 (밀집 표현으로 나타낸) 벡터로 표현하는 방법 (밀집 표현)

- LSA, Word2Vec, FastText, Glove 등이 있음

## Word2Vec

- 단어 벡터의 값이 단어의 의미를 수치화한 것
    
    → 벡터 간 유의미한 유사도 반영 (단어 벡터의 값이 비슷하면 의미가 유사한 것)
    
    → 단어의 의미를 수치화
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/bfc22de0-5027-4388-8733-3b6377a1295b/Untitled.png)

### 분산 표현: 단어의 의미를 다차원 공간에 벡터화하는 방법(공간의 크기를 100차원으로 정하면 단어의 의미를 100차원의 공간을 이용하여 나타냄)

- 의미
    - Word2Vec에서 단어를 벡터로 나타내는 방법
    - 단어의 의미를 여러 차원에다가 분산하여 표현
- **가정: “비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다” (분포 가설)**
    - ex) “동물원에 있는 ~는 정말 귀엽다”에서 나올 단어들은 강아지, 고양이 등이 있다. → 비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가지므로 강아지와 고양이는 분포가설에 의해 비슷한 의미를 가지면 비슷한 벡터의 형태를 가진다
- 방법
    1. 분포 가설을 이용하여 텍스트를 학습
    2. 단어의 의미를 벡터의 여러 차원에 분산하여 표현
- 장점
    - 벡터 간 유의미한 유사도 반영 가능
    - 저차원으로 단어 벡터 표현 가능 (희소표현에 비해)

### CBOW(Continuous Bag of Words): 주변 단어들로부터 중심 단어를 예측하는 방법

ex) 예를 들어, "I love ___"라는 문장이 주어졌을 때, "I love pizza"라는 답을 출력

이때, "pizza"가 중심 단어가 되고, "I", "love"가 주변 단어

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/91e34ce3-4f6f-4b26-a296-491dbb7df6fb/Untitled.png)

인공신경망의 입력은 원핫 벡터

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/569091d9-1d04-4e92-ae8a-ab70fd290007/Untitled.png)

lookup table이 임베딩의 결과가 됨 (또는 w와 w`을 모두 이용하여) 

w와 w`은 처음에 랜덤 값을 갖고 훈련시킴 (w와 w`는 완전 다른 행렬, w, w`을 잘 훈련시키자)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/d8f1218c-8f34-440d-a760-695702f628f4/Untitled.png)

W와 곱해진 원핫벡터(입력벡터)들은 평균으로 합쳐져서 M이 됨

M은 W`과 곱해지고 softmax 함수를 지나면서 y^(추정값)을 도출

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/46ff05c2-7da2-4d0b-b5dd-b3277e9cc144/Untitled.png)

크로스 엔트로피를 이용해 원래 레이블을 이용하여 두 값의 오차를 줄여가며 학습

### Skip-gram: CBOW와 반대로 중심 단어로부터 주변 단어들을 예측하는 방법

ex)예를 들어, "pizza"가 주어졌을 때, "I", "love"를 출력

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/f06d6daa-86a6-4997-945a-70e460093dd7/Untitled.png)

## 영화 유사도에 적용

### → 단어끼리의 유사도가 아닌 줄거리끼리의 유사도 비교가 필요

### → 각 단어 벡터를 모두 합치는 방안 선택

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/bfc22de0-5027-4388-8733-3b6377a1295b/Untitled.png)

- 벡터라고 생각했을 때 합 벡터가 비슷하면 비슷한 영화라고 생각

### → 벡터를 모두 합쳤으므로 크기는 중요하지않기 때문에 각도만 구하는 코사인 유사도를 이용하여 유사도 구함

### Cbow와 Skip-gram

skip-gram

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/3c075e6d-6d8f-435e-ad83-6426f3b29fda/Untitled.png)

cbow

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/ad93aab0-ebc8-4407-af64-f8f68ab3bb6c/Untitled.png)

```python
import pickle
from gensim.models.word2vec import Word2Vec
from cosine_sim import cosine_sim

class Word2vec_movie:

    def __init__(self):

        self.model_word2vec = Word2Vec.load("word2vec_final_skip")
        
        # DB에서 가져와야됨
        with open('data/processed_daum_movie_final.pickle', 'rb') as f:
            data = pickle.load(f)

        self.title = [i['title'] for i in data]
        self.story = [i['story'] for i in data]

    def cosine_sim_calc(self, movie_title):
        
        movie_title_vector = 0

        if movie_title in self.title:
            for word in self.story[self.title.index(movie_title)]:
                if word in self.model_word2vec.wv:
                    movie_title_vector += self.model_word2vec.wv[word]
            #print(movie_title_vector)
            sim_movie = []
            for index in range(len(self.story)):
                if not self.story[index] :
                    continue
                # print(self.story[index])
                movie_diff_vector = 0
                for word in self.story[index]:
                    if word in self.model_word2vec.wv:
                        movie_diff_vector += self.model_word2vec.wv[word]
                #print(cosine_sim().cosine_sim_cal(movie_title_vector, movie_diff_vector))
                cosine_sim_value = cosine_sim().cosine_sim_cal(movie_title_vector, movie_diff_vector)
                # print(cosine_sim_value)
                if cosine_sim_value > 0.5 :
                    tmp = []
                    tmp.append(float(cosine_sim_value))
                    tmp.append(self.title[index])
                    tmp.append(movie_diff_vector)
                    sim_movie.append(tmp)
            return sorted(sim_movie, key=lambda x:-x[0])[1:8]

                

if __name__ == '__main__' :
    movie = Word2vec_movie()
    print(len(movie.title))
    print(len(movie.story))
    for i, j in zip(movie.title, movie.story):
        print(i, j)
    print(movie.title.index("판문점"))
    print(movie.model_word2vec.wv["동료"])
    a=movie.cosine_sim_calc("황혼의 검객")
    # with open("sample.pickle", "wb") as fw:
    #     pickle.dump(a, fw)
    print(a)
```
</details>

<details>
<summary>데모 앱 (배세은)</summary>
  movie.html

```html
<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="index.css" />
    <title>영화 추천 서비스</title>
  </head>
  <body>
    <div class="title">
      <div style="font-size: 40px">비슷한 영화 추천 서비스</div>
    </div>
    <p style="font-size: 23px" ; align="center" ;>
      마음에 들었던 영화 제목을 입력해주세요
      <div style="text-align: center">
        <form id="search-form">
          <input type="text" name="movie" size="40" /><br><br>
          <input type="submit" style="width: 40pt; height: 22pt" value="제출">
        </form>
      </div>
    </p>
    // jQuery는 HTML 이벤트 처리와 같은 기능을 부여하는 오픈소스 기반의 자바스크립트 라이브러리
		// CDN : 웹 주소, 빠르게 JQuery를 로드할 수 있음
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script> // jQuery 라이브러리를 CDN에서 가져오기
    <script>
      $(document).ready(function() { // 문서가 준비되면 매개변수로 넣은 콜백 함수를 실행
        $('#search-form').submit(function(event) { // 검색 폼에서 버튼이 클릭되는 이벤트 발생시 호출됨
          event.preventDefault(); // 검색 버튼 클릭시 기본 동작(새로고침)을 중지
          const movieTitle = $('input[name="movie"]').val(); // 검색어를 가져와 'movieTitle'변수에 할당
          $.ajax({ // AJAX 요청 보냄, 동적인 웹 페이지를 만들기 위한 기법
            url: 'http://3.105.118.201:8000/api/movies/', // 요청할 서버 URL 지정
            type: 'GET', // 요청 방식을 GET방식으로 지정
            dataType: 'json', // 받아올 데이터의 자료형 지정
            data: { movie: movieTitle }, // 딕셔너리 형태 movie 키 값에 입력받은 movieTitle 변수 값 value로 전
            success: function(response) { //요청이 성공한 경우 실행할 콜백 함수
              const movieList = response.movie; // 응답 데이터에서 영화 목록 가져오기
              let movieListHtml = ""; // 생성된 영화 목록 출력 위한 HTML 문자열 저장, 이전 검색 결과에 남아있는 HTML 문자열 초기화 
              for (let i = 0; i < movieList.length; i++) {
                movieListHtml += `<div>${i + 1}. ${movieList[i]}</div>`; // 영화 목록을 돌며 배열에 저장된 목록을 div 태그 형태로 변환 후 변수에 추가 
              }
              localStorage.setItem("movieTitle", movieTitle); // 페이지에서 전달받은 영화 제목 localStorage에 저장
              localStorage.setItem("movieListHtml", movieListHtml); // 검색 결과 HTML 문자열 localStorage에 저장
              const url = `./after.html`; // after.html 페이지 주소 생성
              window.location.href = url; // 페이지 이동
            },
            error: function(error) { // 요청 실패시 error 콜백 함수 호출
              console.log(error);
            }
          });
        });
      });
    </script>    
  </body>
</html>
```

after.html

```html
<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="index.css" />
    <title>영화 추천 서비스</title>
  </head>
  <body>
    <div class="title">
      <div style="font-size: 40px">영화를 검색한 결과입니다!</div>
      <div style="padding: 30px"></div>
      <div style="font-size: 20px">
        검색한 영화 :
        <span id="search"></span> <!--movieTitle 출력
      </div>
      <div style="padding: 10px"></div>
      <div style="font-size: 20px">
        <span id="result"></span>
      </div>
    </div>
    <div id="movie-list" style="margin-top: 30px; font-size: 23px"></div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
      $(document).ready(function () { // 페이지 로딩이 끝나면 실행됨
        const params = new URLSearchParams(window.location.search); // URL 파라미터 가져오기
        const movieTitle = localStorage.getItem("movieTitle"); // localStorage를 이용하여 movie.html에서 저장한 movieTitle 가져오기
        $("#search").text(movieTitle); // id가 search인 html요소에 출력
        const movieListHtml = localStorage.getItem("movieListHtml"); // localStorage를 이용하여 movie.html에서 저장한 영화제목들 가져오기
        $("#result").html(movieListHtml); // id가 result인 html요소에 결과값들 출
      });
    </script>
  </body>
</html>
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/0ede72c8-6b30-481d-9532-ef8096b950c2/Untitled.png)

![2023-05-12.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/de6a9e88-4865-45d4-a901-38d34aebbc8a/583dc7f3-6661-485f-a5d8-47411dfa322c/2023-05-12.png)

### **t-SNE**

- 매니폴드 알고리즘 중 하나
- 비선형적인 고차원 데이터를 저차원으로 투영하는 차원 축소 방법으로, 고차원 데이터의 간의 군집 간 거리 관계를 보존
- t-SNE 알고리즘에서 데이터 간 유사성 측정 방법
    1. 고차원 데이터에서 코사인 유사도 등을 이용해 거리 측정.
    2. 고차원 데이터/저차원 데이터의 각각의 유사성을 확률 분포로 변환 -> 고차원 데이터는 정규 분포, 저차원 데이터는 t-분포(자유도: 1)를 사용하여 데이터의 확률 분포를 모델링.
    3. 고차원 데이터와 저차원 데이터 간의 확률 분포의 차이를 최소화하는 방식으로 저차원 데이터를 학습. 즉, 고차원 데이터와 저차원 데이터의 유사성을 최대한 유지하는 방식.
</details>

<br>

### 3. bert를 이용한 유사한 음악 동영상 추천 <a href="https://docs.google.com/presentation/d/1s2oMH_I8BhrLWoHcPZ7lIQ9P8rGuU9aZtOE5QWU9rIo/edit?pli=1#slide=id.p1">(Presentation)</a>
**개발 기간: 2023년 월 일 요일 ~ 월 일 요일**

<br>

### 4.WordDictionary <a href="https://docs.google.com/presentation/d/1h7AJ3oC5FaML3LA510PhZqCrAyvJwMBJZ9be1RgL_80/edit#slide=id.g2381826a579_2_75">(Presentation)</a>
**개발 기간: 2023년 월 일 수요일 ~ 월 일 요일**





<!--
## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/enbrainai/enbrain-kookmin.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/enbrainai/enbrain-kookmin/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
-->
