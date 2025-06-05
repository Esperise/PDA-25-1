import re
import urllib.request
from urllib.parse import urlparse,parse_qs
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import datetime
#
# 사용자 에이전트 정의
headers = {
    # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
                  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
}
crawling_succ_count=0
crawling_fail_count=0
total_crawling_comment=0

# 뉴스 URL 수집 함수
def news_url(query, date1) -> tuple[list, list, datetime.date]:
    page = 1
    url_list = []
    title_list = []
    link_set = set()
    # date1= datetime.date(2024, 12, 4)
    one_day = datetime.timedelta(days=1)
    date2= date1+ one_day
    date1_str= str(date1).replace("-", ".")
    date2_str=str(date2).replace("-", ".")
    date1_str_not_separated= str(date1).replace("-", "")
    date2_str_not_separated= str(date2).replace("-", "")



    while page <= 150:
        #url에 대해
        # url = (
        #         "https://m.search.naver.com/search.naver?where=m_news&sm=tab_pge&query="
        #         + query +
        #         "&sort=0&photo=0&field=0&pd=1&ds=&de=&cluster_rank=129&mynews=0&office_type=0"
        #         "&office_section_code=0&news_office_checked=&nso=so:r,p:1w,a:all&start=" + str(page) #start은 최소 1 최대 1000임
        # )
        url= (
                "https://m.search.naver.com/search.naver?ssc=tab.m_news.all&query="
                + query + "&sm=mtb_opt&sort=0&photo=0&field=0&pd=3&ds="
                + date1_str + "&de=" + date2_str + "&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Ar%2Cp%3Afrom"
                + date1_str_not_separated + "to" + date1_str_not_separated + "&is_sug_officeid=0&office_category=0&service_area=0" + str(page)
        )
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'referer':url
        }

        response = requests.get(url, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, "lxml")
        # atags = soup.select('.news_tit')
        #위에 atags=soup.select('.news_tit')은 위 response로 받아온 html에서 뉴스 링크가 있는 a태그에 있는 클래스였는데 4/28일 확인해보니까
        #이부분이 암호화 되어서 이 코드가 더 이상 유효하지 않는거 같습니다.
        atags = soup.select('._AuHeQ05X7PwSlhb6H2B') #처럼 암호화 된 부분을 확인해서 넣으면 작동하긴 하는데 이게 나중에 어떻게 변할지 모르겠습니다.

        # for a_tag in soup.find_all('a', href=True):
        #     href= a_tag['href']
        #     if href.startswith('https://n.news.naver.com/'):
        #         link_set.add(href)
        # atags= link_list= list(link_set)
        print(atags)

        for i in atags:
            if "https://n.news.naver.com/" in i['href']:
                url_list.append(i['href'])
                news_title_name=get_title_name(i)
                title_list.append(news_title_name)

        if len(atags) < 15:
            break

        # page += 15
        page+= 150
        time.sleep(0.5)  # 너무 빠르게 요청하지 않도록 sleep

    return url_list ,title_list, date1  #zip으로 묶거나 받는 변수를 2개 나 튜플로 해야함




# 뉴스 연령대 데이터가 필요하면 여기서 더 추가해서 사용 가능함
# def getAge(url_list,headers,url_list_num,oid_1,oid_2):
#     print("asd")
#     getAge_url=url_list[url_list_num]
#     response = requests.get(getAge_url, headers=headers)
#     html=response.text
#     soup = BeautifulSoup(html, "lxml")
#     news_id=f'news{oid_2},{oid_1}'
#     url_api= 'https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json?ticket=news&templateId=view_society&pool=cbox5&_wr&_callback=jQuery11240673401066245984_1638166575411&lang=ko&country=KR&objectId=' + news_id + '&categoryId=&pageSize=10&indexSize=10&groupId=&listType=OBJECT&pageType=more&page=1&initialize=true&userType=&useAltSort=true&replyPageSize=20&sort=favorite&includeAllStatus=true&_=1638166575413'
#     getAge_req=requests.get(url_api,headers=headers)
#     getAge_json= json.loads(getAge_req.text[getAge_req.text.find('{'):-2])
#
#     gender_male = getAge_json['result']['graph']['gender']['male']
#     gender_female = getAge_json['result']['graph']['gender']['female']
#
#     ## 연령 통게정보 가져오기
#     ages_group_10 = getAge_json['result']['graph']['old'][0]['value']
#     ages_group_20 = getAge_json['result']['graph']['old'][1]['value']
#     ages_group_30 = getAge_json['result']['graph']['old'][2]['value']
#     ages_group_40 = getAge_json['result']['graph']['old'][3]['value']
#     ages_group_50 = getAge_json['result']['graph']['old'][4]['value']
#     ages_group_60 = getAge_json['result']['graph']['old'][5]['value']
#     return gender_male, gender_female, ages_group_10, ages_group_20, ages_group_30, ages_group_40, ages_group_50, ages_group_60

#oid_2를 매개변수로 넣으면 언론사 이름이 string으로 return되는 함수입니다.


#언론사 id(oid_2)를 언론사 이름으로 바꾸는 함수  ex) "023" -> "연합뉴스"
# html요청 없이 따로 딕셔너리 형태로 만들어서 사용하면  더 빨라지는데 언론사 수가 생각보다 많아서 그냥 이렇게 했습니다.
def get_press_name(press_id):
    url_company='https://news.naver.com/main/officeList.naver'
    html_company = urllib.request.urlopen(url_company).read()
    soup_company = BeautifulSoup(html_company, "lxml")
    title_company=soup_company.find_all(class_='list_press nclicks(\'rig.renws2pname\')')
    for i in title_company:
        parts= urlparse(i.attrs['href'])
        if parse_qs(parts.query)['officeId'][0]==press_id:
            news_name = i.text.strip()
            return news_name


# 뉴스 이름을 html에서 가져오는 함수- news_url 함수 안에서 사용해서 별도의 html요청 없이 링크를 가져오려고 하는 html에서 추출
def get_title_name(atag):
    news_title_name=atag.find('span').text.strip()
    print(news_title_name) #가져 오는 뉴스 이름 확인용
    return news_title_name



# 댓글 수집 함수
def comment(url_list,news_title_list, news_date):
    global crawling_succ_count
    global crawling_fail_count
    global total_crawling_comment
    url_total_num= len(url_list)
    url_list_num=0
    news_title_list_num=0
    total_comment = [];user_nickname=[];comments=[];userIdNo=[];visible=[];replyLevel=[];regTime=[];parentCommentNo=[];sympathyCount=[];antipathyCount=[];replyAllCount=[]
    # following=[]
    for url_ex in url_list:
        url = url_ex.split('?')[0]
        oid_1 = url.split('/')[-1]
        oid_2 = url.split('/')[-2]
        print(news_title_list[news_title_list_num],url_list[url_list_num])
        i = 1



        # getAge(url_list,headers,url_list_num,oid_1,oid_2)


        while True:
            params = {
                'ticket': 'news',
                'templateId': 'default_society',
                'pool': 'cbox5',
                'lang': 'ko',
                'country': 'KR',
                'objectId': f'news{oid_2},{oid_1}',
                'pageSize': '100',
                'indexSize': '10',
                'page': str(i),
                'currentPage': '0',
                'moreParam.direction': 'next',
                'moreParam.prev': '',
                'moreParam.next': '',
                'followSize': '100',
                'includeAllStatus': 'true',
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'referer': url_ex
            }
            response = requests.get(
                'https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json',
                params=params,
                headers=headers
            )

            response.encoding = "UTF-8-sig"
            res = response.text.replace("_callback(", "")[:-2]


            try:
                temp = json.loads(res)
                comment_list = temp['result'].get('commentList', [])
                total_crawling_comment+=len(comment_list)
                if not comment_list:
                    print("comment 가져오기 실패")
                    crawling_fail_count+=1
                    log_text(
                        news_title_list[news_title_list_num],
                        url_list[url_list_num],
                        news_date,crawling_succ_count,
                        crawling_fail_count,
                        total_crawling_comment,
                        2)
                    url_list_num += 1
                    news_title_list_num += 1
                    break
                #아래는 가져온 comment_list( 위에서 요청한 json 내용 으로 'contents': '탄해반대집회는~~' 형태로 존재)

                if ((not user_nickname) and (not comments)):
                    user_nickname = [k['userName'] for k in comment_list]  # json 내용에서 userName에 해당하는 부분을 가져와 리스트로 저장시킴
                    comments = [c['contents'] for c in comment_list]  # json에서 contents에 해당하는 부분을 가져와 리스트로 저장시킴
                    # hiddenByCleanbot= [a['hiddenByCleanbot'] for a in comment_list]
                    userIdNo=[b['userIdNo'] for b in comment_list]
                    visible= [c1['visible'] for c1 in comment_list]
                    replyLevel=[d['replyLevel'] for d in comment_list]
                    regTime=[e['regTime'] for e in comment_list]
                    parentCommentNo=[f['parentCommentNo'] for f in comment_list]
                    sympathyCount= [g['sympathyCount'] for g in comment_list]
                    antipathyCount=[h['antipathyCount'] for h in comment_list]
                    # following=[p['following'] for p in comment_list]
                    replyAllCount=[r['replyAllCount'] for r in comment_list]



                else:
                    user_nickname.extend([k['userName'] for k in comment_list])
                    comments.extend([c['contents'] for c in comment_list])
                    # hiddenByCleanbot.extend= [a['hiddenByCleanbot'] for a in comment_list]
                    userIdNo.extend([b['userIdNo'] for b in comment_list])
                    visible.extend([c['visible'] for c in comment_list])
                    replyLevel.extend([d['replyLevel'] for d in comment_list])
                    regTime.extend([e['regTime'] for e in comment_list])
                    parentCommentNo.extend([f['parentCommentNo'] for f in comment_list])
                    sympathyCount.extend ([g['sympathyCount'] for g in comment_list])
                    antipathyCount.extend([h['antipathyCount'] for h in comment_list])
                    # following.extend([p['following'] for p in comment_list])
                    replyAllCount.extend(r['replyAllCount'] for r in comment_list)


                if len(comment_list) < 100:
                    #total_comment.extend(comments) #없어도 되는 코드 gptexample에 있던 코드
                    comment_list_sum=list(
                        zip(
                            user_nickname,
                            userIdNo,
                            comments,
                            visible,
                            replyLevel,
                            regTime,
                            parentCommentNo,
                            sympathyCount,
                            antipathyCount,
                            replyAllCount
                        )
                    ) #df로 저장하기 위해 묶어줌
                    col=['user_nickname','userIdNo','comments','visible','replyLevel','regTime','parentCommentNo','sympathyCount','antipathyCount','replyAllCount']
                    df=pd.DataFrame(comment_list_sum, columns=col) # columns를 작성자, 내용 으로 가지는 df생성
                    #파일을 저장하기 위한 경로 이름 생성, 파일 이름 생성
                    news_name=get_press_name(oid_2) #언론사 id를 언론사 이름 str로 바꾸는 함수
                    folder_path= f'../src/news_comments/{news_name}'
                    # folder_path=f'./{news_name}'
                    os.makedirs(folder_path, exist_ok=True) #이 py가 있는 경로에 "언론사이름"을 이름으로 하는 폴더 생성, 존재시 넘어감
                    # print(news_title_list_num)

                    if news_title_list_num >= len(news_title_list):
                        print(f"news_title_list_num{news_title_list_num}이 news_title_list {len(news_title_list)}길이를 초과했습니다.")
                        url_list_num += 1
                        news_title_list_num += 1
                        # url_list_num = 0
                        # news_title_list_num = 0
                        break
                    newstitle_regex= re.sub(r'[\\/*?:"<>|]', '_',news_title_list[news_title_list_num])
                    file_name= f"{news_date}_{news_name}_{newstitle_regex}.csv"

                    #파일 이름을 언론사_뉴스제목 으로 정의
                    if os.path.exists(f"{folder_path}/{file_name}"): #같은 이름 존재시 break
                        print(f"{folder_path}/{file_name} 이미 존재함. ")
                        log_text(
                            news_title_list[news_title_list_num],
                            url_list[url_list_num],
                            news_date,
                            crawling_succ_count,
                            crawling_fail_count,
                            total_crawling_comment,
                            3
                        )
                        url_list_num += 1
                        news_title_list_num += 1
                        continue;
                    df.to_csv(f"{folder_path}/{file_name}", index=False) #엑셀파일로 저장 다른 형식으로 바꾸고자 하면  .xlsx와 이 메소드 바꾸면 됨.
                    comment_list_sum = []
                    comments=[]
                    user_nickname= []
                    # hiddenByCleanbot= []
                    userIdNo=[];visible= []
                    replyLevel=[];regTime=[];replyAllCount=[];
                    parentCommentNo=[];sympathyCount=[];antipathyCount=[];following=[]

                    print(f"\n{news_date}: {news_title_list_num+1}/{url_total_num}")
                    crawling_succ_count+=1
                    print(f'성공: {crawling_succ_count} 실패: {crawling_fail_count}')
                    log_text(
                        news_title_list[news_title_list_num],
                        url_list[url_list_num],
                        news_date,
                        crawling_succ_count,
                        crawling_fail_count,
                        total_crawling_comment,
                        1
                        )
                    url_list_num += 1
                    news_title_list_num += 1
                    break
                else:
                    i += 1 # params= ..., 'page'=str(i),...에서  pageSize=100은 한 페이지당 불러올 댓글 최대가 100이기 떄문에 다음페이지로
                    #넘겨야함 즉 i 값 증가시켜서 다음 페이지 댓글 최대 100개를 불러옴
                    time.sleep(0.3)
            except Exception as e:
                print(f"[에러 발생] {url_ex}: {e}")
                break
            finally:
                print('|',end="")

    return total_comment #필요 없음
def log_text(news_url,news_title,news_date,succ_count,fail_count,total_crawling_comment,current_status):
    if (current_status==1):
        with open('../src/news_comments/info.txt', 'a', encoding='utf-8') as file:
            file.write(f'성공: {news_url} , {news_title} , {news_date} , 성공: {succ_count} 실패: {fail_count} 총 댓글 개수: {total_crawling_comment}\n')
    elif  (current_status==2):
        with open('news_comments/info.txt', 'a', encoding='utf-8') as file:
            file.write( f'실패: {news_url} , {news_title} , {news_date} , 성공: {succ_count} 실패: {fail_count} 총 댓글 개수: {total_crawling_comment}\n')
    elif(current_status==3):
        with open('./news_comments/info.txt', 'a', encoding='utf-8') as file:
            file.write( f'중복: {news_url} , {news_title} , {news_date} , 성공: {succ_count} 실패: {fail_count} 총 댓글 개수: {total_crawling_comment}\n')

query = "윤석열 탄핵"  # 원하는 검색어
encoded_query = urllib.parse.quote(query) #url에 쿼리스트링에 한글 사용하려면 인코딩 필수
date= datetime.date(2024, 12, 4)
while(date< datetime.date(2025, 4, 15)):
    news_links, news_title ,news_date= news_url(encoded_query, date)
    print(f"{date} 뉴스 url 수집 완료")
    comment(news_links, news_title, news_date)
    print(news_links)
    date= date+ datetime.timedelta(days=1)




# print(f"{len(news_links)}개의 뉴스 링크 수집 완료")
# print(news_links)

# all_comments = comment(news_links,news_title, news_date)
# print(f"{len(all_comments)}개의 댓글 수집 완료")
