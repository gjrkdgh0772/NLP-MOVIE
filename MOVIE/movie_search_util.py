import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
import requests

import warnings
warnings.simplefilter('ignore')



mldf = pd.read_csv("./dataset/MOVIE_MERGI.csv")
C = mldf['vote_average'].mean()
m = mldf['vote_count'].quantile(0.95)

#---------------------------------------------------------
# 2. weight ranking based 장르 검색
#---------------------------------------------------------
def my_calc_wr_def(mdf):
    R = mdf['vote_average']
    v = mdf['vote_count']
    WR = (v / (v+m)) * R + (m/ (v+m)) *C
    return WR

def my_search_by_genres(search_genres ='Family', percnet=0.95):
    mldf['wr'] = mldf.apply(my_calc_wr_def, axis=1)
    df5 = mldf[mldf['vote_count'] > m]

    genres_dict = {'액션': 'Action',
                   '어드벤쳐': 'Adventure',
                   '애니메이션': 'Animation',
                   '코미디': 'Comedy',
                   '범죄': 'Crime',
                   '다큐멘터리': 'Documentary',
                   '드라마': 'Drama',
                   '가족': 'Family',
                   '판타지': 'Fantasy',
                   '외국': 'Foreign',
                   '역사': 'History',
                   '호러': 'Horror',
                   '음악': 'Music',
                   '미스터리': 'Mystery',
                   '로맨스': 'Romance',
                   'SF': 'Science Fiction',
                   'TV 영화': 'TV Movie',
                   '스릴러': 'Thriller',
                   '전쟁': 'War',
                   '서부': 'Western'}

    try :
        if not search_genres.encode().isalpha():
            val = genres_dict[search_genres]
            search_genres = val
    except :
        search_genres = 'Family'
    
        
    df5 = df5[df5['genres'].str.contains(search_genres, case = False)]
    df5 = df5[['title','poster_path2', 'overview','title2','overview2','wr']].sort_values('wr', ascending=False).head()
    return df5.values

#---------------------------------------------------------
# 3. Review based  리뷰 검색
#---------------------------------------------------------
def my_search_by_review(title = "Toy Story", topn=10):
    s = mldf['title']
    title_s = pd.Series(s.index, index=s.values)  # 값 <--> 인덱스 서로 자리 변경

    if title.encode().isalpha():
        title_s = mldf[mldf['title'].str.contains(title, case=False)]['title']
    else:
        title_s = mldf[mldf['title2'].str.contains(title, case=False)]['title']

    if len(title_s) > 0:
        idx = title_s[:1].index
    else:
        idx = 0
    tfidf = TfidfVectorizer(stop_words='english')  # , max_df=0.8, min_df=0.2)  ngram_range=(1, 2)
    tfidf_matrix = tfidf.fit_transform(mldf['view_tag'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx_list = pd.Series(cos_sim[idx].reshape(-1)).sort_values(ascending = False).index[1:topn+1] # 0번재는 본인. 1~10번째
    df5 = mldf.loc[idx_list, ['title','poster_path2', 'overview','title2','overview2']].head()
    return df5.values

#---------------------------------------------------------
# 4. Actor, Driect... based 메타검색
#---------------------------------------------------------
def my_search_by_meta(title = "Toy Story", topn=10):
    s = mldf['title']
    title_s = pd.Series(s.index, index=s.values)  # 값 <--> 인덱스 서로 자리 변경
    if title.encode().isalpha():
        title_s = mldf[mldf['title'].str.contains(title, case=False)]['title']
    else:
        title_s = mldf[mldf['title2'].str.contains(title, case=False)]['title']

    if len(title_s) > 0:
        idx = title_s[:1].index
    else:
        idx = 0

    tfidf = CountVectorizer()  # , max_df=0.8, min_df=0.2)  ngram_range=(1, 2)
    matrix = tfidf.fit_transform(mldf['search4'])
    cos_sim = cosine_similarity(matrix, matrix)
    idx_list = pd.Series(cos_sim[idx].reshape(-1)).sort_values(ascending = False).index[1:topn+1] # 0번재는 본인. 1~10번째
    df5 = mldf.loc[idx_list,['title','poster_path2', 'overview','title2','overview2']].head()
    return df5.values

# #--- call
# res = my_search_by_genres('Fantasy', 0.97)
# print(res)
#
# #--- call
# res = my_search_by_review('TOY',5)
# print(res)
#
# # --- call
# res = my_search_by_meta('Toy',5)   #Batman Forever
# print(res)

# ######################### 최초 1번만 merge 최종 csv 만든다.###########################################################
def lambda_get_director_def(s):   #[{'job': 'Director', 'name': 'John Lasseter'} , .... ]
    for dict in s:                #{'job': 'Director', 'name': 'John Lasseter'}
        if dict['job'] == 'Director':
            dict['name'] = dict['name'].replace(' ', '')
            return [dict['name'].lower()]  # [john lasseter]
    return ['']

def lambda_get_name_def(s):
    cast_list = []
    for dict in s:
        dict['name'] = dict['name'].replace(' ', '')
        cast_list.append(dict['name'].lower())
    return cast_list[:3]

# def init_frame_merge():
#     mdf = pd.read_csv("./dataset/movies_metadata_2.csv")
#     koko = pd.read_csv("./dataset/movie_metadata_koko.csv")
#     ldf = pd.read_csv("./dataset/links_small.csv")
#     cdf = pd.read_csv("./dataset/credits.csv")
#     kdf = pd.read_csv("./dataset/keywords.csv")
#
#     mdf = mdf.drop(mdf[mdf['id'].str.len() > 6].index, axis=0)
#     mdf['id'] = mdf['id'].astype('int64')
#
#     koko = koko.drop(koko[koko['id'].str.len() > 6].index, axis=0)
#     koko['id'] = koko['id'].astype('int64')
#
#     mdf = pd.merge(mdf, koko, on="id", how="inner")
#     mldf = pd.merge(mdf, ldf, left_on="id", right_on='tmdbId', how="inner")
#     mldf = mldf.merge(cdf, on='id')
#     mldf = mldf.merge(kdf, on='id')  # cast	crew	keywords 추가
#
#     mldf['tagline'] = mldf['tagline'].fillna('')
#     mldf['overview'] = mldf['overview'].fillna('')
#     mldf['view_tag'] = mldf['overview'] + mldf['tagline']
#     mldf = mldf.drop(mldf[mldf['view_tag'].str.len() < 1].index, axis=0)
#
#     mldf = mldf.reset_index(drop=True)
#
#     mldf['cast'] = mldf['cast'].apply(literal_eval)  # 배우
#     mldf['crew'] = mldf['crew'].apply(literal_eval)  # 감독
#     mldf['keywords'] = mldf['keywords'].apply(literal_eval)  # 대표키워드
#     mldf['genres2'] = mldf['genres'].apply(literal_eval)  # 장르
#
#     mldf['director'] = mldf['crew'].apply(lambda_get_director_def)
#     mldf['actor'] = mldf['cast'].apply(lambda_get_name_def)
#     mldf['key'] = mldf['keywords'].apply(lambda_get_name_def)
#     mldf['search4'] = mldf['director'] + mldf['actor'] + mldf['key'] + mldf['genres2']
#     mldf['search4'] = mldf['search4'].astype('str')
#
#     #--------- 포스터가 없으면 삭제
#     idx = mldf[mldf['poster_path2'].isna()].index
#     mldf = mldf.drop(idx, axis=0)
#
#     # --------- 한글 오버뷰 없으면 영문으로 대체
#     mldf.loc[mldf['overview2'].isna(), 'overview2'] = mldf['overview']
#
#     print(  mldf[mldf['overview2'].isna()].index)
#
#     print(mldf.info())
#     mldf.to_csv("./dataset/MOVIE_MERGI.csv", index=False)
#
# # init_frame_merge()  #최초 1번만 실행해서 최종 csv 만든다.



# ####################### 영화 포스터&한글제목&한글리뷰 데이터 추가 크롤링 ###########################

# #---------------------------------------
# # [회원가입]                            https://www.themoviedb.org/
# # [Key를사용해JSON요청]                  https://api.themoviedb.org/3/movie/862?api_key=08cf79d69ff8a9b33dcc4595d9608330
# # [Key를사용해JSON요청_한국어버전]         https://api.themoviedb.org/3/movie/popular?api_key=334075ba2b018bdb3e91bc504676f9b9&language=ko-KR
# # [JSON내 poster_path를 통한 이미지 보기] https://image.tmdb.org/t/p/original/7G9915LfUQ2lVfwMEEhDsn3kT4B.jpg
# #----------------------------------------
# def get_json_imdb_image(id=862) :
#     # url = f"https://api.themoviedb.org/3/movie/{uid}?api_key=08cf79d69ff8a9b33dcc4595d9608330"
#     url = f"https://api.themoviedb.org/3/movie/{id}?api_key=334075ba2b018bdb3e91bc504676f9b9&language=ko-KR"
#     dict = requests.get(url).json()
#     poster_path = dict["poster_path"]
#     title = dict["title"]
#     overview = dict["overview"]
#
#     print(poster_path, title, overview)
# # get_json_imdb_image(862)  #634649
#
#
# def get_dump_imdb_image() :
#     df = pd.read_csv("./dataset/movies_metadata_2.csv")
#     id_list = df["id"].values
#     poster_list = []
#     try:
#         for id in id_list:
#             dict = {}
#             # url = f"https://api.themoviedb.org/3/movie/{id}?api_key=08cf79d69ff8a9b33dcc4595d9608330"
#             url = f"https://api.themoviedb.org/3/movie/{id}?api_key=334075ba2b018bdb3e91bc504676f9b9&language=ko-KR"
#             # print(id, url)
#             try:
#                #---------파싱 -----------
#                print(id)
#                json_res = requests.get(url).json()
#                # poster_path = json_res["belongs_to_collection"]["poster_path"]
#                try:
#                    poster_path = json_res["poster_path"]
#                    title = json_res["title"]
#                    overview = json_res["overview"]
#                except Exception as e:
#                    poster_path = json_res["poster_path"]
# 
#                dict["id"] = id
#                dict["poster_path2"] = poster_path
#                dict["title2"] = title
#                dict["overview2"] = overview
# 
#                poster_list.append(dict)
#             except Exception as e:
#                 print("error:", id)
#                 continue
#     except Exception as e:
#         print("수집 중 에러 발생")
#     finally:
#         poster_df = pd.DataFrame(poster_list)
#         poster_df.to_csv("./dataset/movie_metadata_koko.csv", index=False)
#     print("---done---")
# ### get_dump_imdb_image() 절대열지말것.....몇시간 걸림
