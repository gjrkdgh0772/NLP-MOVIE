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


####################### 영화 포스터&한글제목&한글리뷰 데이터 추가 크롤링 ###########################

#---------------------------------------
# [회원가입]                            https://www.themoviedb.org/
# [Key를사용해JSON요청]                  https://api.themoviedb.org/3/movie/862?api_key=08cf79d69ff8a9b33dcc4595d9608330
# [Key를사용해JSON요청_한국어버전]         https://api.themoviedb.org/3/movie/popular?api_key=334075ba2b018bdb3e91bc504676f9b9&language=ko-KR
# [JSON내 poster_path를 통한 이미지 보기] https://image.tmdb.org/t/p/original/7G9915LfUQ2lVfwMEEhDsn3kT4B.jpg
#----------------------------------------
def get_json_imdb_image(id=862) :
    # url = f"https://api.themoviedb.org/3/movie/{uid}?api_key=08cf79d69ff8a9b33dcc4595d9608330"
    url = f"https://api.themoviedb.org/3/movie/{id}?api_key=334075ba2b018bdb3e91bc504676f9b9&language=ko-KR"
    dict = requests.get(url).json()
    poster_path = dict["poster_path"]
    title = dict["title"]
    overview = dict["overview"]

    print(poster_path, title, overview)
get_json_imdb_image(862)  #634649


def get_dump_imdb_image() :
    df = pd.read_csv("./dataset/movies_metadata_2.csv")
    id_list = df["id"].values
    poster_list = []
    try:
        for id in id_list:
            dict = {}
            # url = f"https://api.themoviedb.org/3/movie/{id}?api_key=08cf79d69ff8a9b33dcc4595d9608330"
            url = f"https://api.themoviedb.org/3/movie/{id}?api_key=334075ba2b018bdb3e91bc504676f9b9&language=ko-KR"
            # print(id, url)
            try:
               #---------파싱 -----------
               print(id)
               json_res = requests.get(url).json()
               poster_path = json_res["belongs_to_collection"]["poster_path"]
               try:
                   poster_path = json_res["poster_path"]
                   title = json_res["title"]
                   overview = json_res["overview"]
               except Exception as e:
                   poster_path = json_res["poster_path"]

               dict["id"] = id
               dict["poster_path2"] = poster_path
               dict["title2"] = title
               dict["overview2"] = overview

               poster_list.append(dict)
            except Exception as e:
                print("error:", id)
                continue
    except Exception as e:
        print("수집 중 에러 발생")
    finally:
        poster_df = pd.DataFrame(poster_list)
        poster_df.to_csv("./dataset/movie_metadata_koko.csv", index=False)
    print("---done---")
# ### get_dump_imdb_image() 절대열지말것.....몇시간 걸림
