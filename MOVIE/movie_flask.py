import json

import pandas as pd
from flask import Flask, make_response, jsonify, request, render_template
import cx_Oracle
import sqlalchemy as sa
# from flask_cors import CORS, cross_origin
import requests
from bs4 import BeautifulSoup

# ------------------------------------------------
import NLP.movie_lkh.movie_search_util as movie
# ------------------------------------------------

app = Flask(__name__, template_folder="template", static_folder="static")
# CORS(app)


@app.route('/')
def index():
    return render_template("index.html",)

@app.route('/search', methods=['POST', 'GET'])
def search():
    # language = 'eng'  # 'eng', 'kor'
    search_gubun = request.args.get('search_gubun')
    search_str   = request.args.get('search_str')
    redirect_url = "result.html"
    print(search_gubun, search_str)

    res = []
    if search_gubun == 'genres':
       res = movie.my_search_by_genres(search_str)
    elif search_gubun == 'story':
        res = movie.my_search_by_review(search_str)
    elif search_gubun == 'actor':
        res = movie.my_search_by_meta(search_str)
    else:
        redirect_url = "index.html"
    print(res)
    return render_template(redirect_url, MY_INFO=res)

# @app.route('/result')
# def result():
#     return render_template("result.html", )


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8088)