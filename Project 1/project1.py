import pandas as pd


def co_occurence_characters():
    with open('book.txt') as text:
        text_book = text.read()
        chapters = text_book.split('CHAPTER')


co_occurence_characters()