def make_question(label):
    if label == 1:
        return "한글 맞춤법, 띄어쓰기 오류가 발생한 부분은?"
    elif label == 2:
        return "단어 선택 오류가 발생한 부분은?"
    elif label == 3:
        return "비문이 발생한 부분은?"
    elif label == 4:
        return "미완성 또는 불완전한 문장이 발생한 부분은?"
    elif label == 5:
        return "키워드 또는 중요 내용 오류가 발생한 부분은?"
    elif label == 6:
        return "유사한 내용 반복 오류가 발생한 부분은?"
    else :
        return None