from collections import Counter
import re


def calculate_char_score(query, text):
    # 计算字符频率
    query_chars = Counter(query.lower())
    text_chars = Counter(text.lower())

    # 计算共同字符的总数
    common_chars = query_chars & text_chars
    score = sum(common_chars.values())

    return score


# 示例问题集
questions = [
    "如何提高编程效率？",
    "什么是机器学习？",
    "如何学习Python编程？",
    "推荐几本关于数据科学的书籍。",
    "如何选择适合自己的编程语言？"
]

# 用户输入
user_input = input("请输入你的查询: ")

# 计算用户输入与每个问题之间的字符相似度得分
scored_questions = [(question, calculate_char_score(user_input, question)) for question in questions]

# 根据得分对问题进行排序（降序）
scored_questions.sort(key=lambda x: x[1], reverse=True)

# 输出得分最高的前两个问题
print("得分最高的前两个问题是:")
for question, score in scored_questions[:2]:
    print(f"{question}")
