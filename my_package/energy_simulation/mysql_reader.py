
sql_file_path = r'E:\02_research\05_data\OverhangTest241104\OverhangTest241104.sql'
# 读取 SQL 文件内容
with open(sql_file_path, 'r', encoding='gbk') as file:
    sql_content = file.read()
    print(sql_content)  # 或者保存到其他文件，或进行处理
