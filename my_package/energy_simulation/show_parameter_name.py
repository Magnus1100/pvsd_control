from fmpy import *

# 加载 FMU 文件
fmu_filename = 'OverhangTest241104.idf'

# 使用 instantiate() 方法实例化 FMU
fmu = instantiate_fmu(fmu_filename)

# 获取模型描述
model_description = fmu['modelDescription']

# 打印模型描述的信息
print("Model Name:", model_description.name)
print("Model Version:", model_description.version)

# 获取输入变量
input_variables = [var for var in model_description.modelVariables if var['causality'] == 'input']

# 打印输入变量名称及其因果关系
for var in input_variables:
    print(f"Input Variable Name: {var['name']}, Causality: {var['causality']}")