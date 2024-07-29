import eppy
from eppy.modeleditor import IDF
from enum import Enum
import os


def get_unique_filename(base_path, base_name, extension):
    """
    获取唯一文件名，如果文件夹中已有同名文件，就在文件名后加上数字递增。

    参数:
    - base_path (str): 文件夹路径。
    - base_name (str): 基础文件名。
    - extension (str): 文件扩展名。

    返回:
    - str: 唯一文件名。
    """
    n = 1
    unique_filename = f"{base_name}{n}.{extension}"
    while os.path.exists(os.path.join(base_path, unique_filename)):
        n += 1
        unique_filename = f"{base_name}{n}.{extension}"
    return unique_filename


def add_context_to_idf(idf_file_path, adding_key, adding_type, output_base_path):
    """
    读取IDF文件，添加自定义内容，并保存修改后的IDF文件。
    """
    # idf路径
    idf_template_path = r'C:\EnergyPlusV24-1-0\Energy+.idd'

    # 输出路径
    unique_filename = get_unique_filename(output_base_path, "in_V", "idf")
    output_idf_file_path = os.path.join(output_base_path, unique_filename)

    # 检查IDF模板文件是否存在
    if not os.path.isfile(idf_template_path):
        raise FileNotFoundError(f"IDF template file not found at: {idf_template_path}")

    # 设置EnergyPlus的idf文件模板路径
    IDF.setiddname(idf_template_path)

    # 读取IDF文件
    idf = IDF(idf_file_path)

    # 添加外部装置
    idf.newidfobject(
        adding_key,
        **adding_type.value
    )

    # 保存修改后的IDF文件
    idf.save(output_idf_file_path)

    print(f"Modified IDF file saved as {output_idf_file_path}")


def main():
    # 输入
    my_idf_file_path = (r'D:\pythonProject\pythonProject\.venv\my_package\source\energy_model\test__\openstudio\run'
                        r'\in.idf')
    my_shading_key = "Shading:Building:Detailed"
    my_shading_type1 = ShadingType.TYPE1  # 添加的内容
    my_output_path = r'D:\pythonProject\pythonProject\.venv\my_package\source\energy_model\test__\openstudio\run'

    # 把内容加入到idf中
    add_context_to_idf(my_idf_file_path, my_shading_key, my_shading_type1, my_output_path)


if __name__ == "__main__":
    main()
