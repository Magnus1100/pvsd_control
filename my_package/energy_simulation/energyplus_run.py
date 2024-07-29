import subprocess


def run_energyplus_simulation(energyplus_exe, epw_file, idf_file, output_dir):
    """
    运行 EnergyPlus 模拟。

    Args:
    - energyplus_exe (str): EnergyPlus 可执行文件的路径。
    - epw_file (str): EPW 文件的路径。
    - idf_file (str): IDF 文件的路径。
    - output_dir (str): 输出目录的路径。

    Returns:
    - result (subprocess.CompletedProcess): 包含命令执行结果的对象。
    """
    # 构建 EnergyPlus 的命令行参数
    command = [
        energyplus_exe,
        "-w", epw_file,
        "-d", output_dir,
        idf_file
    ]

    # 打印构建的命令来检查路径
    print("Command to be run:", " ".join(command))

    # 使用 shell=True
    result = subprocess.run(" ".join(command), capture_output=True, text=True, shell=True)

    # 打印输出和错误信息来检查潜在问题
    print(result.stdout)
    print(result.stderr)

    return result


if __name__ == "__main__":
    # 调用
    my_energyplus_exe = "energyplus"
    my_epw_file = (r'D:\pythonProject\pythonProject\.venv\my_package\source\epw_file\CHN_Guangdong.Shenzhen'
                   r'.594930_SWERA.epw')

    idf_version = "0705_V2"
    my_output_dir = fr'D:\pythonProject\pythonProject\.venv\my_package\source\energy_result\result_{idf_version}'
    my_idf_file1 = fr'D:\pythonProject\pythonProject\.venv\my_package\energy_simulation\idf_test_{idf_version}.idf'
    my_idf_file2 = (r'D:\pythonProject\pythonProject\.venv\my_package\source\rhino_energy_model\test__\openstudio\run'
                    r'/in.idf')

    my_result = run_energyplus_simulation(my_energyplus_exe, my_epw_file, my_idf_file1, my_output_dir)
    print(my_result)
